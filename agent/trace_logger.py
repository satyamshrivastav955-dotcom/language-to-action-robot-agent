import json
import os
import time
from typing import Dict, List, Optional
from datetime import datetime
import threading

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.settings import LOGS_DIR


def _json_safe(value, _depth: int = 0):
    """Coerce a log payload into something json.dumps accepts.

    Verification results carry numpy scalars/arrays, and callers occasionally
    pass whole frame buffers. Large arrays are summarised rather than dumped so
    a trace file stays readable and small.
    """
    if _depth > 6:
        return "<max depth>"

    if isinstance(value, dict):
        return {str(k): _json_safe(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 64:
            return f"<{type(value).__name__} of {len(value)} items>"
        return [_json_safe(v, _depth + 1) for v in value]

    try:
        import numpy as _np
        if isinstance(value, _np.ndarray):
            if value.size > 64:
                return f"<ndarray shape={value.shape}>"
            return _json_safe(value.tolist(), _depth + 1)
        if isinstance(value, _np.generic):
            return value.item()
    except ImportError:
        pass

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


class TraceLogger:
    """Structured trace of a run.

    Format: newline-delimited JSON (JSONL), one entry per line, written as the
    run proceeds. A previous version streamed JSONL *and* then had save()
    overwrite the same path with a pretty-printed JSON array, so a log's format
    depended on whether the run reached the end. Entries are now appended in one
    format only; `save()` flushes and returns the path, and `save_array()` is
    available when a single JSON document is explicitly wanted.
    """

    def __init__(self, log_file: Optional[str] = None, console_output: bool = True):
        self.log_file = log_file
        self.console_output = console_output
        self.entries: List[Dict] = []
        self.session_start: float = time.time()
        self._lock = threading.Lock()
        self._new_entry_callbacks: List[callable] = []

        if log_file is None:
            log_file = os.path.join(LOGS_DIR, f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")

        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.log_file = log_file
    
    def log(self, level: str, message: str, subtask_id: Optional[int] = None,
            attempt: Optional[int] = None, **metadata):
        entry = {
            "timestamp": time.time() - self.session_start,
            "datetime": datetime.now().isoformat(),
            "level": level,
            "message": message,
            "subtask_id": subtask_id,
            "attempt": attempt,
            **metadata
        }
        
        with self._lock:
            self.entries.append(entry)
        
        self._write_entry(entry)
        self._notify_callbacks(entry)
        
        if self.console_output:
            print(self.format_entry(entry))
    
    def log_instruction(self, instruction: str):
        self.log("INFO", f"Instruction: '{instruction}'", type="instruction_start")
    
    def log_plan(self, plan: Dict, used_fallback: bool = False):
        subtask_count = len(plan.get("subtasks", []))
        source = "heuristic fallback (LLM unavailable)" if used_fallback else "LLM"
        self.log("WARNING" if used_fallback else "INFO",
                 f"Plan decomposed into {subtask_count} subtasks via {source}",
                 type="plan", plan=plan, plan_source="fallback" if used_fallback else "llm")

    def log_policy_disclosure(self, policy_info: Dict):
        """Record up front how outcomes are produced.

        Emitted at the top of every run so a scripted result can never be read
        as a policy result further down the trace.
        """
        if policy_info.get("outcomes_are_scripted"):
            self.log("WARNING",
                     f"Policy '{policy_info.get('name')}' produces SCRIPTED outcomes - "
                     f"{policy_info.get('note', '')}",
                     type="policy_disclosure", policy=policy_info)
        else:
            self.log("INFO",
                     f"Policy '{policy_info.get('name')}' driving rollouts",
                     type="policy_disclosure", policy=policy_info)

    def log_subtask_start(self, subtask: Dict):
        self.log(
            "INFO",
            f"Starting subtask {subtask.get('id', '?')}: {subtask.get('action', '?')} {subtask.get('object', '?')} -> {subtask.get('target', '?')}",
            subtask_id=subtask.get("id"),
            type="subtask_start"
        )
    
    def log_attempt(self, subtask_id: int, attempt: int, prompt: str, retry_params: Dict = None):
        self.log(
            "INFO",
            f"Subtask {subtask_id} - Attempt {attempt}",
            subtask_id=subtask_id,
            attempt=attempt,
            prompt=prompt,
            retry_params=retry_params or {},
            type="attempt_start"
        )

    def log_attempt_result(self, subtask_id: int, attempt: int, success: bool,
                            duration: float, reason: str = "", verification: Dict = None,
                            outcome_source: str = None):
        level = "SUCCESS" if success else "WARNING"
        msg = f"Attempt {attempt}: {'SUCCESS' if success else 'FAILED'} ({duration:.2f}s)"
        if outcome_source == "scripted":
            msg += " [scripted outcome]"
        if not success and reason:
            msg += f" - {reason}"

        self.log(level, msg, subtask_id=subtask_id, attempt=attempt,
                 type="attempt_result", success=success, reason=reason,
                 verification=verification, outcome_source=outcome_source)
    
    def log_retry(self, subtask_id: int, attempt: int, diagnosis: str):
        self.log(
            "WARNING",
            f"Retrying... (diagnosis: {diagnosis})",
            subtask_id=subtask_id,
            attempt=attempt,
            diagnosis=diagnosis,
            type="retry"
        )
    
    def log_subtask_complete(self, subtask: Dict, total_attempts: int, total_time: float):
        self.log(
            "SUCCESS",
            f"Subtask {subtask.get('id')} completed in {total_attempts} attempt(s), {total_time:.2f}s",
            subtask_id=subtask.get("id"),
            total_attempts=total_attempts,
            type="subtask_complete"
        )
    
    def log_subtask_failed(self, subtask: Dict, total_attempts: int, reason: str):
        self.log(
            "ERROR",
            f"Subtask {subtask.get('id')} FAILED after {total_attempts} attempts — {reason}. Moving to next subtask.",
            subtask_id=subtask.get("id"),
            type="subtask_failed",
            reason=reason
        )
    
    def log_task_complete(self, successful: int, failed: int, total_time: float):
        self.log(
            "SUCCESS" if failed == 0 else "WARNING",
            f"Task complete. {successful} successful, {failed} failed. Total time: {total_time:.2f}s",
            type="task_complete"
        )
    
    def log_clarification(self, question: str):
        """Record that the agent asked rather than guessed."""
        self.log("WARNING", f"Clarification needed: {question}",
                 type="clarification_requested", question=question)

    def log_error(self, message: str, error: Exception = None, **metadata):
        self.log("ERROR", message, type="error", error=str(error) if error else None, **metadata)
    
    def log_state_transition(self, transition: Dict):
        event = transition.get("event", "unknown")
        metadata = transition.get("metadata", {})
        
        if event == "object_moved":
            self.log("DEBUG", f"Object {metadata.get('object')} moved to {metadata.get('new_position')}",
                     type="object_update")
    
    def format_entry(self, entry: Dict, colour: bool = True) -> str:
        timestamp = f"[{entry.get('timestamp', 0):.2f}s]"
        level = entry.get("level", "INFO")
        subtask = f"[subtask {entry['subtask_id']}]" if entry.get("subtask_id") else ""
        attempt = f" (attempt {entry['attempt']})" if entry.get("attempt") else ""

        message = entry.get("message", "")

        level_colors = {
            "SUCCESS": "\033[92m",
            "ERROR": "\033[91m",
            "WARNING": "\033[93m",
            "INFO": "\033[94m",
            "DEBUG": "\033[90m",
        }
        reset = "\033[0m"

        # ANSI codes render as literal escape garbage in a Gradio textbox, so
        # any non-terminal consumer passes colour=False.
        color = level_colors.get(level, "") if colour else ""
        if not colour:
            reset = ""

        return f"{timestamp} {color}{level}{reset}{subtask}{attempt}: {message}"

    def _write_entry(self, entry: Dict):
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(_json_safe(entry)) + "\n")
        except Exception as e:
            if self.console_output:
                print(f"Failed to write log: {e}")
    
    def _notify_callbacks(self, entry: Dict):
        for callback in self._new_entry_callbacks:
            try:
                callback(entry)
            except Exception:
                pass
    
    def add_entry_callback(self, callback: callable):
        self._new_entry_callbacks.append(callback)
    
    def get_recent_entries(self, count: int = 20) -> List[Dict]:
        with self._lock:
            return self.entries[-count:]
    
    def get_all_entries(self) -> List[Dict]:
        with self._lock:
            return list(self.entries)
    
    def save(self, path: Optional[str] = None) -> str:
        """Finalise the trace and return its path.

        Entries were already streamed as JSONL by `_write_entry`, so the default
        case rewrites nothing - it just reports where the log is. Passing an
        explicit `path` writes a JSONL copy there.
        """
        if path is None or os.path.abspath(path) == os.path.abspath(self.log_file):
            return self.log_file

        with self._lock:
            entries = list(self.entries)

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(_json_safe(entry)) + "\n")
        return path

    def save_array(self, path: Optional[str] = None) -> str:
        """Write the trace as a single pretty-printed JSON array.

        Separate from `save()` on purpose: mixing the two formats in one file
        was the original bug.
        """
        save_path = path or (os.path.splitext(self.log_file)[0] + ".json")
        with self._lock:
            entries = [_json_safe(e) for e in self.entries]
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
        return save_path

    def get_readable_log(self, colour: bool = True) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("EXECUTION TRACE LOG")
        lines.append("=" * 60)

        for entry in self.entries:
            lines.append(self.format_entry(entry, colour=colour))

        lines.append("=" * 60)

        return "\n".join(lines)
    
    def reset(self):
        with self._lock:
            self.entries = []
            self.session_start = time.time()


def create_logger(log_file: Optional[str] = None, console_output: bool = True) -> TraceLogger:
    return TraceLogger(log_file=log_file, console_output=console_output)


if __name__ == "__main__":
    print("Testing Trace Logger...")
    
    logger = create_logger(console_output=True)
    
    logger.log_instruction("Put the red block in the bin, then stack blue on green")
    
    test_plan = {
        "subtasks": [
            {"id": 1, "action": "pick_and_place", "object": "red block", "target": "bin"},
            {"id": 2, "action": "stack", "object": "blue block", "target": "green block"}
        ]
    }
    logger.log_plan(test_plan)
    
    logger.log_subtask_start(test_plan["subtasks"][0])
    logger.log_attempt(1, 1, "Pick up the red block and place it in the bin")
    logger.log_attempt_result(1, 1, False, 1.5, "Grasp failed - object not picked up")
    logger.log_retry(1, 2, "Pre-grasp approach failed")
    
    logger.log_attempt(1, 2, "Pick up the red block and place it in the bin")
    logger.log_attempt_result(1, 2, True, 0.8)
    logger.log_subtask_complete({"id": 1}, 2, 2.3)
    
    logger.log_subtask_start(test_plan["subtasks"][1])
    logger.log_attempt(2, 1, "Pick up the blue block and stack on green block")
    logger.log_attempt_result(2, 1, True, 1.2)
    logger.log_subtask_complete({"id": 2}, 1, 1.2)
    
    logger.log_task_complete(2, 0, 3.5)
    
    print("\n" + "=" * 60)
    print("READABLE OUTPUT:")
    print("=" * 60)
    print(logger.get_readable_log())
    
    saved_path = logger.save()
    print(f"\nLog saved to: {saved_path}")
    
    print("\nTrace Logger test complete!")
