"""Single orchestrator shared by the CLI (run_agent.py) and the Gradio UI.

Both front-ends previously carried their own copy of the plan/execute/verify/
retry loop. The two copies drifted: the dashboard logged the wrong prompt,
never wired its sliders, could raise UnboundLocalError on an empty retry range,
and its Stop button set a flag nothing read. There is now one loop here and the
front-ends only render its events.

`run()` is a generator of ProgressEvent so a UI can stream partial state;
`execute_instruction()` drains it and returns the final result dict for the CLI.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional


from agent.planner import create_planner
from agent.executor import Executor
from agent.verifier import NullVerifier, PerceptionVerifier, Verifier
from agent.retry_controller import RetryController
from env.faults import attempt_rng
from perception import PerceptionSystem
from perception.overlay import annotate
from agent.state_manager import TaskStateManager
from agent.trace_logger import TraceLogger
from env.so101_env import create_environment
from configs.settings import (
    MAX_RETRIES_PER_SUBTASK,
    MAX_STEPS_PER_ROLLOUT,
)


@dataclass
class ProgressEvent:
    """One observable step of a run. Front-ends render these; they hold no HTML."""

    status: str
    subtasks: List[Dict] = field(default_factory=list)
    log_text: str = ""
    video_path: Optional[str] = None
    summary: Optional[Dict] = None
    error: Optional[str] = None
    # Set when the planner declined to guess and asked a question instead.
    clarification: Optional[str] = None
    # Overhead camera view with detections, as the verifier saw it (RGB array).
    camera_image: Optional[object] = None


class RobotTaskAgent:

    def __init__(self, use_mock_policy: bool = False,
                 console_output: bool = True, env=None, planner=None,
                 verifier_mode: str = "perception", retry_strategy: str = "corrected",
                 perception_mode: str = "segmentation", rewind_on_retry: bool = False,
                 record_frames: bool = True, fault_profile: str = "nominal", seed: int = 0,
                 record_oracle: bool = True, estimator=None, retry_alpha: float = 0.85):
        """
        verifier_mode:   "perception" (camera-based), "oracle" (simulator truth), "none".
        retry_strategy:  "corrected" (use the measured error) or "blind".
        perception_mode: "segmentation" / "vlm" (act on camera estimates) or "oracle".
        rewind_on_retry: restore the pre-subtask world before a retry. Off by default:
                         a real robot cannot undo a failed attempt.
        record_oracle:   also score each attempt against simulator truth. Evaluation
                         logging only; the agent never reads it.
        """
        self.env = env if env is not None else create_environment()
        self.seed = seed
        self.fault_profile = fault_profile
        self.rewind_on_retry = rewind_on_retry
        self.record_oracle = record_oracle
        physics = bool(getattr(self.env, "is_physics_backed", False)) and not use_mock_policy

        self.planner = planner if planner is not None else create_planner()

        self.perception = None
        if physics:
            self.perception = PerceptionSystem(self.env, estimator=estimator, seed=seed)
        acting_perception = self.perception if perception_mode != "oracle" else None
        self.executor = Executor(env=self.env, use_mock_policy=use_mock_policy,
                                 perception=acting_perception, record_frames=record_frames)

        if verifier_mode == "none":
            self.verifier = NullVerifier()
        elif verifier_mode == "perception" and self.perception is not None:
            self.verifier = PerceptionVerifier(self.env, self.perception)
        else:
            self.verifier = Verifier(self.env)
        self.verifier_mode = verifier_mode if (verifier_mode != "perception" or self.perception) else "oracle"
        self._oracle = Verifier(self.env) if (record_oracle and physics) else None

        self.retry_controller = RetryController(strategy=retry_strategy, alpha=retry_alpha)
        self.state_manager = TaskStateManager()
        self.logger = TraceLogger(console_output=console_output)

        self.max_retries = MAX_RETRIES_PER_SUBTASK
        self.max_steps = MAX_STEPS_PER_ROLLOUT

        self.is_running = False
        self._stop_requested = False

        self._setup_callbacks()

    def _setup_callbacks(self):
        self.state_manager.add_transition_callback(
            lambda t: self.logger.log_state_transition(t)
        )

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def request_stop(self):
        """Ask the run to end at the next attempt boundary.

        The loop checks this between attempts and between subtasks, so a Stop
        press lands within one rollout instead of being silently ignored.
        """
        self._stop_requested = True

    # Kept as an alias: the dashboard's old button called stop_execution().
    stop_execution = request_stop

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def execute_instruction(self, instruction: str,
                            max_retries: Optional[int] = None,
                            max_steps: Optional[int] = None,
                            save_videos: bool = True,
                            block_positions: Optional[Dict] = None,
                            reset_world: bool = True) -> Dict:
        """Blocking run. Returns the final result dict (used by the CLI)."""
        final = None
        for event in self.run(instruction, max_retries, max_steps, save_videos, block_positions,
                              reset_world):
            final = event

        if final is None:
            return {"success": False, "error": "No events produced", "timestamp": time.time()}

        if final.clarification:
            return {"success": False, "clarification": final.clarification,
                    "timestamp": time.time()}

        if final.error:
            return {"success": False, "error": final.error, "timestamp": time.time()}

        summary = final.summary or {}
        return {
            "success": summary.get("failed", 1) == 0 and summary.get("stopped_early") is not True,
            "timestamp": time.time(),
            "summary": summary,
            "results": summary.get("results", []),
            "log_path": summary.get("log_path"),
        }

    def run(self, instruction: str,
            max_retries: Optional[int] = None,
            max_steps: Optional[int] = None,
            save_videos: bool = True,
            block_positions: Optional[Dict] = None,
            reset_world: bool = True) -> Generator[ProgressEvent, None, None]:
        """Stream the full plan/execute/verify/retry loop as ProgressEvents."""

        max_retries = max_retries or self.max_retries
        max_steps = max_steps or self.max_steps

        self.is_running = True
        self._stop_requested = False
        start_time = time.time()

        self.logger.reset()
        self.retry_controller.reset()
        self.state_manager.reset()

        self.logger.log_instruction(instruction)
        self.logger.log_policy_disclosure(self.executor.describe_policy())

        if reset_world:
            self.env.reset(block_positions)
        if hasattr(self.env, "set_faults"):
            self.env.set_faults(self.fault_profile)

        object_positions = {}
        if self.perception is not None:
            for name, det in self.perception.perceive().items():
                object_positions[name] = det.xyz.tolist()
        else:
            for obj in self.env.object_names:
                object_positions[obj] = self.env.get_object_pose(obj).tolist()

        yield ProgressEvent(status="Planning...", log_text=self._log_text())

        try:
            plan = self.planner.decompose(instruction)
            self.logger.log_plan(plan, used_fallback=self.planner.used_fallback)
        except Exception as e:
            self.logger.log_error("Planning failed", e)
            self.is_running = False
            yield ProgressEvent(status=f"Planning failed: {e}",
                                log_text=self._log_text(),
                                error=f"Planning error: {e}")
            return

        # The planner may decline to guess and ask a question instead. Surface
        # it and stop; the caller re-runs with the answer folded in.
        if plan.get("clarification_needed"):
            question = plan.get("question", "Which object did you mean?")
            self.logger.log_clarification(question)
            self.is_running = False
            yield ProgressEvent(status="Clarification needed",
                                log_text=self._log_text(),
                                clarification=question)
            return

        subtasks = plan.get("subtasks", [])
        if not subtasks:
            self.logger.log_error("No subtasks generated")
            self.is_running = False
            yield ProgressEvent(status="No subtasks generated",
                                log_text=self._log_text(),
                                error="No subtasks generated")
            return

        self.state_manager.initialize(plan, instruction, object_positions)

        view = [
            {
                "id": st.get("id", i + 1),
                "action": st.get("action"),
                "object": st.get("object"),
                "target": st.get("target"),
                "status": "pending",
                "attempts": 0,
            }
            for i, st in enumerate(subtasks)
        ]

        yield ProgressEvent(status=f"Plan: {len(subtasks)} subtask(s)",
                            subtasks=_copy_view(view), log_text=self._log_text())

        results = []
        total_retries = 0
        stopped_early = False

        for index, subtask in enumerate(subtasks):
            if self._stop_requested:
                stopped_early = True
                break

            for event in self._run_subtask(subtask, index, view, len(subtasks),
                                           max_retries, max_steps, save_videos):
                if isinstance(event, ProgressEvent):
                    yield event
                else:
                    results.append(event)
                    total_retries += event["retries"]

            if self._stop_requested and view[index]["status"] not in ("completed", "failed"):
                stopped_early = True
                break

        # Deliberately no env.reset() here: the final world state stays
        # readable after the run (the dashboard and any post-run inspection
        # see the scene the instruction actually produced).
        elapsed_time = time.time() - start_time

        summary = self.state_manager.get_summary()
        summary["total_retries"] = total_retries
        summary["elapsed_time"] = elapsed_time
        summary["results"] = results
        summary["stopped_early"] = stopped_early
        summary["policy"] = self.executor.describe_policy()

        if stopped_early:
            self.logger.log("WARNING", "Run stopped by user request", type="stopped")

        self.logger.log_task_complete(summary["successful"], summary["failed"], elapsed_time)

        log_path = self.logger.save()
        summary["log_path"] = log_path

        self.is_running = False

        if stopped_early:
            status = f"Stopped. {summary['successful']}/{summary['total_subtasks']} succeeded"
        else:
            status = f"Complete. {summary['successful']}/{summary['total_subtasks']} succeeded"

        yield ProgressEvent(status=status, subtasks=_copy_view(view),
                            log_text=self._log_text(), summary=summary)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run_subtask(self, subtask: Dict, index: int, view: List[Dict],
                     total_subtasks: int, max_retries: int, max_steps: int,
                     save_videos: bool):
        """Yields ProgressEvents, then finally one plain dict: the result row."""

        subtask_id = subtask.get("id", index + 1)
        action = subtask.get("action", "pick_and_place")
        obj = subtask.get("object", "unknown")
        target = subtask.get("target", "unknown")

        self.logger.log_subtask_start(subtask)

        # Sets started_at, without which get_summary()["total_time"] stays 0.0.
        self.state_manager.start_subtask(subtask)

        view[index]["status"] = "in_progress"
        yield ProgressEvent(
            status=f"Subtask {subtask_id}/{total_subtasks}: {action} {obj} -> {target}",
            subtasks=_copy_view(view), log_text=self._log_text())

        success = False
        execution_result = None
        verification_result = None
        attempts = 0
        video_path = None

        # Snapshot the world as this subtask found it. A failed attempt is
        # rewound to THIS state - not to a factory-reset scene - so retries
        # do not destroy the work of earlier subtasks in the same instruction.
        pre_subtask_state = self.env.save_state() if self.rewind_on_retry else None
        attempt_log: List[Dict] = []

        for attempt in range(1, max_retries + 1):
            if self._stop_requested:
                break

            attempts = attempt
            view[index]["attempts"] = attempt

            if attempt > 1 and pre_subtask_state is not None:
                self.env.restore_state(pre_subtask_state)
            if hasattr(self.env, "seed_attempt"):
                self.env.seed_attempt(attempt_rng(self.seed, index, attempt))

            # Diagnosis from the previous failure feeds the next attempt.
            retry_params = self.retry_controller.get_adjusted_parameters(subtask, attempt)

            prompt = self.executor.create_subgoal_prompt(action, obj, target)
            self.logger.log_attempt(subtask_id, attempt, prompt, retry_params=retry_params)

            yield ProgressEvent(status=f"Subtask {subtask_id} - attempt {attempt}/{max_retries}",
                                subtasks=_copy_view(view), log_text=self._log_text())

            execution_result = self.executor.run_subtask(
                subtask,
                max_steps=max_steps,
                save_video=save_videos,
                video_name=f"subtask_{subtask_id}_attempt_{attempt}",
                retry_params=retry_params,
            )

            verification_result = self.verifier.verify(subtask, execution_result)
            # What the agent believes happened: moved-ness comes from what it
            # observed, not from simulator state.
            if verification_result.get("object_moved") is not None:
                execution_result["object_moved"] = verification_result["object_moved"]
            elif execution_result.get("perception_failure"):
                execution_result["object_moved"] = False
            correction = self.retry_controller.observe(subtask, execution_result, verification_result)

            oracle_ok = None
            if self._oracle is not None:
                truth = self._oracle.verify(subtask, {**execution_result, "target_pos_est": None})
                oracle_ok = bool(truth.get("success"))
            attempt_log.append({
                "attempt": attempt,
                "verdict": bool(verification_result.get("success", False)),
                "oracle_success": oracle_ok,
                "observed_by": verification_result.get("observed_by"),
                "reason": verification_result.get("reason", ""),
                "error_vector": verification_result.get("error_vector"),
                "object_position": verification_result.get("object_position"),
                "grasp_detected": execution_result.get("grasp_detected"),
                "perception_failure": execution_result.get("perception_failure", False),
                "correction_applied": (retry_params or {}).get("place_correction"),
                "correction_next": correction,
                "video_path": execution_result.get("video_path"),
            })

            view[index]["attempt_log"] = [
                {k: a[k] for k in ("attempt", "verdict", "oracle_success", "observed_by")} for a in attempt_log]
            camera_image = None
            if self.perception is not None and self.perception.last_rgbd is not None                     and verification_result.get("observed_by") == "perception":
                camera_image = annotate(self.perception.last_rgbd, self.perception.last,
                                        verification_result.get("target_position"), obj)

            success = verification_result.get("success", False)
            duration = execution_result.get("duration", 0)
            reason = verification_result.get("reason", "")
            video_path = execution_result.get("video_path") or video_path

            self.logger.log_attempt_result(subtask_id, attempt, success, duration,
                                           reason, verification_result,
                                           outcome_source=execution_result.get("outcome_source"))

            yield ProgressEvent(
                status=f"Subtask {subtask_id} - attempt {attempt}: "
                       f"{'SUCCESS' if success else 'FAILED'}",
                subtasks=_copy_view(view), log_text=self._log_text(),
                video_path=video_path, camera_image=camera_image)

            if success:
                break

            self.state_manager.increment_retry(subtask)

            # Ask the controller rather than assuming the range bound decides.
            if not self.retry_controller.should_retry(
                    {"id": subtask_id, "attempts": attempt}, verification_result,
                    max_retries=max_retries):
                break

            diagnosis = self.retry_controller.diagnose(
                subtask, execution_result, verification_result)
            self.logger.log_retry(subtask_id, attempt + 1, diagnosis)

        if success:
            self.state_manager.mark_complete(subtask, execution_result, verification_result)
            self.logger.log_subtask_complete(
                subtask, attempts, execution_result.get("duration", 0) if execution_result else 0)
            view[index]["status"] = "completed"
        elif execution_result is not None:
            reason = "Stopped by user" if self._stop_requested else "Max retries exceeded"
            self.state_manager.mark_failed(subtask, execution_result, verification_result,
                                           reason=reason)
            self.logger.log_subtask_failed(subtask, attempts, reason)
            view[index]["status"] = "failed"

        if execution_result is not None:
            self.state_manager.update_object_position(
                obj, (verification_result or {}).get("object_position") or [0, 0, 0])

        yield ProgressEvent(status=f"Subtask {subtask_id} "
                                   f"{'completed' if success else 'failed'}",
                            subtasks=_copy_view(view), log_text=self._log_text(),
                            video_path=video_path)

        yield {
            "subtask_id": subtask_id,
            "action": action,
            "object": obj,
            "target": target,
            "success": success,
            "attempts": attempts,
            "retries": max(0, attempts - 1),
            "duration": execution_result.get("duration", 0) if execution_result else 0,
            "video_path": video_path,
            "believed_success": bool(success),
            "oracle_success": attempt_log[-1]["oracle_success"] if attempt_log else None,
            "attempt_log": attempt_log,
        }

    def _log_text(self) -> str:
        # No ANSI colour: this string goes into a Gradio textbox as often as a
        # terminal, and escape codes render as literal garbage there.
        return self.logger.get_readable_log(colour=False)

    def close(self):
        self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def _copy_view(view: List[Dict]) -> List[Dict]:
    return [dict(v) for v in view]


def create_agent(use_mock_policy: bool = False,
                 console_output: bool = True) -> RobotTaskAgent:
    return RobotTaskAgent(use_mock_policy=use_mock_policy,
                          console_output=console_output)
