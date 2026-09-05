"""LLM task planner with pluggable backends.

Default backend: Ollama (local, offline, zero API cost).
Optional backend: AWS Bedrock (set LLM_BACKEND=bedrock).

Both backends share all plan validation, grounding and heuristic fallback
logic from BasePlanner. Only _call_llm differs between them.
"""
import json
import os
import re
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.settings import (
    PLANNER_SYSTEM_PROMPT,
    PLANNER_EXAMPLES,
    OBJECT_GROUNDING,
    LLM_BACKEND,
    OLLAMA_MODEL,
    OLLAMA_HOST,
    BEDROCK_MODEL_ID,
    BEDROCK_REGION,
)


class BasePlanner:
    """Shared validation, grounding and heuristic fallback. Backend-agnostic."""

    def __init__(self):
        self.last_plan: Optional[Dict] = None
        self.used_fallback: bool = False
        self.backend_name: str = "base"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(self, instruction: str, max_retries: int = 2) -> Dict:
        self.used_fallback = False
        last_error = None

        for attempt in range(max_retries):
            try:
                plan = self._call_llm(instruction)
                validated = self._validate_and_ground(plan)
                self.last_plan = validated
                print(f"[Planner] SOURCE: REAL_LLM ({self.backend_name})")
                return validated
            except json.JSONDecodeError as e:
                last_error = f"model returned non-JSON: {e}"
            except ConnectionRefusedError as e:
                last_error = f"connection refused - is Ollama running? (ollama serve): {e}"
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
            print(f"[Planner] Attempt {attempt + 1}/{max_retries} failed - {last_error}")

        print(f"[Planner] WARNING: LLM unavailable ({last_error}).")
        print("[Planner] WARNING: using heuristic fallback - LLM was NOT used.")
        print("[Planner] SOURCE: FALLBACK (heuristic parser)")
        self.used_fallback = True
        fallback = self._create_fallback_plan(instruction)
        self.last_plan = fallback
        return fallback

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    def _call_llm(self, instruction: str) -> Dict:
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _build_messages(self, instruction: str) -> List[Dict]:
        messages = [{"role": "system", "content": PLANNER_SYSTEM_PROMPT}]
        for example in PLANNER_EXAMPLES:
            messages.append({"role": "user", "content": example["instruction"]})
            messages.append({"role": "assistant", "content": json.dumps(example["plan"])})
        messages.append({"role": "user", "content": instruction})
        return messages

    def _extract_json(self, text: str) -> Dict:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            return json.loads(text[json_start:json_end])
        return json.loads(text)

    def _validate_and_ground(self, plan: Dict) -> Dict:
        if plan.get("clarification_needed"):
            return {
                "clarification_needed": True,
                "question": str(plan.get("question") or "Which object did you mean?"),
                "subtasks": [],
            }
        if "subtasks" not in plan:
            raise ValueError("Plan must have 'subtasks' key")
        valid_subtasks = []
        for i, subtask in enumerate(plan["subtasks"]):
            valid_subtasks.append({
                "id": subtask.get("id", i + 1),
                "action": self._validate_action(subtask.get("action", "pick_and_place")),
                "object": self._ground_object(subtask.get("object", "red block")),
                "target": self._validate_target(subtask.get("target", "bin")),
                "status": "pending",
                "attempts": 0,
                "result": None,
            })
        return {"subtasks": valid_subtasks}

    def _validate_action(self, action: str) -> str:
        return action if action in ("pick_and_place", "stack", "push") else "pick_and_place"

    def _ground_object(self, obj_name: str) -> str:
        obj_lower = str(obj_name).lower().strip()
        return OBJECT_GROUNDING.get(obj_lower, obj_lower.replace(" ", "_"))

    def _validate_target(self, target: str) -> str:
        target_lower = str(target).lower().strip()
        return OBJECT_GROUNDING.get(target_lower, target_lower.replace(" ", "_"))

    def _find_objects_in_order(self, text: str) -> List[tuple]:
        matches = []
        consumed = [False] * len(text)
        for phrase in sorted(OBJECT_GROUNDING, key=len, reverse=True):
            start = 0
            while True:
                idx = text.find(phrase, start)
                if idx < 0:
                    break
                end = idx + len(phrase)
                before_ok = idx == 0 or not text[idx - 1].isalnum()
                after_ok = end >= len(text) or not text[end].isalnum()
                if before_ok and after_ok and not any(consumed[idx:end]):
                    for i in range(idx, end):
                        consumed[i] = True
                    matches.append((idx, OBJECT_GROUNDING[phrase]))
                start = end
        matches.sort(key=lambda m: m[0])
        return matches

    AMBIGUITY_COLORS = ("red", "blue", "green", "yellow")

    def _detect_ambiguity(self, instruction: str) -> Optional[str]:
        text = instruction.lower()
        if any(c in text for c in self.AMBIGUITY_COLORS):
            return None
        if "blocks" in text:
            return None
        if "block" not in text:
            return None
        return ("There are four blocks on the table (red, blue, green, "
                "yellow). Which one did you mean?")

    def _create_fallback_plan(self, instruction: str) -> Dict:
        question = self._detect_ambiguity(instruction)
        if question:
            return {"clarification_needed": True, "question": question, "subtasks": []}
        text = instruction.lower()
        clauses = re.split(r"\bthen\b|\bafter that\b|\bnext\b|[;.]|\band then\b", text)
        clauses = [c.strip() for c in clauses if c.strip()] or [text]
        subtasks = []
        for clause in clauses:
            found = self._find_objects_in_order(clause)
            if not found:
                continue
            blocks = [name for _, name in found if name != "bin"]
            has_bin = any(name == "bin" for _, name in found)
            is_stack = "stack" in clause
            is_push = "push" in clause
            if is_stack and len(blocks) >= 2:
                action, obj, target = "stack", blocks[0], blocks[1]
            elif has_bin and blocks:
                action, obj, target = "pick_and_place", blocks[0], "bin"
            elif is_push and blocks:
                action = "push"
                obj = blocks[0]
                target = blocks[1] if len(blocks) >= 2 else "table_center"
            elif len(blocks) >= 2:
                action, obj, target = "pick_and_place", blocks[0], blocks[1]
            elif blocks:
                action, obj, target = "pick_and_place", blocks[0], "bin"
            else:
                continue
            subtasks.append({
                "id": len(subtasks) + 1,
                "action": action, "object": obj, "target": target,
                "status": "pending", "attempts": 0, "result": None,
            })
        if not subtasks:
            subtasks.append({"id": 1, "action": "pick_and_place",
                             "object": "red_block", "target": "bin",
                             "status": "pending", "attempts": 0, "result": None})
        return {"subtasks": subtasks}


class OllamaPlanner(BasePlanner):
    """Offline LLM planner via Ollama (http://localhost:11434).

    Requirements: Ollama installed and running (`ollama serve`).
    Model pulled: e.g. `ollama pull qwen2.5:7b`

    Uses Ollama's native `format: "json"` to guarantee structured output
    without needing to prompt-engineer JSON fences.
    """

    def __init__(self, model: str = OLLAMA_MODEL, host: str = OLLAMA_HOST):
        super().__init__()
        self.model = model
        self.host = host.rstrip("/")
        self.backend_name = f"ollama/{model}"
        print(f"[Planner] Backend: Ollama  model={model}  host={host}")

    def _call_llm(self, instruction: str) -> Dict:
        import requests  # stdlib-adjacent; always present via gradio transitive dep
        messages = self._build_messages(instruction)
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.1, "num_predict": 512},
        }
        try:
            resp = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionRefusedError(
                f"Cannot reach Ollama at {self.host}. "
                "Start it with: ollama serve"
            ) from e

        data = resp.json()
        content = data["message"]["content"]
        return self._extract_json(content)


class BedrockPlanner(BasePlanner):
    """AWS Bedrock planner (cloud, requires boto3 + AWS credentials).

    Activate with: LLM_BACKEND=bedrock
    """

    def __init__(self, model_id: str = BEDROCK_MODEL_ID, region: str = BEDROCK_REGION):
        super().__init__()
        self.model_id = model_id
        self.region = region
        self.backend_name = f"bedrock/{model_id}"

        try:
            import boto3
            from botocore.config import Config
            config = Config(
                region_name=self.region,
                retries={"max_attempts": 3, "mode": "standard"}
            )
            self.client = boto3.client("bedrock-runtime", config=config)
            print(f"[Planner] Backend: AWS Bedrock  model={model_id}  region={region}")
        except ImportError:
            raise ImportError(
                "boto3 is required for the Bedrock backend. "
                "Install with: pip install boto3"
            )

    def _call_llm(self, instruction: str) -> Dict:
        from botocore.exceptions import ClientError, BotoCoreError
        messages = self._build_messages(instruction)
        request_body = {
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.1,
        }
        try:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(request_body),
                contentType="application/json",
                accept="application/json",
            )
        except (ClientError, BotoCoreError) as e:
            raise RuntimeError(f"Bedrock call failed: {e}") from e

        body = json.loads(response["body"].read())
        text = self._extract_response_text(body)
        return self._extract_json(text)

    def _extract_response_text(self, body: Dict) -> str:
        if "choices" in body:
            return body["choices"][0]["message"]["content"]
        if "output" in body:
            out = body["output"]
            return out.get("content", "") if isinstance(out, dict) else out
        if "generation" in body:
            return body["generation"]
        if "results" in body:
            return body["results"][0].get("outputText", "")
        if "content" in body:
            c = body["content"]
            return c[0].get("text", "") if isinstance(c, list) else c
        return str(body)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_planner(backend: Optional[str] = None) -> BasePlanner:
    """Return the appropriate planner for the configured or requested backend.

    Args:
        backend: "ollama" | "bedrock" | None (reads LLM_BACKEND env/setting).
    """
    backend = (backend or LLM_BACKEND).lower()
    if backend == "bedrock":
        return BedrockPlanner()
    return OllamaPlanner()


# Alias so old code doing `from agent.planner import Planner` still works.
Planner = OllamaPlanner


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--backend", default=None)
    args = p.parse_args()

    planner = create_planner(args.backend)
    for instr in [
        "put the red block in the bin",
        "stack the blue block on the green block",
        "put red in bin then stack blue on green",
    ]:
        print(f"\nInstruction: '{instr}'")
        plan = planner.decompose(instr)
        for st in plan.get("subtasks", []):
            print(f"  {st}")
    print("\nPlanner self-test complete.")
