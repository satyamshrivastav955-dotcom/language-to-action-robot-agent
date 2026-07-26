import boto3
import json
import os
import re
from typing import Dict, Optional, List
from botocore.config import Config
from botocore.exceptions import ClientError, BotoCoreError

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.settings import (
    PLANNER_SYSTEM_PROMPT,
    PLANNER_EXAMPLES,
    OBJECT_GROUNDING,
)

BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "zai.glm-5")
BEDROCK_REGION = os.environ.get("AWS_DEFAULT_REGION", "eu-north-1")


class Planner:
    
    def __init__(self, model_id: Optional[str] = None, region: Optional[str] = None):
        self.model_id = model_id or BEDROCK_MODEL_ID
        self.region = region or BEDROCK_REGION
        
        config = Config(
            region_name=self.region,
            retries={'max_attempts': 3, 'mode': 'standard'}
        )
        
        self.client = boto3.client('bedrock-runtime', config=config)
        self.last_plan: Optional[Dict] = None
        self.used_fallback: bool = False
        
        print(f"[Planner] Initialized Bedrock client for model: {self.model_id}")
    
    def decompose(self, instruction: str, max_retries: int = 2) -> Dict:
        self.used_fallback = False
        last_error = None

        for attempt in range(max_retries):
            try:
                plan = self._call_llm(instruction)
                validated = self._validate_and_ground(plan)
                self.last_plan = validated
                print("[Planner] SOURCE: REAL_LLM (Bedrock)")
                return validated
            except json.JSONDecodeError as e:
                last_error = f"model returned non-JSON: {e}"
            except (ClientError, BotoCoreError) as e:
                last_error = f"{type(e).__name__}: {e}"
            except (ValueError, KeyError, IndexError) as e:
                last_error = f"malformed plan ({type(e).__name__}): {e}"
            print(f"[Planner] Attempt {attempt + 1}/{max_retries} failed - {last_error}")

        print(f"[Planner] WARNING: Bedrock unavailable ({last_error}).")
        print("[Planner] WARNING: using heuristic fallback plan - LLM was NOT used.")
        print("[Planner] SOURCE: FALLBACK (heuristic parser)")
        self.used_fallback = True
        fallback = self._create_fallback_plan(instruction)
        self.last_plan = fallback
        return fallback
    
    def _call_llm(self, instruction: str) -> Dict:
        messages = self._build_messages(instruction)
        
        request_body = {
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.1,
        }
        
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json"
        )

        response_body = json.loads(response['body'].read())

        response_text = self._extract_response_text(response_body)
        
        return self._extract_json(response_text)
    
    def _build_messages(self, instruction: str) -> List[Dict]:
        messages = []
        
        system_prompt = PLANNER_SYSTEM_PROMPT
        messages.append({"role": "system", "content": system_prompt})
        
        for example in PLANNER_EXAMPLES:
            messages.append({"role": "user", "content": example["instruction"]})
            messages.append({"role": "assistant", "content": json.dumps(example["plan"])})
        
        messages.append({"role": "user", "content": instruction})
        
        return messages
    
    def _extract_response_text(self, response_body: Dict) -> str:
        if "choices" in response_body:
            return response_body["choices"][0]["message"]["content"]
        
        if "output" in response_body:
            if isinstance(response_body["output"], dict):
                return response_body["output"].get("content", "")
            return response_body["output"]
        
        if "generation" in response_body:
            return response_body["generation"]
        
        if "results" in response_body:
            return response_body["results"][0].get("outputText", "")
        
        if "content" in response_body:
            if isinstance(response_body["content"], list):
                return response_body["content"][0].get("text", "")
            return response_body["content"]
        
        print(f"[Planner] Unknown response format: {list(response_body.keys())}")
        return str(response_body)
    
    def _extract_json(self, text: str) -> Dict:
        text = text.strip()
        
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = text[json_start:json_end]
            return json.loads(json_str)
        
        return json.loads(text)
    
    def _validate_and_ground(self, plan: Dict) -> Dict:
        # A clarification request is a valid plan shape with no subtasks.
        if plan.get("clarification_needed"):
            return {
                "clarification_needed": True,
                "question": str(plan.get("question")
                                or "Which object did you mean?"),
                "subtasks": [],
            }

        if "subtasks" not in plan:
            raise ValueError("Plan must have 'subtasks' key")
        
        valid_subtasks = []
        for i, subtask in enumerate(plan["subtasks"]):
            validated = {
                "id": subtask.get("id", i + 1),
                "action": self._validate_action(subtask.get("action", "pick_and_place")),
                "object": self._ground_object(subtask.get("object", "red block")),
                "target": self._validate_target(subtask.get("target", "bin")),
                "status": "pending",
                "attempts": 0,
                "result": None,
            }
            valid_subtasks.append(validated)
        
        return {"subtasks": valid_subtasks}
    
    def _validate_action(self, action: str) -> str:
        valid_actions = ["pick_and_place", "stack", "push"]
        if action in valid_actions:
            return action
        return "pick_and_place"
    
    def _ground_object(self, obj_name: str) -> str:
        obj_lower = str(obj_name).lower().strip()
        if obj_lower in OBJECT_GROUNDING:
            return OBJECT_GROUNDING[obj_lower]
        return obj_lower.replace(" ", "_")

    def _validate_target(self, target: str) -> str:
        target_lower = str(target).lower().strip()
        if target_lower in OBJECT_GROUNDING:
            return OBJECT_GROUNDING[target_lower]
        return target_lower.replace(" ", "_")
    
    def _find_objects_in_order(self, text: str) -> List[tuple]:
        """Return [(char_index, grounded_name), ...] in order of appearance.

        Longest phrases are matched first so "red block" wins over "red", and
        each character span is consumed once so a single mention cannot produce
        two hits.
        """
        matches = []
        consumed = [False] * len(text)

        for phrase in sorted(OBJECT_GROUNDING, key=len, reverse=True):
            start = 0
            while True:
                idx = text.find(phrase, start)
                if idx < 0:
                    break
                end = idx + len(phrase)
                # Require whole-word match and an unconsumed span.
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
        """Heuristic counterpart to the LLM's clarification check.

        Fires when the instruction refers to a block without saying which one,
        and no colour anywhere disambiguates it. Plural ("the blocks") means
        all of them, which is not ambiguous.
        """
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
            return {"clarification_needed": True, "question": question,
                    "subtasks": []}

        """Heuristic plan used only when the LLM is unreachable.

        Splits the instruction into clauses and reads each clause's own objects
        in the order they appear. The previous version indexed a de-duplicated
        list (objects[0]/objects[1]) built by scanning OBJECT_GROUNDING's dict
        order, so "stack the blue block on the green block" produced
        "stack red_block -> blue_block". Clause-local ordering fixes that.
        """
        text = instruction.lower()

        # Split on sequencing words so each clause is planned independently.
        clauses = re.split(r"\bthen\b|\bafter that\b|\bnext\b|[;.]|\band then\b", text)
        clauses = [c.strip() for c in clauses if c.strip()]
        if not clauses:
            clauses = [text]

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
                # "stack X on Y" -> X moves onto Y, in stated order.
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
                "action": action,
                "object": obj,
                "target": target,
                "status": "pending",
                "attempts": 0,
                "result": None,
            })

        if not subtasks:
            subtasks.append({
                "id": 1,
                "action": "pick_and_place",
                "object": "red_block",
                "target": "bin",
                "status": "pending",
                "attempts": 0,
                "result": None,
            })

        return {"subtasks": subtasks}


def create_planner(model_id: Optional[str] = None) -> Planner:
    return Planner(model_id=model_id)


def plan_instruction(instruction: str, model_id: Optional[str] = None) -> Dict:
    planner = Planner(model_id=model_id)
    return planner.decompose(instruction)


if __name__ == "__main__":
    print("Testing Planner with AWS Bedrock...")
    
    test_instructions = [
        "put the red block in the bin",
        "stack the blue block on the green block",
        "put red in bin then stack blue on green",
    ]
    
    planner = Planner()
    
    for instruction in test_instructions:
        print(f"\nInstruction: '{instruction}'")
        try:
            plan = planner.decompose(instruction)
            print("Plan:")
            for subtask in plan["subtasks"]:
                print(f"  - {subtask}")
        except Exception as e:
            print(f"Error: {e}")
            print("Testing fallback plan:")
            plan = planner._create_fallback_plan(instruction)
            for subtask in plan["subtasks"]:
                print(f"  - {subtask}")
    
    print("\nPlanner test complete!")
