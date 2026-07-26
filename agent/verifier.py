import numpy as np
import os
from typing import Dict, Optional, Tuple
import base64
import cv2

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.settings import (
    VERIFICATION_TOLERANCE,
    OBJECT_GROUNDING,
    VLM_VERIFIER_PROMPT,
    TARGET_REGIONS,
    BIN_MIN_HEIGHT,
    BLOCK_SIZE,
    BEDROCK_REGION,
    VLM_MODEL_ID,
)


class Verifier:
    """Geometric success check, with an optional VLM second opinion.

    Tier A (always): compare the object's pose against the target region.
    Tier B (--vlm-verify): ask a vision model to judge the final frame.

    Tier B previously could not run at all - `enable_vlm` was plumbed through
    but `anthropic_client` was only settable via the constructor and no caller
    passed one, so every call short-circuited on "VLM client not available".
    It now builds a Bedrock client itself, matching the planner's backend.
    """

    def __init__(self, env, enable_vlm: bool = False, vlm_client=None,
                 vlm_model_id: Optional[str] = None, anthropic_client=None):
        self.env = env
        self.enable_vlm = enable_vlm
        self.tolerance = VERIFICATION_TOLERANCE
        self.vlm_model_id = vlm_model_id or VLM_MODEL_ID

        # `anthropic_client` is the old keyword; accept it so existing callers
        # keep working, but treat it as a generic VLM client.
        self.vlm_client = vlm_client or anthropic_client
        self.vlm_unavailable_reason: Optional[str] = None

        if self.enable_vlm and self.vlm_client is None:
            self.vlm_client = self._build_bedrock_client()

    def _build_bedrock_client(self):
        try:
            import boto3
            from botocore.config import Config

            client = boto3.client(
                "bedrock-runtime",
                config=Config(region_name=BEDROCK_REGION,
                              retries={"max_attempts": 2, "mode": "standard"}),
            )
            print(f"[Verifier] VLM verification enabled via Bedrock ({self.vlm_model_id})")
            return client
        except Exception as e:
            self.vlm_unavailable_reason = f"could not create Bedrock client: {e}"
            print(f"[Verifier] VLM verification requested but unavailable - "
                  f"{self.vlm_unavailable_reason}")
            return None

    
    def verify(self, subtask: Dict, execution_result: Dict) -> Dict:
        action = subtask.get("action", "pick_and_place")
        obj = subtask.get("object", "red_block")
        target = subtask.get("target", "bin")
        
        verification_methods = {
            "pick_and_place": self._verify_pick_and_place,
            "stack": self._verify_stack,
            "push": self._verify_push,
        }
        
        verify_fn = verification_methods.get(action, self._verify_pick_and_place)
        result = verify_fn(obj, target, execution_result)
        
        if self.enable_vlm and execution_result.get("frames"):
            vlm_result = self._vlm_verify(subtask, execution_result)
            result["vlm_verification"] = vlm_result
        
        return result
    
    def _verify_pick_and_place(self, obj: str, target: str, execution_result: Dict) -> Dict:
        try:
            object_pos = self.env.get_object_pose(obj)
        except Exception as e:
            return {
                "success": False,
                "reason": f"Object '{obj}' not found in scene",
                "object_position": None,
                "target_position": None,
                "distance": float('inf'),
            }
        
        target_pos = self._get_target_position(target)
        
        distance_2d = np.linalg.norm(object_pos[:2] - target_pos[:2])
        distance_3d = np.linalg.norm(object_pos - target_pos)
        
        height_ok = True
        if target == "bin":
            height_ok = object_pos[2] > BIN_MIN_HEIGHT
        
        success = distance_2d < self.tolerance and height_ok
        
        reason = self._generate_failure_reason(
            distance_2d, self.tolerance, execution_result, "pick_and_place"
        ) if not success else "Object successfully placed at target"
        
        return {
            "success": bool(success),
            "reason": reason,
            "object_position": object_pos.tolist(),
            "target_position": target_pos.tolist(),
            "distance": float(distance_2d),
            "height_ok": bool(height_ok),
        }
    
    def _verify_stack(self, obj: str, target: str, execution_result: Dict) -> Dict:
        try:
            object_pos = self.env.get_object_pose(obj)
        except Exception:
            return {
                "success": False,
                "reason": f"Object '{obj}' not found",
            }
        
        try:
            target_pos = self.env.get_object_pose(target)
        except Exception:
            return {
                "success": False,
                "reason": f"Target '{target}' not found",
            }
        
        distance_horizontal = np.linalg.norm(object_pos[:2] - target_pos[:2])
        
        expected_top = target_pos[2] + BLOCK_SIZE
        height_distance = abs(object_pos[2] - expected_top)
        
        horizontal_ok = distance_horizontal < self.tolerance * 1.5
        # Must be within half a block of the expected top. The old bound
        # (tolerance*2 = 0.1m) was wider than a whole block, so a block still
        # sitting on the table passed as "stacked".
        vertical_ok = height_distance < BLOCK_SIZE * 0.5
        
        success = horizontal_ok and vertical_ok
        
        reason = self._generate_failure_reason(
            distance_horizontal, self.tolerance, execution_result, "stack"
        ) if not success else "Object successfully stacked"
        
        return {
            "success": bool(success),
            "reason": reason,
            "object_position": object_pos.tolist(),
            "target_position": target_pos.tolist(),
            "horizontal_distance": float(distance_horizontal),
            "height_distance": float(height_distance),
        }
    
    def _verify_push(self, obj: str, target: str, execution_result: Dict) -> Dict:
        return self._verify_pick_and_place(obj, target, execution_result)
    
    def _get_target_position(self, target: str) -> np.ndarray:
        # Prefer a live object pose (e.g. stacking onto another block), then
        # fall back to the shared TARGET_REGIONS table. Never hardcode here.
        if target in OBJECT_GROUNDING.values() and target != "bin":
            try:
                return np.asarray(self.env.get_object_pose(target), dtype=float)
            except Exception:
                pass

        if target in TARGET_REGIONS:
            return np.array(TARGET_REGIONS[target], dtype=float)

        return np.array(TARGET_REGIONS["table_center"], dtype=float)
    
    def _generate_failure_reason(self, distance: float, tolerance: float,
                                   execution_result: Dict, action: str) -> str:
        object_moved = execution_result.get("object_moved", False)
        
        if not object_moved:
            return "Grasp failed - object was not picked up (contact/friction issue)"
        
        distance_ratio = distance / tolerance
        if distance_ratio > 5:
            return f"Target missed by large margin ({distance:.2f}m) - placement accuracy issue"
        elif distance_ratio > 2:
            return f"Target missed by moderate margin ({distance:.2f}m) - possible timing issue"
        else:
            return f"Close to target but outside tolerance ({distance:.2f}m > {tolerance:.2f}m)"
    
    def _vlm_verify(self, subtask: Dict, execution_result: Dict) -> Dict:
        if not self.vlm_client:
            reason = self.vlm_unavailable_reason or "VLM client not available"
            return {"success": None, "reason": reason}

        frames = execution_result.get("frames", [])
        if not frames:
            return {"success": None, "reason": "No frames available"}

        final_frame = frames[-1]

        task_description = (f"{subtask.get('action', 'move')} "
                            f"{subtask.get('object', 'object')} to "
                            f"{subtask.get('target', 'target')}")
        prompt = VLM_VERIFIER_PROMPT.format(task_description=task_description)

        try:
            # Frames are RGB in-process; cv2 encodes BGR, so convert or the
            # model is shown colour-swapped blocks - fatal when the task is
            # literally "is the red block in the bin".
            bgr = cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)
            ok, buffer = cv2.imencode('.jpg', bgr)
            if not ok:
                return {"success": None, "reason": "Failed to encode frame as JPEG"}
            img_bytes = buffer.tobytes()

            vlm_response = self._invoke_vlm(img_bytes, prompt)

            upper = vlm_response.upper()
            if "SUCCESS" in upper:
                success = True
            elif "FAIL" in upper:
                success = False
            else:
                # An unparseable answer is not a failure verdict - saying so
                # keeps it from silently counting as one.
                return {"success": None,
                        "reason": f"Unparseable VLM response: {vlm_response[:200]}",
                        "raw_response": vlm_response}

            reason = vlm_response.split(":", 1)[-1].strip() if ":" in vlm_response else vlm_response

            return {"success": success, "reason": reason, "raw_response": vlm_response}
        except Exception as e:
            return {"success": None, "reason": f"VLM error: {type(e).__name__}: {e}"}

    def _invoke_vlm(self, img_bytes: bytes, prompt: str) -> str:
        """Call the VLM and return its text.

        Supports both a Bedrock runtime client (boto3) and an Anthropic-style
        client, since either may be injected.
        """
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        # Anthropic-style SDK client.
        if hasattr(self.vlm_client, "messages"):
            response = self.vlm_client.messages.create(
                model=self.vlm_model_id,
                max_tokens=256,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image",
                         "source": {"type": "base64", "media_type": "image/jpeg",
                                    "data": img_base64}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            return response.content[0].text

        # Bedrock runtime client. Converse() gives one payload shape across
        # model families, so we do not have to hand-roll per-vendor JSON.
        response = self.vlm_client.converse(
            modelId=self.vlm_model_id,
            messages=[{
                "role": "user",
                "content": [
                    {"image": {"format": "jpeg", "source": {"bytes": img_bytes}}},
                    {"text": prompt},
                ],
            }],
            inferenceConfig={"maxTokens": 256, "temperature": 0.0},
        )

        parts = response["output"]["message"]["content"]
        return "".join(p.get("text", "") for p in parts)


def create_verifier(env, enable_vlm: bool = False) -> Verifier:
    return Verifier(env, enable_vlm=enable_vlm)


if __name__ == "__main__":
    print("Testing Verifier...")
    
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from env.so101_env import SO101Environment
    
    env = SO101Environment()
    env.reset()
    print("Environment reset")
    
    print("\nObject positions after reset:")
    for obj_name in ["red_block", "blue_block", "green_block", "yellow_block"]:
        pos = env.get_object_pose(obj_name)
        print(f"  {obj_name}: {pos}")
    
    verifier = create_verifier(env)
    print("\nVerifier created")
    
    print("\nTest 1: Verify red block is at starting position")
    test_subtask = {"action": "pick_and_place", "object": "red_block", "target": "bin"}
    test_result = {"object_moved": False, "frames": []}
    verification = verifier.verify(test_subtask, test_result)
    print(f"Result: {verification}")
    
    print("\nTest 2: Verify pick_and_place with mock movement")
    test_result2 = {"object_moved": True, "frames": []}
    verification2 = verifier.verify(test_subtask, test_result2)
    print(f"Result: {verification2}")
    
    print("\nTest 3: Verify stack operation")
    stack_subtask = {"action": "stack", "object": "blue_block", "target": "green_block"}
    verification3 = verifier.verify(stack_subtask, test_result)
    print(f"Result: {verification3}")
    
    env.close()
    print("\nVerifier test complete!")
