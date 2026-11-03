import numpy as np
from typing import Dict

from configs.settings import (
    VERIFICATION_TOLERANCE,
    OBJECT_GROUNDING,
    TARGET_REGIONS,
    BIN_MIN_HEIGHT,
    BLOCK_SIZE,
)


class Verifier:
    """Geometric success check on an object's pose against its target.

    Where the pose comes from decides who the verifier is: simulator truth by
    default (the oracle), camera estimates in PerceptionVerifier below.
    """

    # Where object poses come from. Default = simulator ground truth (the
    # ORACLE verifier). PerceptionVerifier below swaps in camera-based estimates.
    observed_by = "oracle"
    use_target_estimate = False

    def __init__(self, env, pose_source=None):
        self.env = env
        self._pose = pose_source or env.get_object_pose
        self.tolerance = VERIFICATION_TOLERANCE

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
        result["observed_by"] = self.observed_by

        # Measured error: where the object is minus where it should be (xy).
        # This vector is what a corrected retry feeds on.
        if result.get("object_position") is not None and result.get("target_position") is not None:
            op, tp = np.asarray(result["object_position"]), np.asarray(result["target_position"])
            result["error_vector"] = [float(op[0] - tp[0]), float(op[1] - tp[1])]
            initial = execution_result.get("initial_pos")
            if initial is not None:
                result["object_moved"] = bool(np.linalg.norm(op - np.asarray(initial)) > 0.01)
        
        
        return result
    
    def _verify_pick_and_place(self, obj: str, target: str, execution_result: Dict) -> Dict:
        try:
            object_pos = self._pose(obj)
        except Exception:
            return {
                "success": False,
                "reason": f"Object '{obj}' not found in scene",
                "object_position": None,
                "target_position": None,
                "distance": float('inf'),
            }
        
        target_pos = self._get_target_position(target)
        
        distance_2d = np.linalg.norm(object_pos[:2] - target_pos[:2])
        
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
            object_pos = self._pose(obj)
        except Exception:
            return {
                "success": False,
                "reason": f"Object '{obj}' not found",
            }
        
        # A stack target is hidden under the stacked block, so a camera cannot
        # re-measure it afterwards; use the pose estimated before the attempt.
        target_pos = execution_result.get("target_pos_est") if self.use_target_estimate else None
        if target_pos is None:
            try:
                target_pos = self._pose(target)
            except Exception:
                return {
                    "success": False,
                    "reason": f"Target '{target}' not found",
                }
        target_pos = np.asarray(target_pos, dtype=float)
        
        distance_horizontal = np.linalg.norm(object_pos[:2] - target_pos[:2])
        
        expected_top = target_pos[2] + BLOCK_SIZE
        height_distance = abs(object_pos[2] - expected_top)
        
        # A cube overhanging its support by more than a quarter of its width tips.
        horizontal_ok = distance_horizontal < BLOCK_SIZE * 0.75
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
                return np.asarray(self._pose(target), dtype=float)
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


class PerceptionVerifier(Verifier):
    """Judges outcomes from a fresh RGB-D capture; never reads simulator state."""

    observed_by = "perception"
    use_target_estimate = True

    def __init__(self, env, perception, **kwargs):
        super().__init__(env, pose_source=self._observed_pose, **kwargs)
        self.perception = perception

    def _observed_pose(self, name: str) -> np.ndarray:
        pose = self.perception.pose(name)
        if pose is None:
            raise KeyError(f"{name} not visible")
        return pose

    def verify(self, subtask: Dict, execution_result: Dict) -> Dict:
        self.perception.perceive(park=True)
        return super().verify(subtask, execution_result)


class NullVerifier:
    """The no-verifier ablation arm: believes the rollout worked."""

    observed_by = "none"

    def verify(self, subtask: Dict, execution_result: Dict) -> Dict:
        return {"success": True, "reason": "not verified", "observed_by": "none",
                "unverified": True}


def create_verifier(env) -> Verifier:
    return Verifier(env)


if __name__ == "__main__":
    print("Testing Verifier...")

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
