import os
from typing import Dict, Optional, Tuple
import time

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.settings import MAX_RETRIES_PER_SUBTASK


class RetryController:
    
    def __init__(self, max_retries: int = MAX_RETRIES_PER_SUBTASK):
        self.max_retries = max_retries
        self.retry_history: Dict[int, list] = {}
    
    def should_retry(self, subtask: Dict, verification: Dict,
                     max_retries: Optional[int] = None) -> bool:
        """Decide whether another attempt is warranted.

        `max_retries` overrides the instance default so a caller that was given
        a per-run budget (CLI flag, UI slider) stays authoritative.
        """
        limit = self.max_retries if max_retries is None else max_retries
        retry_count = subtask.get("attempts", 0)

        if retry_count >= limit:
            return False

        if verification.get("success", False):
            return False

        return True
    
    def diagnose(self, subtask: Dict, execution_result: Dict, verification: Dict) -> str:
        failure_type = self._classify_failure(execution_result, verification)
        diagnosis = self._generate_diagnosis(failure_type, execution_result, verification)
        
        subtask_id = subtask.get("id", 0)
        if subtask_id not in self.retry_history:
            self.retry_history[subtask_id] = []
        self.retry_history[subtask_id].append({
            "failure_type": failure_type,
            "diagnosis": diagnosis,
            "timestamp": time.time(),
        })
        
        return diagnosis
    
    def _classify_failure(self, execution_result: Dict, verification: Dict) -> str:
        object_moved = execution_result.get("object_moved", False)
        distance = verification.get("distance", float('inf'))
        height_ok = verification.get("height_ok", True)
        reason = verification.get("reason", "").lower()
        
        if not object_moved:
            return "grasp_failure"
        
        if "grasp" in reason or "contact" in reason or "friction" in reason:
            return "grasp_slip"
        
        if not height_ok:
            return "placement_height_issue"
        
        if distance > 0.2:
            return "large_placement_error"
        elif distance > 0.1:
            return "moderate_placement_error"
        else:
            return "minor_placement_error"
    
    def _generate_diagnosis(self, failure_type: str, execution_result: Dict, verification: Dict) -> str:
        diagnoses = {
            "grasp_failure": "Pre-grasp approach failed - gripper did not make contact with object. Consider: repositioning approach vector, checking gripper aperture.",
            "grasp_slip": "Object was grasped but slipped during transport. Consider: closing gripper earlier, slower transport motion.",
            "placement_height_issue": "Object reached target area but at wrong height. Consider: adjusting release height.",
            "large_placement_error": "Placement missed target significantly. Consider: recalibrating motion trajectory.",
            "moderate_placement_error": "Placement was close but outside tolerance. Consider: adjusting final approach motion.",
            "minor_placement_error": "Placement nearly succeeded - minor refinement may help.",
        }
        return diagnoses.get(failure_type, "Unknown failure type - will retry with default approach")
    
    def get_adjusted_parameters(self, subtask: Dict, attempt_number: int) -> Dict:
        """Parameters for the upcoming attempt, derived from the last failure.

        On attempt 1 there is no history, so this returns the baseline. Callers
        can pass the result straight to Executor.run_subtask.
        """
        base_params = {
            "retry_count": attempt_number,
            "speed_factor": 1.0,
        }

        subtask_id = subtask.get("id", 0)
        history = self.retry_history.get(subtask_id, [])

        if len(history) > 0:
            last_failure = history[-1].get("failure_type", "")
            base_params["last_failure"] = last_failure

            if "grasp" in last_failure:
                base_params["approach_adjustment"] = "earlier_gripper_close"
                base_params["speed_factor"] = 0.8
            elif "placement" in last_failure:
                base_params["speed_factor"] = 0.9
                base_params["position_offset"] = [0.01, 0.01, 0]

        return base_params
    
    def get_retry_count(self, subtask_id: int) -> int:
        return len(self.retry_history.get(subtask_id, []))
    
    def reset(self, subtask_id: Optional[int] = None):
        if subtask_id is not None:
            self.retry_history[subtask_id] = []
        else:
            self.retry_history = {}


class RetryDiagnostic:
    
    FAILURE_TYPES = {
        "grasp_failure": {
            "severity": "high",
            "recovery_likelihood": 0.7,
            "recommended_actions": ["adjust_approach", "slow_down"],
        },
        "grasp_slip": {
            "severity": "medium",
            "recovery_likelihood": 0.6,
            "recommended_actions": ["tighten_grip", "slow_transport"],
        },
        "large_placement_error": {
            "severity": "high",
            "recovery_likelihood": 0.5,
            "recommended_actions": ["recalibrate", "check_sensors"],
        },
        "moderate_placement_error": {
            "severity": "medium",
            "recovery_likelihood": 0.8,
            "recommended_actions": ["refine_motion"],
        },
        "minor_placement_error": {
            "severity": "low",
            "recovery_likelihood": 0.9,
            "recommended_actions": ["slight_adjustment"],
        },
        "timing_issue": {
            "severity": "low",
            "recovery_likelihood": 0.85,
            "recommended_actions": ["adjust_timing"],
        },
    }
    
    @classmethod
    def classify(cls, verification: Dict) -> str:
        reason = verification.get("reason", "").lower()
        
        if "grasp" in reason and "failed" in reason:
            return "grasp_failure"
        if "slip" in reason:
            return "grasp_slip"
        if "large margin" in reason:
            return "large_placement_error"
        if "moderate margin" in reason:
            return "moderate_placement_error"
        if "close" in reason:
            return "minor_placement_error"
        
        return "unknown"
    
    @classmethod
    def get_recommended_action(cls, failure_type: str) -> list:
        return cls.FAILURE_TYPES.get(failure_type, {}).get("recommended_actions", [])


def create_retry_controller(max_retries: int = MAX_RETRIES_PER_SUBTASK) -> RetryController:
    return RetryController(max_retries=max_retries)


if __name__ == "__main__":
    print("Testing Retry Controller...")
    
    controller = create_retry_controller(max_retries=3)
    
    test_subtask = {"id": 1, "action": "pick_and_place", "object": "red block", "target": "bin", "attempts": 0}
    
    print("\nTest 1: First failure - should retry")
    test_verification = {"success": False, "reason": "Grasp failed - object was not picked up", "distance": 0.5}
    test_execution = {"object_moved": False}
    
    should_retry = controller.should_retry(test_subtask, test_verification)
    print(f"Should retry: {should_retry}")
    
    if should_retry:
        diagnosis = controller.diagnose(test_subtask, test_execution, test_verification)
        print(f"Diagnosis: {diagnosis}")
        
        params = controller.get_adjusted_parameters(test_subtask, 1)
        print(f"Adjusted params: {params}")
    
    print("\nTest 2: Success - should not retry")
    test_verification_success = {"success": True, "reason": "Object successfully placed"}
    should_retry2 = controller.should_retry(test_subtask, test_verification_success)
    print(f"Should retry: {should_retry2}")
    
    print("\nTest 3: Max retries exceeded")
    test_subtask_max = {"id": 2, "action": "pick_and_place", "object": "red block", "target": "bin", "attempts": 3}
    should_retry3 = controller.should_retry(test_subtask_max, test_verification)
    print(f"Should retry: {should_retry3}")
    
    print("\nRetry history:")
    print(controller.retry_history)
    
    print("\nRetry Controller test complete!")
