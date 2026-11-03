"""Retry policy: decide whether to try again and what to change.

Two strategies, so the benefit of using the measured error can be ablated:

* ``blind``      - retry with the same plan. Objects are re-perceived by the
                   executor, so a dropped block is re-grasped where it now lies,
                   but no correction is derived from the failure.
* ``corrected``  - additionally feed the *measured* placement error back into
                   the next release position: ``c <- c - alpha * error``.

The correction is only applied when the last attempt actually carried the block
to the target area (gripper held it, and it landed within ``carried_radius`` of
the target). A drop mid-carry or a missed grasp says nothing about the release
calibration, so it is not "corrected".
"""

import time
from typing import Dict, List, Optional

import numpy as np

from configs.settings import MAX_RETRIES_PER_SUBTASK


class RetryController:

    def __init__(self, max_retries: int = MAX_RETRIES_PER_SUBTASK, strategy: str = "corrected",
                 alpha: float = 0.85, carried_radius: float = 0.12):
        if strategy not in ("blind", "corrected"):
            raise ValueError(f"unknown retry strategy: {strategy}")
        self.max_retries = max_retries
        self.strategy = strategy
        self.alpha = alpha
        self.carried_radius = carried_radius
        self.retry_history: Dict[int, List[Dict]] = {}
        self.correction: Dict[int, np.ndarray] = {}

    def should_retry(self, subtask: Dict, verification: Dict,
                     max_retries: Optional[int] = None) -> bool:
        """Another attempt is warranted unless the budget is spent or the last one passed.

        `max_retries` overrides the instance default so a per-run budget (CLI
        flag, UI slider) stays authoritative.
        """
        limit = self.max_retries if max_retries is None else max_retries
        if subtask.get("attempts", 0) >= limit:
            return False
        return not verification.get("success", False)

    def observe(self, subtask: Dict, execution_result: Dict, verification: Dict) -> Optional[List[float]]:
        """Update the release correction from a failed attempt. Returns it if changed."""
        if self.strategy != "corrected" or verification.get("success"):
            return None
        err = verification.get("error_vector")
        if err is None or not execution_result.get("grasp_detected"):
            return None
        err = np.asarray(err, dtype=float)
        if np.linalg.norm(err) > self.carried_radius:
            return None
        sid = subtask.get("id", 0)
        prev = self.correction.get(sid, np.zeros(2))
        self.correction[sid] = prev - self.alpha * err
        return self.correction[sid].tolist()

    def diagnose(self, subtask: Dict, execution_result: Dict, verification: Dict) -> str:
        failure_type = self._classify_failure(execution_result, verification)
        diagnosis = self._generate_diagnosis(failure_type, execution_result, verification)
        self.retry_history.setdefault(subtask.get("id", 0), []).append({
            "failure_type": failure_type, "diagnosis": diagnosis, "timestamp": time.time()})
        return diagnosis

    def _classify_failure(self, execution_result: Dict, verification: Dict) -> str:
        object_moved = execution_result.get("object_moved", False)
        distance = verification.get("distance", float("inf"))
        height_ok = verification.get("height_ok", True)

        if not object_moved:
            return "grasp_failure"
        if not height_ok:
            return "placement_height_issue"
        if distance > 0.2:
            return "large_placement_error"
        if distance > 0.1:
            return "moderate_placement_error"
        return "minor_placement_error"

    def _generate_diagnosis(self, failure_type: str, execution_result: Dict, verification: Dict) -> str:
        err = verification.get("error_vector")
        measured = f" Measured error (x,y): ({err[0] * 100:+.1f}, {err[1] * 100:+.1f}) cm." if err else ""
        diagnoses = {
            "grasp_failure": "Block was not lifted; will re-perceive it and re-grasp.",
            "placement_height_issue": "Block reached the target area at the wrong height.",
            "large_placement_error": "Placement missed the target by a large margin.",
            "moderate_placement_error": "Placement was close but outside tolerance.",
            "minor_placement_error": "Placement nearly succeeded.",
        }
        return diagnoses.get(failure_type, "Unknown failure type.") + measured

    def get_adjusted_parameters(self, subtask: Dict, attempt_number: int) -> Dict:
        """Parameters for the upcoming attempt. Attempt 1 gets the baseline."""
        params: Dict = {"retry_count": attempt_number}
        history = self.retry_history.get(subtask.get("id", 0), [])
        if history:
            params["last_failure"] = history[-1].get("failure_type", "")
        corr = self.correction.get(subtask.get("id", 0))
        if corr is not None:
            params["place_correction"] = [float(corr[0]), float(corr[1]), 0.0]
        return params

    def get_retry_count(self, subtask_id: int) -> int:
        return len(self.retry_history.get(subtask_id, []))

    def reset(self, subtask_id: Optional[int] = None):
        if subtask_id is not None:
            self.retry_history[subtask_id] = []
            self.correction.pop(subtask_id, None)
        else:
            self.retry_history = {}
            self.correction = {}


def create_retry_controller(max_retries: int = MAX_RETRIES_PER_SUBTASK,
                            strategy: str = "corrected") -> RetryController:
    return RetryController(max_retries=max_retries, strategy=strategy)
