"""Fault profiles: controlled ways for a rollout to go wrong.

The simulator is deterministic and the scripted controller is accurate, so
without injected faults every task succeeds on attempt 1 and a verifier has
nothing to catch. A FaultProfile perturbs the *actuation* (where the arm
actually grasps/places) and the carry (slip), never the evaluator's ground
truth. All randomness comes from a numpy Generator seeded per
(episode seed, subtask index, attempt number), so two experiment arms see
identical faults on their first attempt and differ only in what they do next.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class FaultProfile:
    name: str
    description: str
    # Std-dev (m) of Gaussian xy error added to where the arm actually grasps.
    grasp_noise_std: float = 0.0
    # Std-dev (m) of Gaussian xy error added to where the arm actually releases.
    place_noise_std: float = 0.0
    # Constant xy offset (m) added to every release: a calibration error.
    place_bias: Tuple[float, float] = (0.0, 0.0)
    # Probability that a grasp slips and drops the block mid-carry.
    slip_prob: float = 0.0


PROFILES: Dict[str, FaultProfile] = {
    "nominal": FaultProfile("nominal", "No injected faults."),
    "noisy": FaultProfile(
        "noisy", "Random grasp and release position error (sigma 2.0 / 2.5 cm).",
        grasp_noise_std=0.020, place_noise_std=0.025),
    "biased": FaultProfile(
        "biased", "Constant 5.4 cm release calibration error (systematic).",
        place_bias=(0.045, -0.030)),
    "slippery": FaultProfile(
        "slippery", "Half of grasps drop the block mid-carry.", slip_prob=0.5),
}


def attempt_rng(seed: int, subtask_index: int, attempt: int) -> np.random.Generator:
    """Generator that depends only on (seed, subtask, attempt)."""
    return np.random.default_rng([int(seed), int(subtask_index), int(attempt)])
