"""PerceptionSystem: park the arm, capture RGB-D, estimate block poses."""

from typing import Dict, Optional

import numpy as np

from perception.camera import Detection
from perception.segmentation import SegmentationEstimator


class PerceptionSystem:
    """The agent's only source of object poses when running in perception mode."""

    def __init__(self, env, estimator=None, rgb_noise: float = 2.0, depth_noise: float = 0.002,
                 seed: int = 0):
        self.env = env
        self.estimator = estimator or SegmentationEstimator()
        self.rgb_noise, self.depth_noise = rgb_noise, depth_noise
        self.rng = np.random.default_rng(seed)
        self.n_captures = 0
        self.last: Dict[str, Detection] = {}
        self.last_rgbd: Optional[Dict] = None

    def perceive(self, park: bool = True) -> Dict[str, Detection]:
        if park:
            self.env.park_arm()
        rgbd = self.env.capture_rgbd(rgb_noise=self.rgb_noise, depth_noise=self.depth_noise,
                                     rng=self.rng)
        self.n_captures += 1
        self.last_rgbd = rgbd
        self.last = self.estimator.estimate(rgbd)
        return self.last

    def pose(self, name: str) -> Optional[np.ndarray]:
        det = self.last.get(name)
        return None if det is None else det.xyz
