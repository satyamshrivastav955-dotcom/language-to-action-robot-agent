"""Colour segmentation + depth: the dependency-free perception baseline."""

from typing import Dict, Optional

import cv2
import numpy as np

from perception.camera import Detection, top_face_center

# OpenCV HSV: H in [0,179]. The table and bin are low-saturation browns
# (S around 90-110), so a saturation floor of 140 isolates the blocks.
HUE_RANGES = {
    "red_block": [(0, 8), (172, 179)],
    "yellow_block": [(22, 38)],
    "green_block": [(48, 75)],
    "blue_block": [(100, 125)],
}
MIN_SAT, MIN_VAL, MIN_PIXELS = 140, 110, 60


class SegmentationEstimator:
    def estimate(self, rgbd: Dict, names: Optional[list] = None) -> Dict[str, Detection]:
        hsv = cv2.cvtColor(rgbd["rgb"], cv2.COLOR_RGB2HSV)
        out: Dict[str, Detection] = {}
        for name in (names or list(HUE_RANGES)):
            mask = np.zeros(hsv.shape[:2], np.uint8)
            for lo, hi in HUE_RANGES[name]:
                mask |= cv2.inRange(hsv, (lo, MIN_SAT, MIN_VAL), (hi, 255, 255))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            n, labels, stats, cents = cv2.connectedComponentsWithStats(mask)
            if n <= 1:
                continue
            k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            if stats[k, cv2.CC_STAT_AREA] < MIN_PIXELS:
                continue
            vs, us = np.nonzero(labels == k)
            out[name] = Detection(name, top_face_center(rgbd, us, vs),
                                  (int(cents[k][0]), int(cents[k][1])), len(us))
        return out
