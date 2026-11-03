"""Perception from pixels: RGB-D in, estimated object poses out.

Nothing in this package reads simulator ground truth. The only inputs are the
RGB-D frame and the camera calibration returned by ``env.capture_rgbd``.
"""

from perception.camera import Detection, backproject
from perception.estimator import PerceptionSystem
from perception.segmentation import SegmentationEstimator

__all__ = ["backproject", "Detection", "SegmentationEstimator", "PerceptionSystem"]
