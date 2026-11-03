"""Pinhole back-projection for a MuJoCo camera (looks along -z, x right, y up)."""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

BLOCK_HALF = 0.02  # blocks are 4 cm cubes; the top face sits this far above the centre


@dataclass
class Detection:
    name: str
    xyz: np.ndarray          # estimated block CENTRE in world frame (m)
    pixel: Tuple[int, int]   # (u, v) of the mask centroid / VLM point
    n_pixels: int            # support (mask size); 0 for point-based detections

    def to_dict(self) -> Dict:
        return {"name": self.name, "xyz": [float(v) for v in self.xyz],
                "pixel": [int(self.pixel[0]), int(self.pixel[1])], "n_pixels": int(self.n_pixels)}


def intrinsics(rgbd: Dict) -> Tuple[float, float, float]:
    h, w = rgbd["height"], rgbd["width"]
    f = (h / 2.0) / np.tan(np.radians(rgbd["fovy_deg"]) / 2.0)
    return f, w / 2.0, h / 2.0


def backproject(rgbd: Dict, u, v) -> np.ndarray:
    """World-frame points for pixel coordinates (u, v) using the depth image."""
    f, cx, cy = intrinsics(rgbd)
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    d = rgbd["depth"][v.astype(int), u.astype(int)].astype(float)
    pc = np.stack([(u - cx) / f * d, -(v - cy) / f * d, -d], axis=-1)
    return rgbd["cam_pos"] + pc @ rgbd["cam_mat"].T


def top_face_center(rgbd: Dict, us: np.ndarray, vs: np.ndarray) -> np.ndarray:
    """Block centre from pixels covering it: top face = the nearest ~6 mm of depth."""
    pts = backproject(rgbd, us, vs)
    depth = rgbd["depth"][vs.astype(int), us.astype(int)]
    near = depth <= np.percentile(depth, 10) + 0.006
    top = pts[near]
    return np.array([top[:, 0].mean(), top[:, 1].mean(), np.median(top[:, 2]) - BLOCK_HALF])


def project(rgbd: Dict, xyz) -> Tuple[int, int]:
    """Pixel (u, v) of a world point; inverse of `backproject` (for overlays)."""
    f, cx, cy = intrinsics(rgbd)
    pc = rgbd["cam_mat"].T @ (np.asarray(xyz, dtype=float) - rgbd["cam_pos"])
    return int(round(cx + f * pc[0] / -pc[2])), int(round(cy - f * pc[1] / -pc[2]))
