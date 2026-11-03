"""Overlay for the verifier's view: detections, the target, and the error arrow."""

import cv2
import numpy as np

from perception.camera import project


def annotate(rgbd, detections, target_xy, obj_name) -> np.ndarray:
    img = cv2.cvtColor(rgbd["rgb"], cv2.COLOR_RGB2BGR).copy()
    for name, det in detections.items():
        u, v = project(rgbd, det.xyz)
        col = (255, 220, 120) if name == obj_name else (200, 200, 200)
        cv2.circle(img, (u, v), 14, col, 2, cv2.LINE_AA)
        cv2.putText(img, name.split("_")[0], (u + 16, v + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    if target_xy is not None:
        tu, tv = project(rgbd, [target_xy[0], target_xy[1], 0.06])
        cv2.drawMarker(img, (tu, tv), (80, 200, 120), cv2.MARKER_CROSS, 22, 2, cv2.LINE_AA)
        cv2.putText(img, "target", (tu + 14, tv - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 120), 1, cv2.LINE_AA)
        if obj_name in detections:
            ou, ov = project(rgbd, detections[obj_name].xyz)
            cv2.arrowedLine(img, (ou, ov), (tu, tv), (80, 200, 255), 1, cv2.LINE_AA, tipLength=0.08)
    return img
