"""VLM grounding through a local Ollama server (e.g. qwen3-vl:4b).

The VLM only returns a 2-D point for "the <colour> cube"; depth then lifts it
to 3-D exactly as for the segmentation baseline, so any accuracy difference
between the two estimators is attributable to 2-D localisation.
"""

import base64
import json
import re
from typing import Dict, Optional

import cv2
import numpy as np
import requests

from perception.camera import BLOCK_HALF, Detection, backproject

COLOR_OF = {"red_block": "red", "green_block": "green", "yellow_block": "yellow", "blue_block": "blue"}


class OllamaVLMGrounder:
    def __init__(self, model: str = "qwen3-vl:4b", host: str = "http://localhost:11434",
                 timeout: float = 300.0):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout
        self.last_latency_s = 0.0
        self.last_raw = ""

    def _ask(self, jpg_b64: str, color: str) -> str:
        prompt = (f"Locate the {color} cube on the brown table. Answer with JSON only: "
                  f'{{"point_2d": [x, y]}} where x and y are the centre of the {color} cube '
                  "in coordinates normalised to 0-1000 (x to the right, y downward).")
        payload = {"model": self.model, "stream": False, "think": False,
                   "messages": [{"role": "user", "content": prompt, "images": [jpg_b64]}],
                   "options": {"temperature": 0.0, "num_predict": 128}}
        r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        r.raise_for_status()
        body = r.json()
        self.last_latency_s = body.get("total_duration", 0) / 1e9
        self.last_raw = body["message"]["content"]
        return self.last_raw

    @staticmethod
    def parse_point(text: str) -> Optional[tuple]:
        num = r"(-?[\d.]+)"
        m = re.search(rf'"point_2d"\s*:\s*\[\s*{num}\s*,\s*{num}\s*\]', text)
        if m:
            return float(m.group(1)), float(m.group(2))
        m = re.search(rf'"bbox_2d"\s*:\s*\[\s*{num}\s*,\s*{num}\s*,\s*{num}\s*,\s*{num}', text)
        if m:
            x1, y1, x2, y2 = map(float, m.groups())
            return (x1 + x2) / 2, (y1 + y2) / 2
        try:
            pt = json.loads(text).get("point_2d")
            return float(pt[0]), float(pt[1])
        except Exception:
            return None

    def estimate(self, rgbd: Dict, names: Optional[list] = None) -> Dict[str, Detection]:
        bgr = cv2.cvtColor(rgbd["rgb"], cv2.COLOR_RGB2BGR)
        jpg = base64.b64encode(cv2.imencode(".jpg", bgr)[1].tobytes()).decode()
        h, w = rgbd["height"], rgbd["width"]
        out: Dict[str, Detection] = {}
        for name in (names or list(COLOR_OF)):
            pt = self.parse_point(self._ask(jpg, COLOR_OF[name]))
            if pt is None:
                continue
            u = int(np.clip(pt[0] / 1000.0 * w, 2, w - 3))
            v = int(np.clip(pt[1] / 1000.0 * h, 2, h - 3))
            win_u, win_v = np.meshgrid(np.arange(u - 2, u + 3), np.arange(v - 2, v + 3))
            pts = backproject(rgbd, win_u.ravel(), win_v.ravel())
            xyz = np.array([pts[:, 0].mean(), pts[:, 1].mean(), np.median(pts[:, 2]) - BLOCK_HALF])
            out[name] = Detection(name, xyz, (u, v), 0)
        return out
