"""Browser-playable video writer (H.264 / yuv420p), with an OpenCV fallback."""

import os
from typing import List, Optional

import cv2
import numpy as np


def write_mp4(frames: List[np.ndarray], path: str, fps: int = 25) -> Optional[str]:
    """Write RGB frames to `path`. Returns the path, or None on failure."""
    if not frames:
        return None
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    try:
        import imageio.v2 as imageio
        h, w = frames[0].shape[:2]
        # H.264 needs even dimensions; crop a row/column rather than rescale.
        frames = [f[: h - h % 2, : w - w % 2] for f in frames]
        writer = imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p",
                                    macro_block_size=1, quality=7,
                                    ffmpeg_params=["-movflags", "+faststart"])
        for f in frames:
            writer.append_data(f)
        writer.close()
        return path
    except Exception as e:
        print(f"[video] H.264 encode failed ({type(e).__name__}: {e}); using OpenCV mp4v")
    try:
        h, w = frames[0].shape[:2]
        out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        out.release()
        return path
    except Exception as e:
        print(f"[video] could not write {path}: {e}")
        return None
