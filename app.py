"""Hugging Face Spaces entry point.

Spaces launches app.py from the repo root; the real dashboard lives in
dashboard/app.py. MuJoCo has no display on Spaces hardware, so force
headless EGL rendering before anything imports mujoco.
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from dashboard.app import demo  # noqa: E402
from configs.settings import VIDEOS_DIR  # noqa: E402

if __name__ == "__main__":
    os.makedirs(VIDEOS_DIR, exist_ok=True)
    demo.launch()
