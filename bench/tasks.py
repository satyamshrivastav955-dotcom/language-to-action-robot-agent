"""Task suite. A task + a seed fully determine the scene and the gold plan."""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from configs.settings import BLOCK_START_POSITIONS

BLOCKS = ["red_block", "green_block", "yellow_block", "blue_block"]
TASKS = ["pick_place", "stack", "two_step"]
JITTER = 0.012      # m, uniform per axis around each slot
MIN_SPACING = 0.06  # m, between block centres (blocks are 4 cm wide)


@dataclass
class TaskSpec:
    task: str
    seed: int
    block_positions: Dict[str, tuple]
    subtasks: List[Dict]
    instruction: str


def _scene(rng: np.random.Generator) -> Dict[str, tuple]:
    slots = list(BLOCK_START_POSITIONS.values())
    for _ in range(200):
        order = rng.permutation(len(BLOCKS))
        pos = {}
        for name, k in zip(BLOCKS, order):
            x, y, z = slots[k]
            pos[name] = (x + rng.uniform(-JITTER, JITTER), y + rng.uniform(-JITTER, JITTER), z)
        pts = np.array([p[:2] for p in pos.values()])
        d = np.linalg.norm(pts[:, None] - pts[None], axis=-1) + np.eye(len(pts)) * 9
        if d.min() >= MIN_SPACING:
            return pos
    return dict(zip(BLOCKS, slots))


def make_task(task: str, seed: int) -> TaskSpec:
    rng = np.random.default_rng([seed, TASKS.index(task)])
    pos = _scene(rng)
    pick = lambda n: [BLOCKS[i] for i in rng.permutation(len(BLOCKS))[:n]]  # noqa: E731
    nice = lambda b: b.split("_")[0]  # noqa: E731

    if task == "pick_place":
        (a,) = pick(1)
        subtasks = [dict(id=1, action="pick_and_place", object=a, target="bin")]
        text = f"put the {nice(a)} block in the bin"
    elif task == "stack":
        a, b = pick(2)
        subtasks = [dict(id=1, action="stack", object=a, target=b)]
        text = f"stack the {nice(a)} block on the {nice(b)} block"
    elif task == "two_step":
        a, c, d = pick(3)
        subtasks = [dict(id=1, action="pick_and_place", object=a, target="bin"),
                    dict(id=2, action="stack", object=c, target=d)]
        text = f"put the {nice(a)} block in the bin, then stack the {nice(c)} block on the {nice(d)} block"
    else:
        raise ValueError(task)
    return TaskSpec(task, seed, pos, subtasks, text)


class StaticPlanner:
    """Returns the gold plan; isolates the verifier ablation from planner quality."""

    used_fallback = False
    backend_name = "gold"

    def __init__(self, subtasks):
        self.subtasks = subtasks

    def decompose(self, instruction):
        return {"subtasks": [dict(s, status="pending", attempts=0, result=None) for s in self.subtasks]}
