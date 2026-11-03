import numpy as np

from agent.verifier import Verifier
from configs.settings import BLOCK_SIZE, TARGET_REGIONS


class FakeEnv:
    def __init__(self, poses):
        self.poses = {k: np.array(v, dtype=float) for k, v in poses.items()}

    def get_object_pose(self, name):
        if name not in self.poses:
            raise KeyError(name)
        return self.poses[name]


def verify(poses, subtask):
    return Verifier(FakeEnv(poses)).verify(subtask, {"object_moved": True, "frames": []})


BIN = {"action": "pick_and_place", "object": "red_block", "target": "bin"}
STACK = {"action": "stack", "object": "blue_block", "target": "green_block"}


def test_block_in_bin_passes():
    assert verify({"red_block": TARGET_REGIONS["bin"]}, BIN)["success"]


def test_block_on_table_fails():
    assert not verify({"red_block": (0.28, 0.09, 0.04)}, BIN)["success"]


def test_block_over_bin_but_too_low_fails():
    x, y, _ = TARGET_REGIONS["bin"]
    assert not verify({"red_block": (x, y, 0.01)}, BIN)["success"]


def test_unknown_object_fails_cleanly():
    assert not verify({}, BIN)["success"]


def test_stacked_passes():
    poses = {"green_block": (0.25, 0.03, 0.04), "blue_block": (0.25, 0.03, 0.04 + BLOCK_SIZE)}
    assert verify(poses, STACK)["success"]


def test_block_still_on_table_is_not_stacked():
    poses = {"green_block": (0.25, 0.03, 0.04), "blue_block": (0.25, 0.03, 0.04)}
    assert not verify(poses, STACK)["success"]


def test_stack_offset_horizontally_fails():
    poses = {"green_block": (0.25, 0.03, 0.04), "blue_block": (0.40, 0.03, 0.04 + BLOCK_SIZE)}
    assert not verify(poses, STACK)["success"]
