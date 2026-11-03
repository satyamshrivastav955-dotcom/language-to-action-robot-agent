import numpy as np
import pytest

from agent.task_agent import RobotTaskAgent
from configs.settings import BLOCK_START_POSITIONS
from env.so101_env import create_environment

pytestmark = pytest.mark.physics


@pytest.fixture(scope="module")
def env():
    e = create_environment()
    if not e.is_physics_backed:
        pytest.skip("MuJoCo scene did not load")
    yield e
    e.close()


def test_reset_places_blocks_at_configured_poses(env):
    env.reset()
    for name, pos in BLOCK_START_POSITIONS.items():
        assert np.allclose(env.get_object_pose(name), pos, atol=5e-3)


def test_save_restore_roundtrip(env):
    env.reset()
    snap = env.save_state()
    before = env.get_object_pose("red_block").copy()
    env.step(np.zeros(7))
    env.restore_state(snap)
    assert np.allclose(env.get_object_pose("red_block"), before, atol=1e-6)


def test_ik_returns_finite_joint_angles(env):
    env.reset()
    for target in ([0.28, 0.09, 0.15], [0.25, -0.12, 0.15], [0.35, 0.0, 0.2]):
        q = env.solve_ik(target)
        assert q.shape == (6,) and np.all(np.isfinite(q))


def test_multi_step_world_persists():
    with RobotTaskAgent(use_mock_policy=False, console_output=False) as agent:
        agent.planner.decompose = lambda text: {"subtasks": [
            {"id": 1, "action": "pick_and_place", "object": "red_block", "target": "bin"},
            {"id": 2, "action": "stack", "object": "blue_block", "target": "green_block"},
        ]}
        agent.planner.used_fallback = False
        result = agent.execute_instruction("test", save_videos=False)
        assert result["success"]
        red = agent.env.get_object_pose("red_block")
        assert abs(red[0] - 0.5) < 0.05 and abs(red[1]) < 0.05
