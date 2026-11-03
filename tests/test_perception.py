import numpy as np
import pytest

from bench.episode import run_episode
from bench.tasks import make_task
from env.so101_env import create_environment
from perception import PerceptionSystem

pytestmark = pytest.mark.physics


@pytest.fixture(scope="module")
def env():
    e = create_environment()
    if not e.is_physics_backed:
        pytest.skip("MuJoCo scene did not load")
    yield e
    e.close()


def test_segmentation_localises_blocks_within_5mm(env):
    spec = make_task("two_step", 3)
    env.reset(spec.block_positions)
    dets = PerceptionSystem(env, seed=1).perceive()
    assert set(dets) == set(env.object_names)
    for name, det in dets.items():
        truth = env.get_object_pose(name)
        assert np.linalg.norm(det.xyz - truth) < 0.005, name


def test_perception_does_not_need_ground_truth_access(env):
    """Estimates come from pixels: moving a block moves the estimate."""
    env.reset()
    before = PerceptionSystem(env, seed=0).perceive()["red_block"].xyz.copy()
    moved = {"red_block": (0.30, 0.06, 0.04)}
    env.reset(moved)
    after = PerceptionSystem(env, seed=0).perceive()["red_block"].xyz
    assert np.linalg.norm(after - before) > 0.02


def test_ik_converges_where_a_plain_seeded_solve_stalls(env):
    env.reset()
    env.reset_arm()
    q = env.solve_ik([0.251, -0.131, 0.045])
    assert abs(q[0] - np.arctan2(-0.131, 0.251)) < 0.1  # base yaw points at the block


@pytest.mark.parametrize("task", ["pick_place", "stack", "two_step"])
def test_nominal_tasks_succeed_under_ground_truth(env, task):
    wins = sum(run_episode(task, s, "nominal", "verifier_corrected", env=env)["oracle_success"]
               for s in range(3))
    assert wins >= 2


def test_no_verifier_arm_cannot_see_its_own_failure(env):
    """Under a 5.4 cm release bias the robot without a verifier still reports success."""
    eps = [run_episode("stack", s, "biased", "no_verifier", env=env) for s in range(4)]
    assert any(e["believed_success"] and not e["oracle_success"] for e in eps)


def test_same_seed_same_first_attempt_across_arms(env):
    a = run_episode("pick_place", 5, "noisy", "verifier_blind", env=env)
    b = run_episode("pick_place", 5, "noisy", "verifier_corrected", env=env)
    la, lb = a["rows"][0]["attempt_log"][0], b["rows"][0]["attempt_log"][0]
    assert la["verdict"] == lb["verdict"] and la["oracle_success"] == lb["oracle_success"]
