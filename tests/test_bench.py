import numpy as np
import pytest

from agent.retry_controller import RetryController
from bench.report import build, paired
from bench.stats import mcnemar_exact, wilson
from bench.tasks import BLOCKS, MIN_SPACING, TASKS, make_task
from env.faults import PROFILES, attempt_rng


def test_wilson_interval_contains_rate_and_widens_for_small_n():
    lo, hi = wilson(9, 10)
    assert lo < 0.9 < hi
    assert (hi - lo) > (wilson(90, 100)[1] - wilson(90, 100)[0])
    assert wilson(0, 0) == (0.0, 0.0)


def test_mcnemar_exact():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 0) == pytest.approx(2 * 0.5 ** 10)
    assert mcnemar_exact(5, 5) == 1.0


@pytest.mark.parametrize("task", TASKS)
def test_tasks_are_deterministic_and_well_spaced(task):
    a, b = make_task(task, 7), make_task(task, 7)
    assert a.block_positions == b.block_positions and a.subtasks == b.subtasks
    pts = np.array([a.block_positions[n][:2] for n in BLOCKS])
    d = np.linalg.norm(pts[:, None] - pts[None], axis=-1) + np.eye(4) * 9
    assert d.min() >= MIN_SPACING - 1e-9


def test_two_step_uses_three_distinct_blocks():
    st = make_task("two_step", 3).subtasks
    blocks = [st[0]["object"], st[1]["object"], st[1]["target"]]
    assert len(set(blocks)) == 3


def test_attempt_rng_depends_only_on_seed_subtask_attempt():
    assert attempt_rng(1, 0, 2).random() == attempt_rng(1, 0, 2).random()
    assert attempt_rng(1, 0, 2).random() != attempt_rng(1, 0, 3).random()


def test_fault_profiles_declared():
    assert set(PROFILES) == {"nominal", "noisy", "biased", "slippery"}
    assert PROFILES["nominal"].slip_prob == 0 and PROFILES["biased"].place_bias != (0.0, 0.0)


FAIL = {"success": False, "error_vector": [0.05, -0.02]}


def test_corrected_retry_accumulates_negative_error():
    rc = RetryController(strategy="corrected", alpha=0.8)
    sub = {"id": 1}
    c = rc.observe(sub, {"grasp_detected": True}, FAIL)
    assert c == pytest.approx([-0.04, 0.016])
    assert rc.get_adjusted_parameters(sub, 2)["place_correction"][0] == pytest.approx(-0.04)


def test_blind_retry_never_corrects():
    rc = RetryController(strategy="blind")
    assert rc.observe({"id": 1}, {"grasp_detected": True}, FAIL) is None
    assert "place_correction" not in rc.get_adjusted_parameters({"id": 1}, 2)


def test_no_correction_when_block_was_not_carried_or_landed_far_away():
    rc = RetryController(strategy="corrected")
    assert rc.observe({"id": 1}, {"grasp_detected": False}, FAIL) is None
    assert rc.observe({"id": 1}, {"grasp_detected": True},
                      {"success": False, "error_vector": [0.3, 0.0]}) is None


def _ep(task, seed, arm, ok, profile="biased"):
    return {"task": task, "seed": seed, "profile": profile, "arm": arm, "oracle_success": ok,
            "believed_success": True, "attempts": 1, "rows": []}


def test_report_pairs_by_task_and_seed():
    eps = [_ep("stack", s, "verifier_blind", False) for s in range(5)] + \
          [_ep("stack", s, "verifier_corrected", True) for s in range(5)]
    p = paired(eps, "biased", "verifier_corrected", "verifier_blind")
    assert (p["n_pairs"], p["only_a"], p["only_b"]) == (5, 5, 0)
    res = build({"meta": {"episodes": 5, "attempt_budget": 3}, "episodes": eps})
    assert res["pooled"]["biased"]["verifier_corrected"]["rate"] == 1.0
