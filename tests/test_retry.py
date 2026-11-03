from agent.retry_controller import RetryController

FAIL = {"success": False, "reason": "x", "distance": 0.3}


def test_respects_attempt_limit():
    rc = RetryController(max_retries=3)
    assert rc.should_retry({"attempts": 2}, FAIL)
    assert not rc.should_retry({"attempts": 3}, FAIL)


def test_no_retry_after_success():
    assert not RetryController().should_retry({"attempts": 1}, {"success": True})


def test_grasp_failure_when_object_did_not_move():
    assert RetryController()._classify_failure({"object_moved": False}, FAIL) == "grasp_failure"


def test_classifies_by_distance():
    rc = RetryController()
    moved = {"object_moved": True}
    assert rc._classify_failure(moved, {**FAIL, "distance": 0.25}) == "large_placement_error"
    assert rc._classify_failure(moved, {**FAIL, "distance": 0.07}) == "minor_placement_error"
