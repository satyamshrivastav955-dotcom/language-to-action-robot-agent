import pytest

pytest.importorskip("gradio")

from dashboard.app import header_html, stats_html, timeline_html  # noqa: E402


def test_empty_timeline_prompts_the_user():
    assert "Enter an instruction" in timeline_html([])


def test_timeline_shows_verdict_and_ground_truth_per_attempt():
    html = timeline_html([{
        "id": 1, "action": "pick_and_place", "object": "red_block", "target": "bin", "status": "completed",
        "attempt_log": [
            {"attempt": 1, "verdict": False, "oracle_success": False, "observed_by": "perception"},
            {"attempt": 2, "verdict": True, "oracle_success": True, "observed_by": "perception"}]}])
    assert "Attempt 1" in html and "Attempt 2" in html and "Done" in html and "truth" in html


def test_unverified_attempt_is_marked_not_checked():
    html = timeline_html([{"id": 1, "action": "stack", "object": "blue_block", "target": "green_block",
                           "status": "completed",
                           "attempt_log": [{"attempt": 1, "verdict": True, "oracle_success": False,
                                            "observed_by": "none"}]}])
    assert "not checked" in html


def test_user_text_is_escaped():
    assert "<script>" not in header_html("<script>x</script>", "nominal", "perception", "corrected")


def test_stats_empty_and_filled():
    assert stats_html(None) == ""
    assert "2/3" in stats_html({"successful": 2, "total_subtasks": 3, "total_retries": 1, "elapsed_time": 4.2})
