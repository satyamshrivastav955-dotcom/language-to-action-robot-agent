import pytest

from agent.planner import BasePlanner


@pytest.fixture
def planner():
    return BasePlanner()


def plan(planner, text):
    return planner._create_fallback_plan(text)["subtasks"]


def test_single_pick_and_place(planner):
    (st,) = plan(planner, "put the green block in the bin")
    assert (st["action"], st["object"], st["target"]) == ("pick_and_place", "green_block", "bin")


def test_stack(planner):
    (st,) = plan(planner, "stack the blue block on the green block")
    assert (st["action"], st["object"], st["target"]) == ("stack", "blue_block", "green_block")


def test_multi_step_order(planner):
    sts = plan(planner, "put red in the bin then stack blue on green")
    assert [s["action"] for s in sts] == ["pick_and_place", "stack"]
    assert [s["object"] for s in sts] == ["red_block", "blue_block"]
    assert [s["id"] for s in sts] == [1, 2]


def test_ambiguous_instruction_asks(planner):
    result = planner._create_fallback_plan("put the block in the bin")
    assert result["clarification_needed"] is True
    assert result["subtasks"] == []


def test_plural_blocks_is_not_ambiguous(planner):
    assert planner._detect_ambiguity("put the blocks in the bin") is None


def test_grounding_synonyms(planner):
    assert planner._ground_object("Red Block") == "red_block"
    assert planner._validate_target("container") == "bin"


def test_validate_replaces_unknown_action(planner):
    out = planner._validate_and_ground(
        {"subtasks": [{"id": 1, "action": "juggle", "object": "red block", "target": "bin"}]})
    assert out["subtasks"][0]["action"] == "pick_and_place"


def test_extract_json_from_fenced_output(planner):
    assert planner._extract_json('```json\n{"a": 1}\n```') == {"a": 1}
