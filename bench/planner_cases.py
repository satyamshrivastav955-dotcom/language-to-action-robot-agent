"""Gold plans for the planner comparison.

Each case: (instruction, gold) where gold is a list of (action, object, target)
steps, or "CLARIFY" when the instruction is genuinely ambiguous. `any_order`
cases are scored as sets (e.g. "move all blocks").
"""

PP, ST = "pick_and_place", "stack"
R, G, Y, B = "red_block", "green_block", "yellow_block", "blue_block"

CASES = [
    ("put the red block in the bin", [(PP, R, "bin")], False),
    ("place the green cube into the container", [(PP, G, "bin")], False),
    ("stack the blue block on the green block", [(ST, B, G)], False),
    ("put blue on yellow", [(ST, B, Y)], False),
    ("put red in the bin then stack blue on green", [(PP, R, "bin"), (ST, B, G)], False),
    ("move all the blocks to the bin", [(PP, R, "bin"), (PP, B, "bin"), (PP, G, "bin"), (PP, Y, "bin")], True),
    ("put the block in the bin", "CLARIFY", False),
    ("pick up the emerald cube and drop it in the box", [(PP, G, "bin")], False),
    ("first put yellow in the bin, and after that put red in the bin", [(PP, Y, "bin"), (PP, R, "bin")], False),
    ("stack red on yellow, then stack green on red", [(ST, R, Y), (ST, G, R)], False),
    ("put the blue block in the bin, then the green one", [(PP, B, "bin"), (PP, G, "bin")], False),
    ("place the yellow brick on top of the blue brick", [(ST, Y, B)], False),
    ("get rid of the red block", [(PP, R, "bin")], False),
    ("clear the red and blue blocks away into the bin", [(PP, R, "bin"), (PP, B, "bin")], True),
    ("move the block", "CLARIFY", False),
    ("put the green block on the red block and then the yellow block in the bin",
     [(ST, G, R), (PP, Y, "bin")], False),
    ("bin the blue one", [(PP, B, "bin")], False),
    ("stack the yellow block on the blue block then put the red block in the bin",
     [(ST, Y, B), (PP, R, "bin")], False),
]


def score(plan: dict, gold, any_order: bool) -> bool:
    if gold == "CLARIFY":
        return bool(plan.get("clarification_needed"))
    if plan.get("clarification_needed"):
        return False
    got = [(s["action"], s["object"], s["target"]) for s in plan.get("subtasks", [])]
    return sorted(got) == sorted(gold) if any_order else got == list(gold)
