import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
VIDEOS_DIR = os.path.join(OUTPUTS_DIR, "videos")
LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")

BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "zai.glm-5")
BEDROCK_REGION = os.environ.get("AWS_DEFAULT_REGION", "eu-north-1")

# Vision model used by the --vlm-verify second opinion. Separate from
# BEDROCK_MODEL_ID because the planner's model need not be multimodal.
VLM_MODEL_ID = os.environ.get("VLM_MODEL_ID", BEDROCK_MODEL_ID)

MAX_RETRIES_PER_SUBTASK = 3
VERIFICATION_TOLERANCE = 0.05
MAX_STEPS_PER_ROLLOUT = 100

HOME_POSITION_DEG = [8.5, 0.4, 35.2, 46.8, -50.2, 25.2]

OBJECT_GROUNDING: Dict[str, str] = {
    "red block": "red_block",
    "red": "red_block",
    "blue block": "blue_block",
    "blue": "blue_block",
    "green block": "green_block",
    "green": "green_block",
    "yellow block": "yellow_block",
    "yellow": "yellow_block",
    "bin": "bin",
    "container": "bin",
    "box": "bin",
}

# Single source of truth for target locations. These MUST match env/scene.xml:
# bin_body is at pos="0.5 0 0.04", so the bin interior floor sits at z=0.04 and
# a block resting inside it centres at roughly z=0.06.
# Previously three different bin positions were hardcoded across settings /
# verifier / so101_env, which (against a 0.05m tolerance) silently decided
# pass-vs-fail. Do not re-hardcode these anywhere; import them.
TARGET_REGIONS: Dict[str, Tuple[float, float, float]] = {
    "bin": (0.5, 0.0, 0.06),
    "table_center": (0.35, 0.0, 0.04),
}

# Minimum z for an object to count as "in" the bin rather than beside it.
BIN_MIN_HEIGHT = 0.04

# Block half-extent is 0.02 in scene.xml, so stacked centres differ by 0.04.
BLOCK_SIZE = 0.04

BLOCK_START_POSITIONS: Dict[str, Tuple[float, float, float]] = {
    "red_block": (0.25, 0.1, 0.04),
    "blue_block": (0.25, -0.1, 0.04),
    "green_block": (0.15, 0.1, 0.04),
    "yellow_block": (0.15, -0.1, 0.04),
}

OBJECT_COLORS: Dict[str, Tuple[int, int, int]] = {
    "red_block": (220, 50, 50),
    "blue_block": (50, 100, 220),
    "green_block": (50, 200, 50),
    "yellow_block": (240, 200, 50),
}

PLANNER_SYSTEM_PROMPT = """You are a robot task planner that decomposes high-level instructions into structured subtasks.

Given a natural language instruction, output a JSON plan with ordered subtasks.

Each subtask must be one of:
- "pick_and_place": Pick up an object and place it on/at a target
- "stack": Place an object on top of another object
- "push": Push an object to a target location

Available objects: red block, blue block, green block, yellow block, bin

Output ONLY valid JSON in this exact format:
{
  "subtasks": [
    {"id": 1, "action": "pick_and_place", "object": "red block", "target": "bin"},
    {"id": 2, "action": "stack", "object": "blue block", "target": "green block"}
  ]
}

Rules:
1. Use lowercase object names exactly as listed above
2. Always include "id" starting from 1
3. For "stack", the target is another block to stack on top of
4. For "pick_and_place", the target can be "bin" or a position
5. Output ONLY the JSON, no other text"""

PLANNER_EXAMPLES = [
    {
        "instruction": "put the red block in the bin",
        "plan": {"subtasks": [{"id": 1, "action": "pick_and_place", "object": "red block", "target": "bin"}]}
    },
    {
        "instruction": "stack the blue block on the green block",
        "plan": {"subtasks": [{"id": 1, "action": "stack", "object": "blue block", "target": "green block"}]}
    },
    {
        "instruction": "put red in bin then stack blue on green",
        "plan": {"subtasks": [
            {"id": 1, "action": "pick_and_place", "object": "red block", "target": "bin"},
            {"id": 2, "action": "stack", "object": "blue block", "target": "green block"}
        ]}
    },
    {
        "instruction": "move all blocks to the bin",
        "plan": {"subtasks": [
            {"id": 1, "action": "pick_and_place", "object": "red block", "target": "bin"},
            {"id": 2, "action": "pick_and_place", "object": "blue block", "target": "bin"},
            {"id": 3, "action": "pick_and_place", "object": "green block", "target": "bin"},
            {"id": 4, "action": "pick_and_place", "object": "yellow block", "target": "bin"}
        ]}
    },
]

VLM_VERIFIER_PROMPT = """Look at this image from a robot workspace. Answer the question about whether the task was completed successfully.

Task: {task_description}

Answer with:
1. "SUCCESS" or "FAIL"
2. A brief explanation (one sentence)

Format: SUCCESS/FAIL: explanation"""
