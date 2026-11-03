"""Typed tool registry: each skill has a JSON-schema input and a handler.

The same registry backs the MCP server (tools/mcp_server.py) and can be called
directly from Python, so an external LLM client and the in-repo agent use one
interface to the robot.
"""

from typing import Any, Callable, Dict, List

from env.faults import PROFILES

BLOCKS = ["red_block", "green_block", "yellow_block", "blue_block"]


class ToolError(ValueError):
    pass


def _validate(schema: Dict, args: Dict) -> Dict:
    """Minimal JSON-schema check: required keys, types, enums, no extras."""
    props = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in args:
            raise ToolError(f"missing required argument: {key}")
    for key, val in args.items():
        if key not in props:
            raise ToolError(f"unknown argument: {key}")
        spec = props[key]
        typ = {"string": str, "integer": int, "number": (int, float), "boolean": bool}[spec["type"]]
        if isinstance(val, bool) and spec["type"] != "boolean" or not isinstance(val, typ):
            raise ToolError(f"{key} must be {spec['type']}")
        if "enum" in spec and val not in spec["enum"]:
            raise ToolError(f"{key} must be one of {spec['enum']}")
    return args


class Tool:
    def __init__(self, name: str, description: str, schema: Dict, handler: Callable[..., Any]):
        self.name, self.description, self.schema, self.handler = name, description, schema, handler

    def spec(self) -> Dict:
        return {"name": self.name, "description": self.description, "inputSchema": self.schema}

    def call(self, args: Dict) -> Any:
        return self.handler(**_validate(self.schema, args or {}))


def _obj(props: Dict, required: List[str]) -> Dict:
    return {"type": "object", "properties": props, "required": required, "additionalProperties": False}


def build_tools(session) -> Dict[str, Tool]:
    block = {"type": "string", "enum": BLOCKS}
    attempts = {"type": "integer", "description": "Attempt budget (1-5); each failed verification triggers a retry."}
    tools = [
        Tool("reset_scene", "Reset the workspace (blocks to start poses, arm home). Optional seed randomises poses.",
             _obj({"seed": {"type": "integer"}}, []), session.reset_scene),
        Tool("set_fault_profile", "Inject faults into subsequent attempts (for robustness tests).",
             _obj({"profile": {"type": "string", "enum": list(PROFILES)}}, ["profile"]),
             session.set_fault_profile),
        Tool("get_scene", "Perceive the workspace: estimated block positions (m) from the RGB-D camera.",
             _obj({}, []), session.get_scene),
        Tool("pick_and_place", "Pick a block and place it in the bin; verified from pixels, retried on failure.",
             _obj({"object": block, "target": {"type": "string", "enum": ["bin"]}, "max_attempts": attempts},
                  ["object", "target"]), session.pick_and_place),
        Tool("stack", "Pick a block and stack it on another block; verified from pixels, retried on failure.",
             _obj({"object": block, "target": block, "max_attempts": attempts}, ["object", "target"]),
             session.stack),
        Tool("run_instruction", "Plan a natural-language instruction with the local LLM and execute it closed-loop.",
             _obj({"instruction": {"type": "string"}, "max_attempts": attempts}, ["instruction"]),
             session.run_instruction),
    ]
    return {t.name: t for t in tools}
