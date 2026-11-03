import io
import json

import pytest

from tools.mcp_server import McpServer
from tools.registry import ToolError, build_tools


class FakeSession:
    def __init__(self):
        self.calls = []

    def reset_scene(self, seed=None):
        self.calls.append(("reset", seed))
        return {"ok": True}

    def set_fault_profile(self, profile):
        return {"ok": True, "profile": profile}

    def get_scene(self):
        return {"blocks": {}}

    def pick_and_place(self, object, target, max_attempts=3):
        self.calls.append(("pp", object, target, max_attempts))
        return {"success": True}

    def stack(self, object, target, max_attempts=3):
        return {"success": True}

    def run_instruction(self, instruction, max_attempts=3):
        return {"success": True}


@pytest.fixture
def server():
    return McpServer(FakeSession())


def rpc(server, method, params=None, mid=1):
    return server.handle({"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}})


def test_tools_have_typed_schemas():
    for tool in build_tools(FakeSession()).values():
        assert tool.schema["type"] == "object" and tool.schema["additionalProperties"] is False


def test_initialize_and_list(server):
    assert rpc(server, "initialize")["result"]["serverInfo"]["name"]
    names = {t["name"] for t in rpc(server, "tools/list")["result"]["tools"]}
    assert {"pick_and_place", "stack", "get_scene", "run_instruction"} <= names


def test_call_valid(server):
    res = rpc(server, "tools/call", {"name": "pick_and_place",
                                     "arguments": {"object": "red_block", "target": "bin"}})
    assert res["result"]["isError"] is False
    assert json.loads(res["result"]["content"][0]["text"]) == {"success": True}


@pytest.mark.parametrize("args", [
    {"object": "purple_block", "target": "bin"},      # not in enum
    {"object": "red_block"},                          # missing target
    {"object": "red_block", "target": "bin", "x": 1},  # unknown argument
    {"object": 5, "target": "bin"},                   # wrong type
    {"object": "red_block", "target": "bin", "max_attempts": True},  # bool is not an integer
])
def test_call_invalid_is_tool_error_not_crash(server, args):
    res = rpc(server, "tools/call", {"name": "pick_and_place", "arguments": args})
    assert res["result"]["isError"] is True


def test_unknown_tool_and_method(server):
    assert rpc(server, "tools/call", {"name": "fly"})["error"]["code"] == -32602
    assert rpc(server, "nope")["error"]["code"] == -32601


def test_notification_gets_no_reply(server):
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_stdio_loop_roundtrip(server):
    out = io.StringIO()
    inp = io.StringIO(json.dumps({"jsonrpc": "2.0", "id": 7, "method": "ping"}) + "\n")
    server.serve(inp, out)
    assert json.loads(out.getvalue())["id"] == 7


def test_validate_rejects_bad_enum():
    tool = build_tools(FakeSession())["set_fault_profile"]
    with pytest.raises(ToolError):
        tool.call({"profile": "chaos"})
