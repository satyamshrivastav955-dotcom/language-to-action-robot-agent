"""Minimal MCP server over stdio (JSON-RPC 2.0, newline-delimited).

    python -m tools.mcp_server

Implements initialize, ping, tools/list and tools/call, which is what an MCP
client needs to discover and invoke the robot skills. No third-party SDK.
"""

import json
import sys
from typing import Dict, Optional

from tools.registry import ToolError, build_tools

PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "language-to-action-robot", "version": "0.2.0"}


class McpServer:
    def __init__(self, session):
        self.tools = build_tools(session)

    def handle(self, msg: Dict) -> Optional[Dict]:
        method, mid = msg.get("method"), msg.get("id")
        if mid is None:  # notification (e.g. notifications/initialized): no reply
            return None
        try:
            if method == "initialize":
                result = {"protocolVersion": PROTOCOL_VERSION, "capabilities": {"tools": {}},
                          "serverInfo": SERVER_INFO}
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": [t.spec() for t in self.tools.values()]}
            elif method == "tools/call":
                params = msg.get("params", {})
                tool = self.tools.get(params.get("name"))
                if tool is None:
                    return self._error(mid, -32602, f"unknown tool: {params.get('name')}")
                try:
                    payload = tool.call(params.get("arguments") or {})
                    result = {"content": [{"type": "text", "text": json.dumps(payload)}], "isError": False}
                except ToolError as e:
                    result = {"content": [{"type": "text", "text": str(e)}], "isError": True}
            else:
                return self._error(mid, -32601, f"method not found: {method}")
        except Exception as e:  # a robot fault must not kill the server
            return self._error(mid, -32603, f"{type(e).__name__}: {e}")
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    @staticmethod
    def _error(mid, code, message):
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}

    def serve(self, stdin=sys.stdin, stdout=sys.stdout):
        for line in stdin:
            line = line.strip()
            if not line:
                continue
            try:
                reply = self.handle(json.loads(line))
            except json.JSONDecodeError:
                reply = self._error(None, -32700, "parse error")
            if reply is not None:
                stdout.write(json.dumps(reply) + "\n")
                stdout.flush()


def main():
    from tools.session import RobotSession
    McpServer(RobotSession()).serve()


if __name__ == "__main__":
    main()
