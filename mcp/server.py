"""Minimal optional MCP-style stdio server for AegisVault."""

from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path
from typing import Any, TextIO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_TOOLS_MODULE = "aegisvault_mcp_demo_tools"
_TOOLS_PATH = ROOT / "mcp" / "tools.py"


def _load_demo_tools() -> Any:
    spec = importlib.util.spec_from_file_location(_TOOLS_MODULE, _TOOLS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load MCP demo tools from {_TOOLS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_TOOLS_MODULE] = module
    spec.loader.exec_module(module)
    return module


_tools = _load_demo_tools()
MCPToolGuard = _tools.MCPToolGuard
build_demo_guard = _tools.build_demo_guard


def handle_request(request: dict[str, Any], guard: MCPToolGuard) -> dict[str, Any]:
    """Handle one JSON-RPC-style request."""

    request_id = request.get("id")
    method = request.get("method")
    try:
        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "server": "aegisvault-mcp-demo",
                    "tools": [_tool_descriptor()],
                },
            }
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": [_tool_descriptor()]}}
        if method == "tools/call":
            params = dict(request.get("params") or {})
            name = str(params.get("name", ""))
            arguments = params.get("arguments")
            if arguments is not None and not isinstance(arguments, dict):
                return _error(request_id, -32602, "params.arguments must be an object")
            result = guard.authorize_and_execute(name, arguments)
            return {"jsonrpc": "2.0", "id": request_id, "result": result.to_dict()}
        if method == "shutdown":
            return {"jsonrpc": "2.0", "id": request_id, "result": {"ok": True}}
        return _error(request_id, -32601, f"Unknown method {method!r}")
    except Exception as exc:
        return _error(request_id, -32000, f"MCP demo server error: {exc}")


def serve(stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> int:
    """Run the stdio JSON-RPC loop."""

    guard = build_demo_guard()
    for raw_line in stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            response = _error(None, -32700, f"Invalid JSON: {exc}")
        else:
            if not isinstance(request, dict):
                response = _error(None, -32600, "Request must be a JSON object")
            else:
                response = handle_request(request, guard)
        stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        stdout.flush()
        if isinstance(request, dict) and request.get("method") == "shutdown":
            break
    return 0


def main() -> int:
    return serve()


def _tool_descriptor() -> dict[str, Any]:
    return {
        "name": "get_policy_summary",
        "description": "Return a read-only summary of the active AegisVault policy.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    }


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


if __name__ == "__main__":
    raise SystemExit(main())
