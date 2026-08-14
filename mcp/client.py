"""Tiny MCP demo client that calls the AegisVault-protected tool."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "mcp" / "server.py"


def call_tool(tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Start the demo server and call one tool."""

    process = subprocess.Popen(
        [sys.executable, str(SERVER)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=str(ROOT),
    )
    assert process.stdin is not None
    assert process.stdout is not None
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments or {}},
        },
        {"jsonrpc": "2.0", "id": 3, "method": "shutdown", "params": {}},
    ]
    for request in requests:
        process.stdin.write(json.dumps(request) + "\n")
    process.stdin.flush()
    process.stdin.close()
    responses = [json.loads(process.stdout.readline()) for _ in requests]
    stderr = process.stderr.read() if process.stderr is not None else ""
    return_code = process.wait(timeout=10)
    if return_code != 0:
        raise RuntimeError(f"MCP demo server exited with {return_code}: {stderr}")
    return responses[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Call the optional AegisVault MCP demo server.")
    parser.add_argument("tool", nargs="?", default="get_policy_summary", help="Tool name to call.")
    args = parser.parse_args()

    response = call_tool(args.tool)
    result = response.get("result", {})
    print("User request -> MCP client -> MCP tool request -> AegisVault Action Gate")
    print(f"Action Gate verdict: {result.get('verdict')}")
    print(f"Allowed: {result.get('allowed')} | Executed: {result.get('executed')}")
    print(json.dumps(response, indent=2, sort_keys=True))
    return 0 if result.get("allowed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())

