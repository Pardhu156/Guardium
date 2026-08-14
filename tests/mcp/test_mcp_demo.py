from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SERVER = ROOT / "mcp" / "server.py"


def _server_roundtrip(request: dict) -> dict:
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
    process.stdin.write(json.dumps(request) + "\n")
    process.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 99, "method": "shutdown"}) + "\n")
    process.stdin.flush()
    process.stdin.close()
    response = json.loads(process.stdout.readline())
    _ = process.stdout.readline()
    stderr = process.stderr.read() if process.stderr is not None else ""
    assert process.wait(timeout=10) == 0, stderr
    return response


def test_mcp_server_starts_and_lists_tools() -> None:
    response = _server_roundtrip({"jsonrpc": "2.0", "id": 1, "method": "initialize"})

    assert response["result"]["server"] == "aegisvault-mcp-demo"
    assert response["result"]["tools"][0]["name"] == "get_policy_summary"


def test_allowed_mcp_tool_executes_after_action_gate() -> None:
    response = _server_roundtrip(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "get_policy_summary", "arguments": {}},
        }
    )

    result = response["result"]
    assert result["allowed"] is True
    assert result["executed"] is True
    assert result["verdict"] == "EXECUTE"
    assert result["result"]["application"]["name"] == "mcp-policy-summary-demo"
    assert result["action_decision"]["ollama_called"] is False


def test_unknown_mcp_tool_is_blocked_and_not_executed() -> None:
    response = _server_roundtrip(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "delete_everything", "arguments": {}},
        }
    )

    result = response["result"]
    assert result["allowed"] is False
    assert result["executed"] is False
    assert result["verdict"] == "BLOCK"
    assert result["result"] is None


def test_agentdojo_import_does_not_load_mcp_demo_modules() -> None:
    code = (
        "import sys\n"
        "import evaluation.agentdojo.run_pilot_benchmark\n"
        "print(any(name in {'mcp.server', 'mcp.client', 'mcp.tools'} for name in sys.modules))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=True,
    )

    assert completed.stdout.strip() == "False"
