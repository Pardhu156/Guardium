from __future__ import annotations

import subprocess
import sys
from typing import Any

import pytest

from aegisvault.agent_runtime import AgentRuntime, default_tool_registry
from aegisvault.agent_runtime.exceptions import ToolRegistryError
from aegisvault.agent_runtime.mock_tools import calculator, clear_notes, list_notes, save_note
from aegisvault.agent_runtime.ollama_client import OllamaChatResult
from aegisvault.agent_runtime.parser import parse_tool_calls
from aegisvault.agent_runtime.tools import ToolDefinition, ToolRegistry


class FakeClient:
    model = "fake-qwen"

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> OllamaChatResult:
        self.calls += 1
        if self.calls == 1:
            return OllamaChatResult(
                payload={
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"function": {"name": "calculator", "arguments": {"expression": "42*18"}}}],
                    }
                },
                latency_ms=1.0,
            )
        return OllamaChatResult(payload={"message": {"role": "assistant", "content": "The result is 756."}}, latency_ms=1.0)


def test_tool_registry_register_and_execute() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="hello",
            description="Say hello",
            parameters={"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
            function=lambda name: f"hello {name}",
        )
    )
    result = registry.execute("hello", {"name": "Ada"})
    assert result.result == "hello Ada"
    assert result.error is None
    with pytest.raises(ToolRegistryError):
        registry.register(registry.get("hello"))


def test_tool_execution_missing_parameter_records_error() -> None:
    result = default_tool_registry().execute("calculator", {})
    assert result.error is not None
    assert "missing required" in result.error


def test_mock_tools() -> None:
    clear_notes()
    assert calculator("42 * 18") == 756.0
    assert "Saved" in save_note("Meeting at 5")
    assert list_notes() == ["Meeting at 5"]


def test_native_tool_call_parser() -> None:
    calls = parse_tool_calls({"tool_calls": [{"function": {"name": "echo", "arguments": {"text": "hi"}}}]})
    assert calls[0].name == "echo"
    assert calls[0].arguments == {"text": "hi"}


def test_json_fallback_tool_call_parser() -> None:
    calls = parse_tool_calls({"content": '{"tool_calls":[{"name":"echo","arguments":{"text":"hi"}}]}'})
    assert calls[0].name == "echo"


def test_json_scalar_content_is_not_a_tool_call() -> None:
    assert parse_tool_calls({"content": "756"}) == []


def test_runtime_executes_tool_and_traces() -> None:
    runtime = AgentRuntime(client=FakeClient(), tools=default_tool_registry())
    result = runtime.run("Calculate 42 * 18")
    assert result.final_response == "The result is 756."
    assert [record.tool_name for record in result.tool_records] == ["calculator"]
    assert any(event.event_type == "tool_execution" for event in result.trace.events)
    assert result.trace.total_latency_ms is not None


def test_cli_lists_tools() -> None:
    completed = subprocess.run(
        [sys.executable, "run_agent.py", "--list-tools"],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "calculator" in completed.stdout
    assert "get_time" in completed.stdout
