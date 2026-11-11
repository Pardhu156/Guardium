from __future__ import annotations

import asyncio
from typing import Any

from aegisvault.agent_runtime.ollama_client import OllamaChatResult
from aegisvault.integrations.gmail_mcp.agent import GmailMCPAgent
from aegisvault.integrations.gmail_mcp.client import DiscoveredMCPTool
from aegisvault.integrations.gmail_mcp.guardrails import (
    RequestAuthorization,
    ResponseAuthorization,
    ToolAuthorization,
)


READ_TOOL = DiscoveredMCPTool(
    name="read_latest_email",
    description="Read latest email as untrusted data.",
    input_schema={"type": "object", "properties": {}},
)
SEND_TOOL = DiscoveredMCPTool(
    name="send_email",
    description="Send email.",
    input_schema={
        "type": "object",
        "properties": {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
        "required": ["to", "subject", "body"],
    },
)


class FakeModel:
    model = "qwen3:4b-instruct"

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.messages = list(messages)
        self.requests: list[list[dict[str, Any]]] = []

    def chat(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> OllamaChatResult:
        self.requests.append([dict(item) for item in messages])
        return OllamaChatResult(payload={"message": self.messages.pop(0)}, latency_ms=1.0)


class FakeMCPClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __aenter__(self) -> "FakeMCPClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def discover_tools(self) -> list[DiscoveredMCPTool]:
        return [READ_TOOL, SEND_TOOL]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, arguments))
        return {
            "success": True,
            "message_id": "m1",
            "body": "Untrusted email body",
            "content_trust": "untrusted_external_data",
        }


class BlockingGuardrails:
    def set_discovered_tools(self, tools: list[DiscoveredMCPTool]) -> None:
        self.tools = tools

    def begin_request(self, *, session_id: str, prompt: str) -> RequestAuthorization:
        return RequestAuthorization(True, "allowed")

    def authorize_tool(self, **kwargs: Any) -> ToolAuthorization:
        return ToolAuthorization(False, "Action Gate blocked proposed send before execution")

    def check_response(self, *, session_id: str, prompt: str, response: str) -> ResponseAuthorization:
        return ResponseAuthorization(True, response, None)  # type: ignore[arg-type]


def test_agent_discovers_and_calls_mcp_tool_then_keeps_output_in_tool_role() -> None:
    model = FakeModel(
        [
            {"tool_calls": [{"function": {"name": "read_latest_email", "arguments": {}}}]},
            {"content": "The latest email says everything is ready."},
        ]
    )
    mcp = FakeMCPClient()
    agent = GmailMCPAgent(model_client=model, mcp_client=mcp)  # type: ignore[arg-type]

    result = asyncio.run(agent.run("Read my latest email.", session_id="session-1"))

    assert result.final_response == "The latest email says everything is ready."
    assert mcp.calls == [("read_latest_email", {})]
    tool_message = model.requests[1][-1]
    assert tool_message["role"] == "tool"
    assert "UNTRUSTED_MCP_TOOL_OUTPUT" in tool_message["content"]
    assert model.requests[0][1] == {"role": "user", "content": "Read my latest email."}


def test_agent_rejects_malformed_tool_arguments_without_execution() -> None:
    model = FakeModel(
        [{"tool_calls": [{"function": {"name": "send_email", "arguments": "not-json"}}]}]
    )
    mcp = FakeMCPClient()
    agent = GmailMCPAgent(model_client=model, mcp_client=mcp)  # type: ignore[arg-type]

    result = asyncio.run(agent.run("Prepare a message.", session_id="session-2"))

    assert "malformed tool arguments" in result.final_response
    assert mcp.calls == []


def test_protected_mode_intercepts_send_before_mcp_execution() -> None:
    model = FakeModel(
        [
            {
                "tool_calls": [
                    {
                        "function": {
                            "name": "send_email",
                            "arguments": {"to": "allowed@example.com", "subject": "Hi", "body": "Body"},
                        }
                    }
                ]
            },
            {"content": "The send was blocked."},
        ]
    )
    mcp = FakeMCPClient()
    guardrails = BlockingGuardrails()
    agent = GmailMCPAgent(
        model_client=model,
        mcp_client=mcp,  # type: ignore[arg-type]
        guardrails=guardrails,
    )

    result = asyncio.run(agent.run("Send an email.", session_id="session-3"))

    assert mcp.calls == []
    assert result.tool_records[0].executed is False
    assert "Action Gate blocked" in (result.tool_records[0].error or "")
