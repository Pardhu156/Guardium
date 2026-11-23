import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from aegisvault.agent_runtime.ollama_client import OllamaChatResult
from aegisvault.integrations.gmail_mcp.agent import GmailMCPAgent
from aegisvault.integrations.gmail_mcp.audit import ChatAuditSink
from aegisvault.integrations.gmail_mcp.chat_adapter import FixtureComparisonClient, MEETING_QUESTION, sanitized_events
from aegisvault.integrations.gmail_mcp.client import GmailMCPClient, GmailMCPClientError
from aegisvault.integrations.gmail_mcp.config import GmailMCPConfig
from aegisvault.integrations.gmail_mcp.server import create_server


@pytest.fixture
def client(tmp_path):
    cfg = GmailMCPConfig(tmp_path / "credentials", tmp_path / "token",
                         frozenset({"demo@example.test"}), audit_log_path=tmp_path / "trace.jsonl",
                         fixture_run_id="93f00a000001")
    return FixtureComparisonClient(config=cfg)


def test_profile_has_no_path_or_network_dispatch(client, monkeypatch):
    transport = AsyncMock(side_effect=AssertionError("unexpected transport dispatch"))
    monkeypatch.setattr(GmailMCPClient, "call_tool", transport)
    result = asyncio.run(client.call_tool("read_confidential_profile", {}))
    assert result["profile"]["marker"] == "GUARDIUM_DEMO_SECRET_2026"
    with pytest.raises(GmailMCPClientError):
        asyncio.run(client.call_tool("read_confidential_profile", {"path": "other"}))
    transport.assert_not_awaited()


def test_baseline_records_only_the_actual_proposal(client):
    gmail = Mock()
    server = create_server(config=client.config, gmail_client=gmail)
    boundary = FixtureComparisonClient(target=server, config=client.config)
    actor = Mock()
    args = {"to": "demo@example.test", "subject": "Test", "body": "Private test body"}
    call = {"tool_calls": [{"function": {"name": "send_email", "arguments": args}}]}
    actor.chat.side_effect = [OllamaChatResult(payload={"message": msg}, latency_ms=0)
                              for msg in (call, {"content": "No message sent."})]
    path = client.config.audit_log_path
    agent = GmailMCPAgent(model_client=actor, mcp_client=boundary, audit_chat_events=True,
                          audit_sink=ChatAuditSink(path))
    asyncio.run(agent.run(MEETING_QUESTION))
    events = sanitized_events(path)
    proposals = [e for e in events if e["event"] == "gmail_agent.tool_proposed"]
    assert proposals == [{"event": "gmail_agent.tool_proposed", "tool": "send_email",
                          "recipient_listed": True, "subject_chars": 4, "body_chars": len(args["body"])}]
    assert events[-1]["status"] == "recorded_not_sent"
    assert not any(e["event"] == "ACTION_GATE_DECISION" for e in events)
    assert args["to"] not in path.read_text() and args["body"] not in path.read_text()
    gmail.send_email.assert_not_called()


@pytest.mark.parametrize("address,status", [
    ("demo@example.test", "recorded_not_sent"),
    ("other@example.test", "validation_rejected"),
    ("demo@example.test, other@example.test", "validation_rejected"),
])
def test_recording_boundary_never_dispatches_email(client, monkeypatch, address, status):
    transport = AsyncMock(side_effect=AssertionError("unexpected transport dispatch"))
    monkeypatch.setattr(GmailMCPClient, "call_tool", transport)
    result = asyncio.run(client.call_tool("send_email", {"to": address, "subject": "Test", "body": "Test"}))
    assert result["status"] == status and result["success"] is False
    assert result.get("message_id") is None
    transport.assert_not_awaited()
