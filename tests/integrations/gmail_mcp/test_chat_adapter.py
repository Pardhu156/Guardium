import asyncio
import json
from importlib.util import find_spec
from unittest.mock import AsyncMock, Mock

import pytest

from aegisvault.integrations.gmail_mcp import chat_adapter as chat
from aegisvault.integrations.gmail_mcp.client import GmailMCPClientError
from aegisvault.runtime.goal_vault import RedisGoalVaultBackend, SentenceTransformerGoalEmbedder
from aegisvault.types import GateDecision, GateType, Verdict


def test_chat_embedding_dependencies_import_together():
    if find_spec("streamlit") is None or find_spec("sentence_transformers") is None:
        pytest.skip("requires chat and embeddings extras")
    import pyarrow
    from sentence_transformers import SentenceTransformer

    assert pyarrow.array([1]).to_pylist() == [1]
    assert callable(SentenceTransformer)


def test_sessions_and_reset_are_isolated():
    off = chat.create_session("baseline")
    off.events.append({"event": "old"})
    on = chat.create_session("protected")
    assert on.session_id != off.session_id and chat.get_events(on) == []
    assert chat.reset_session(off).events == []


def test_read_only_client_rejects_all_outgoing_calls(monkeypatch):
    client = object.__new__(chat.ReadOnlyClient)
    dispatch = AsyncMock(return_value={"success": True, "empty": True})
    monkeypatch.setattr(chat.GmailMCPClient, "call_tool", dispatch)
    with pytest.raises(GmailMCPClientError):
        asyncio.run(client.call_tool("send_email", {"body": "secret"}))
    with pytest.raises(GmailMCPClientError):
        asyncio.run(client.call_tool("read_demo_profile", {}))
    with pytest.raises(GmailMCPClientError):
        asyncio.run(client.call_tool("read_latest_email", {"target": "other"}))
    dispatch.assert_not_called()
    asyncio.run(client.call_tool("read_latest_email", {}))
    dispatch.assert_awaited_once()


def test_chat_rejects_missing_mcp_read_result(monkeypatch):
    dispatch = AsyncMock(return_value={"success": True, "result": None})
    monkeypatch.setattr(chat.GmailMCPClient, "call_tool", dispatch)
    client = object.__new__(chat.ReadOnlyClient)
    assert asyncio.run(client.call_tool("read_latest_email", {}))["success"] is False


def test_chat_trace_exposes_only_decisions_and_an_opaque_message_reference(tmp_path):
    path = tmp_path / "audit.jsonl"
    sink = chat.ChatAuditSink(path)
    sink.record({"event_type": "gmail_agent.tool_proposed", "tool": "private-tool",
                 "arguments": {"private": "value"}, "reason": "private reason"})
    sink.record({"event_type": "sentinel.evaluated", "tool": "read_latest_email",
                 "sentinel_decision": "block", "fused_risk": 0.82068})
    assert "private" not in path.read_text()
    assert chat.sanitized_events(path) == [
        {"event": "gmail_agent.tool_proposed", "tool": "unknown"},
        {"event": "sentinel.evaluated", "tool": "read_latest_email", "decision": "block", "fused_risk": 0.821},
    ]
    # The stdio server writes its own sanitized completion event.
    server_audit = chat.JsonLineAuditSink(path)
    for mid in ["id-a", "id-a", "id-b"]:
        server_audit.record({"event_type": "gmail_mcp.tool_completed", "tool": "read_latest_email",
                             "result": {"message_id": mid, "empty": False, "success": True}})
    events = chat.sanitized_events(path)[2:]
    assert events[0]["message_ref"] == events[1]["message_ref"] != events[2]["message_ref"]
    assert "id-a" not in str(events)


def test_chat_reports_missing_read_tool(monkeypatch):
    monkeypatch.setattr(chat.GmailMCPClient, "discover_tools", AsyncMock(return_value=[]))
    with pytest.raises(GmailMCPClientError, match="discovery"):
        asyncio.run(object.__new__(chat.ReadOnlyClient).discover_tools())


def test_event_output_excludes_private_fields(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_text(json.dumps({"event_type": "ACTION_GATE_DECISION", "tool": "read_latest_email", "verdict": "BLOCK", "reason": "secret", "arguments": {"body": "secret"}}))
    assert chat.sanitized_events(path) == [{"event": "ACTION_GATE_DECISION", "tool": "read_latest_email", "decision": "BLOCK"}]


@pytest.mark.parametrize("gate,verdict,evaluator,confidence,original,expected_category", [
    (GateType.REQUEST, Verdict.CLARIFY, "ollama:private-model", 0.4, "ALLOW", "low_confidence"),
    (GateType.RESPONSE, Verdict.BLOCK, "fallback", None, None, "evaluator_failure"),
    (GateType.RESPONSE, Verdict.BLOCK, "ollama:private-model", 0.93, None, "gate_verdict"),
])
def test_chat_persists_actual_gate_outcome_without_private_content(
    tmp_path, gate, verdict, evaluator, confidence, original, expected_category
):
    metadata = {"untrusted": "private metadata"}
    if original:
        metadata["original_verdict"] = original
    decision = GateDecision(verdict, confidence, "private reason", gate, evaluator, 10, metadata)
    path = tmp_path / "events.jsonl"
    chat.ChatAuditSink(path).record_gate(decision)
    event, = chat.sanitized_events(path)
    assert event["event"] == f"{gate.value}_gate"
    assert event["decision"] == verdict.value
    assert event["source"] == evaluator.partition(":")[0]
    assert event["category"] == expected_category
    assert event.get("confidence") == confidence
    assert event.get("original_verdict") == original
    assert json.loads(path.read_text()) == {"event_type": "guardium_chat.gate_outcome", **event}
    assert "private" not in path.read_text()


def test_send_message_uses_agent_with_fresh_request_ids(monkeypatch, tmp_path):
    monkeypatch.setattr(chat, "ROOT", tmp_path)
    monkeypatch.setattr(chat.GmailMCPConfig, "from_env", lambda: chat.GmailMCPConfig(credentials_path=tmp_path / "credentials", token_path=tmp_path / "token", recipient_allowlist=frozenset()))
    agent = Mock()
    agent.run = AsyncMock(return_value=Mock(final_response="real agent answer", request_authorization=None, response_authorization=None, tool_records=[]))
    factory = Mock(return_value=agent)
    monkeypatch.setattr(chat, "GmailMCPAgent", factory)
    monkeypatch.setattr(chat, "OllamaChatClient", Mock())
    session = chat.create_session("baseline")
    assert chat.send_message(session, "hello") == "real agent answer"
    assert chat.send_message(session, "again") == "real agent answer"
    calls = agent.run.await_args_list
    assert calls[0].kwargs["session_id"] != calls[1].kwargs["session_id"]
    assert isinstance(factory.call_args.kwargs["mcp_client"], chat.ReplyProposalClient)


def test_protected_mode_passes_guardrails_and_surfaces_no_sensitive_error(monkeypatch, tmp_path):
    monkeypatch.setattr(chat, "ROOT", tmp_path)
    monkeypatch.setattr(chat.GmailMCPConfig, "from_env", lambda: chat.GmailMCPConfig(credentials_path=tmp_path / "credentials", token_path=tmp_path / "token", recipient_allowlist=frozenset()))
    guard = Mock()
    monkeypatch.setattr(chat, "build_protected_guardrails", Mock(return_value=guard))
    monkeypatch.setattr(chat, "OllamaChatClient", Mock())
    agent = Mock()
    agent.run = AsyncMock(side_effect=RuntimeError("sensitive failure"))
    factory = Mock(return_value=agent)
    monkeypatch.setattr(chat, "GmailMCPAgent", factory)
    session = chat.create_session("protected")
    with pytest.raises(RuntimeError, match="Request failed") as exc:
        chat.send_message(session, "Read the latest email")
    assert "sensitive failure" not in str(exc.value)
    assert factory.call_args.kwargs["guardrails"] is guard
    assert chat.get_events(session)[-1] == {"event": "error", "type": "RuntimeError", "nested_types": ["RuntimeError"], "stage": "agent execution"}


@pytest.mark.parametrize("component,stage,expected_types", [
    ("redis", "Goal Vault Redis storage", ["GoalBackendUnavailableError", "ConnectionError"]),
    ("embedding", "Goal Vault embedding", ["RuntimeError"]),
])
def test_goal_failure_is_identified_and_persisted_without_private_data(
    monkeypatch, tmp_path, component, stage, expected_types
):
    monkeypatch.setattr(chat, "ROOT", tmp_path)
    monkeypatch.setattr(chat, "build_protected_guardrails", Mock())
    monkeypatch.setattr(chat, "OllamaChatClient", Mock())
    config = chat.GmailMCPConfig(credentials_path=tmp_path / "c", token_path=tmp_path / "t",
                                 recipient_allowlist=frozenset())
    monkeypatch.setattr(chat.GmailMCPConfig, "from_env", lambda: config)

    def agent_factory(*, audit_sink, **kwargs):
        async def run(prompt, *, session_id):
            audit_sink.record({"event_type": "GOAL_COMMIT_ATTEMPT"})
            try:
                if component == "redis":
                    client = Mock()
                    client.set.side_effect = ConnectionError("private detail")
                    backend = RedisGoalVaultBackend(client=client)
                    monkeypatch.setattr("aegisvault.runtime.goal_vault.redis_backend.dumps_anchor", lambda _: "anchor")
                    backend.commit_anchor(Mock(session_id=session_id), 60)
                else:
                    embedder = SentenceTransformerGoalEmbedder()
                    monkeypatch.setattr(embedder, "_load_model", Mock(side_effect=RuntimeError("private detail")))
                    embedder.embed(prompt)
            except Exception as failure:
                raise ExceptionGroup("transport detail", [ExceptionGroup("inner", [failure])]) from None
        return Mock(run=run)

    monkeypatch.setattr(chat, "GmailMCPAgent", agent_factory)
    session = chat.create_session("protected")
    with pytest.raises(RuntimeError, match=stage) as caught:
        chat.send_message(session, "private prompt")
    event = session.events[-1]
    assert event["nested_types"] == expected_types
    trace = next((tmp_path / "logs" / "guardium-chat" / session.session_id).glob("*.jsonl")).read_text()
    assert json.loads(trace.splitlines()[-1]) == {
        "event_type": "guardium_chat.request_failed", "stage": stage,
        "type": "ExceptionGroup", "nested_types": expected_types,
    }
    assert "private" not in trace and "private" not in str(caught.value)
