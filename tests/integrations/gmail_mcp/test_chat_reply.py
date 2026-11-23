import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from aegisvault.integrations.gmail_mcp import chat_adapter as chat
from aegisvault.integrations.gmail_mcp.client import GmailMCPClient, GmailMCPClientError
from aegisvault.integrations.gmail_mcp.config import GmailMCPConfig
from aegisvault.agent_runtime.ollama_client import OllamaChatResult
from aegisvault.integrations.gmail_mcp.agent import GmailMCPAgent
from aegisvault.integrations.gmail_mcp.client import DiscoveredMCPTool
from aegisvault.integrations.gmail_mcp.guardrails import RequestAuthorization, ResponseAuthorization, ToolAuthorization


@pytest.fixture
def config(tmp_path):
    return GmailMCPConfig(tmp_path / "credentials", tmp_path / "token",
                          frozenset({"sender@example.test"}))


def test_reply_requires_read_target_and_successful_preview(config, monkeypatch):
    client = chat.ReplyProposalClient(config=config)
    call = AsyncMock(return_value={"success": True, "status": "validated_not_sent",
                                   "recipient": "sender@example.test", "subject": "Re: Meeting"})
    monkeypatch.setattr(GmailMCPClient, "call_tool", call)
    with pytest.raises(GmailMCPClientError):
        asyncio.run(client.call_tool("reply_email", {"original_id": "id", "body": "Yes"}))
    client.read_id = "id"
    result = asyncio.run(client.call_tool("reply_email", {"original_id": "id", "body": "Yes"}))
    assert result["status"] == "awaiting_operator_approval"
    assert client.pending["recipient"] == "sender@example.test"
    with pytest.raises(GmailMCPClientError):
        asyncio.run(client.call_tool("reply_email", {"original_id": "id", "body": "Yes"}))
    call.assert_awaited_once()


def test_send_proposal_uses_dry_run_preview(config, monkeypatch):
    client = chat.ReplyProposalClient(config=config)
    preview = AsyncMock(return_value={"success": True, "status": "validated_not_sent",
                                      "recipient": "sender@example.test"})
    monkeypatch.setattr(GmailMCPClient, "call_tool", preview)
    args = {"to": "sender@example.test", "subject": "Meeting", "body": "I will attend."}
    result = asyncio.run(client.call_tool("send_email", args))
    assert result["status"] == "awaiting_operator_approval"
    assert client.pending == {"tool": "send_email", "recipient": args["to"],
                              "subject": args["subject"], "body": args["body"]}
    preview.assert_awaited_once_with(client, "send_email", args)


@pytest.mark.parametrize("mode", ["baseline", "protected"])
def test_reply_binds_real_read_id_before_guard_and_preview(mode, config, monkeypatch):
    read_result = {"success": True, "empty": False, "message_id": "gmail-id",
                   "body": "Meeting", "content_trust": "untrusted_external_data"}
    preview = {"success": True, "status": "validated_not_sent",
               "recipient": "sender@example.test", "subject": "Re: Meeting"}
    transport = AsyncMock(side_effect=[read_result, preview])
    monkeypatch.setattr(GmailMCPClient, "call_tool", transport)

    class Client(chat.ReplyProposalClient):
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def discover_tools(self):
            schema = {"type": "object", "properties": {}}
            return [DiscoveredMCPTool("read_latest_email", "Read", schema),
                    DiscoveredMCPTool("reply_email", "Reply", schema)]

    client = Client(config=config)
    actor = Mock()
    messages = [
        {"tool_calls": [{"function": {"name": "read_latest_email", "arguments": {}}}]},
        {"tool_calls": [{"function": {"name": "reply_email", "arguments": {"body": "I will attend."}}}]},
        {"content": "Awaiting approval."},
    ]
    actor.chat.side_effect = [OllamaChatResult({"message": message}, 0) for message in messages]
    guard = None
    if mode == "protected":
        guard = Mock()
        guard.begin_request.return_value = RequestAuthorization(True, "ok")
        guard.authorize_tool.return_value = ToolAuthorization(True, "ok")
        guard.check_response.return_value = ResponseAuthorization(True, "Awaiting approval.", None)
    agent = GmailMCPAgent(model_client=actor, mcp_client=client, guardrails=guard)
    result = asyncio.run(agent.run("Read my email and reply that I will attend."))

    assert result.tool_records[1].arguments == {"original_id": "gmail-id", "body": "I will attend."}
    assert client.pending["original_id"] == "gmail-id"
    assert transport.await_args_list[1].args == (client, "reply_email", {
        "original_id": "gmail-id", "body": "I will attend.",
    })
    if guard:
        assert guard.authorize_tool.call_args_list[1].kwargs["arguments"]["original_id"] == "gmail-id"


def test_reply_binding_rejects_supplied_target_and_requires_read(config):
    client = chat.ReplyProposalClient(config=config)
    with pytest.raises(GmailMCPClientError, match="Read a message"):
        client.bind_tool_arguments("reply_email", {"body": "Yes"})
    client.read_id = "gmail-id"
    with pytest.raises(GmailMCPClientError, match="Invalid reply"):
        client.bind_tool_arguments("reply_email", {"body": "Yes", "original_id": "different"})
    client.pending = {"body": "Yes"}
    with pytest.raises(GmailMCPClientError, match="already pending"):
        client.bind_tool_arguments("reply_email", {"body": "Yes"})


def test_cancel_removes_proposal():
    session = chat.create_session("protected")
    session.pending_reply = {"mode": "protected"}
    chat.cancel_reply(session)
    assert session.pending_reply is None
    with pytest.raises(ValueError):
        chat.approve_reply(session)


def test_approval_dispatches_once_and_reserves_across_sessions(config, monkeypatch, tmp_path):
    monkeypatch.setattr(chat, "ROOT", tmp_path)
    monkeypatch.setattr(chat.GmailMCPConfig, "from_env", lambda: config)
    dispatch = AsyncMock(return_value={"success": True, "status": "sent", "message_id": "sent1"})
    fake_client = Mock()
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=None)
    fake_client.call_tool = dispatch
    monkeypatch.setattr(chat, "GmailMCPClient", lambda **kwargs: fake_client)
    pending = {"original_id": "id", "body": "Yes", "recipient": "sender@example.test",
               "subject": "Re: Meeting", "mode": "protected"}
    session = chat.create_session("protected")
    session.pending_reply = pending.copy()
    assert chat.approve_reply(session) == {"status": "API_ACCEPTED", "message_id": "sent1"}
    other = chat.create_session("baseline")
    other.pending_reply = {**pending, "mode": "baseline"}
    with pytest.raises(ValueError, match="already reserved"):
        chat.approve_reply(other)
    dispatch.assert_awaited_once()
    assert not any("sender@example.test" in p.read_text() for p in Path(tmp_path).rglob("*") if p.is_file())


def test_failure_keeps_reservation_and_prevents_retry(config, monkeypatch, tmp_path):
    monkeypatch.setattr(chat, "ROOT", tmp_path)
    monkeypatch.setattr(chat.GmailMCPConfig, "from_env", lambda: config)
    fake_client = Mock()
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=None)
    fake_client.call_tool = AsyncMock(side_effect=RuntimeError("unavailable"))
    monkeypatch.setattr(chat, "GmailMCPClient", lambda **kwargs: fake_client)
    session = chat.create_session("baseline")
    pending = {"original_id": "id", "body": "Yes", "recipient": "sender@example.test",
               "subject": "Re: Meeting", "mode": "baseline"}
    session.pending_reply = pending.copy()
    with pytest.raises(RuntimeError, match="unknown"):
        chat.approve_reply(session)
    session.pending_reply = pending.copy()
    with pytest.raises(ValueError, match="already reserved"):
        chat.approve_reply(session)


def test_agent_reply_client_rejects_live_configuration(config):
    from dataclasses import replace

    live = replace(config, dry_run=False, live_send_approved=True)
    with pytest.raises(ValueError, match="dry-run"):
        chat.ReplyProposalClient(config=live)


def test_protected_chat_requires_approval(config, monkeypatch, tmp_path):
    monkeypatch.setattr(chat, "ROOT", tmp_path)
    monkeypatch.setattr(chat.GmailMCPConfig, "from_env", lambda: config)
    monkeypatch.setattr(chat, "OllamaChatClient", Mock())
    monkeypatch.setattr(chat, "build_protected_guardrails", Mock())
    pending = {"original_id": "id", "body": "Yes", "recipient": "sender@example.test",
               "subject": "Re: Meeting"}
    monkeypatch.setattr(chat, "ReplyProposalClient", lambda **kwargs: Mock(pending=pending))
    agent = Mock()
    agent.run = AsyncMock(return_value=Mock(final_response="No message sent."))
    monkeypatch.setattr(chat, "GmailMCPAgent", lambda **kwargs: agent)
    approval = Mock(return_value={"message_id": "sent1"})
    monkeypatch.setattr(chat, "approve_reply", approval)
    session = chat.create_session("protected")
    answer = chat.send_message(session, "Read and reply.")
    approval.assert_not_called()
    assert "No email has been sent" in answer
    assert session.pending_reply["recipient"] == "sender@example.test"


def test_baseline_auto_send_dispatches_once_after_proposal(config, monkeypatch, tmp_path):
    monkeypatch.setattr(chat, "ROOT", tmp_path)
    monkeypatch.setattr(chat.GmailMCPConfig, "from_env", lambda: config)
    monkeypatch.setattr(chat, "OllamaChatClient", Mock())
    pending = {"original_id": "id", "body": "Yes", "recipient": "sender@example.test",
               "subject": "Re: Meeting"}
    monkeypatch.setattr(chat, "ReplyProposalClient", lambda **kwargs: Mock(pending=pending))
    agent = Mock()
    agent.run = AsyncMock(return_value=Mock(final_response="Awaiting approval."))
    monkeypatch.setattr(chat, "GmailMCPAgent", lambda **kwargs: agent)
    dispatch = AsyncMock(return_value={"success": True, "status": "sent", "message_id": "sent1"})
    fake_client = Mock()
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=None)
    fake_client.call_tool = dispatch
    monkeypatch.setattr(chat, "GmailMCPClient", lambda **kwargs: fake_client)
    session = chat.create_session("baseline")
    assert "sent1" in chat.send_message(session, "Read and reply.")
    assert session.pending_reply is None
    assert any(event["event"] == "reply_auto_dispatch" for event in session.events)
    assert "reply_auto_dispatch" in next((tmp_path / "logs" / "guardium-chat-dispatch").glob("*.jsonl")).read_text()
    dispatch.assert_awaited_once()
    session.pending_reply = {**pending, "mode": "baseline"}
    with pytest.raises(ValueError, match="already reserved"):
        chat.approve_reply(session, automatic=True)
    dispatch.assert_awaited_once()
    assert "already reserved" in chat.send_message(session, "Read and reply.")
    assert session.pending_reply is None
    dispatch.assert_awaited_once()


def test_auto_send_requires_baseline_and_allowlisted_recipient(config, monkeypatch):
    pending = {"original_id": "id", "body": "Yes", "recipient": "other@example.test",
               "subject": "Re: Meeting", "mode": "baseline"}
    session = chat.create_session("baseline")
    session.pending_reply = pending
    monkeypatch.setattr(chat.GmailMCPConfig, "from_env", lambda: config)
    with pytest.raises(ValueError, match="no longer allowlisted"):
        chat.approve_reply(session, automatic=True)
    protected = chat.create_session("protected")
    protected.pending_reply = {**pending, "mode": "protected"}
    with pytest.raises(ValueError, match="not enabled"):
        chat.approve_reply(protected, automatic=True)


def test_automatic_send_message_uses_allowlisted_mcp_dispatch(config, monkeypatch, tmp_path):
    monkeypatch.setattr(chat, "ROOT", tmp_path)
    monkeypatch.setattr(chat.GmailMCPConfig, "from_env", lambda: config)
    fake_client = Mock()
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=None)
    fake_client.call_tool = AsyncMock(return_value={"success": True, "status": "sent", "message_id": "sent1"})
    monkeypatch.setattr(chat, "GmailMCPClient", lambda **kwargs: fake_client)
    session = chat.create_session("baseline")
    session.pending_reply = {"tool": "send_email", "recipient": "sender@example.test",
                             "subject": "Meeting", "body": "Yes", "mode": "baseline"}
    assert chat.approve_reply(session, automatic=True)["message_id"] == "sent1"
    fake_client.call_tool.assert_awaited_once_with("send_email", {
        "to": "sender@example.test", "subject": "Meeting", "body": "Yes",
    })


def test_protected_keeps_approval(config, monkeypatch, tmp_path):
    monkeypatch.setattr(chat, "ROOT", tmp_path)
    monkeypatch.setattr(chat.GmailMCPConfig, "from_env", lambda: config)
    monkeypatch.setattr(chat, "OllamaChatClient", Mock())
    monkeypatch.setattr(chat, "build_protected_guardrails", Mock())
    monkeypatch.setattr(chat, "ReplyProposalClient", lambda **kwargs: Mock(pending={
        "original_id": "id", "body": "Yes", "recipient": "sender@example.test", "subject": "Re: Meeting",
    }))
    agent = Mock()
    agent.run = AsyncMock(return_value=Mock(final_response="No message sent."))
    monkeypatch.setattr(chat, "GmailMCPAgent", lambda **kwargs: agent)
    dispatch = Mock()
    monkeypatch.setattr(chat, "approve_reply", dispatch)
    session = chat.create_session("protected")
    assert "No email has been sent" in chat.send_message(session, "Read and reply.")
    assert session.pending_reply is not None
    dispatch.assert_not_called()


def test_streamlit_rerun_does_not_dispatch(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    send = Mock(side_effect=AssertionError("Rerun must not start an agent request"))
    approve = Mock(side_effect=AssertionError("Rerun must not dispatch"))
    monkeypatch.setattr(chat, "send_message", send)
    monkeypatch.setattr(chat, "approve_reply", approve)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[3] / "guardium_chat.py"))
    app.run()
    app.run()
    assert not app.exception
    send.assert_not_called()
    approve.assert_not_called()


def test_streamlit_review_controls_follow_session_mode(monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    dispatch = Mock()
    monkeypatch.setattr(chat, "approve_reply", dispatch)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[3] / "guardium_chat.py")).run()
    protected = app.session_state["guardium_session"]
    protected.pending_reply = {"recipient": "sender@example.test", "subject": "Re: Meeting",
                               "body": "Yes", "mode": "protected"}
    app.run()
    assert not app.exception
    assert "Approve reply" in [button.label for button in app.button]
    assert app.chat_input[0].disabled

    app.toggle[0].set_value(False).run()
    baseline = app.session_state["guardium_session"]
    assert baseline.session_id != protected.session_id
    assert baseline.pending_reply is None
    # Restoring a stale OFF proposal must neither display approval nor send on rerun.
    baseline.pending_reply = {**protected.pending_reply, "mode": "baseline"}
    app.run()
    assert not app.exception
    assert {"Approve reply", "Approve send", "Cancel reply"}.isdisjoint(b.label for b in app.button)
    assert not app.chat_input[0].disabled
    dispatch.assert_not_called()


def test_legacy_reservation_blocks_both_modes(config, monkeypatch, tmp_path):
    import hashlib
    import json

    monkeypatch.setattr(chat, "ROOT", tmp_path)
    monkeypatch.setattr(chat.GmailMCPConfig, "from_env", lambda: config)
    pending = {"original_id": "id", "body": "Yes", "recipient": "sender@example.test",
               "subject": "Re: Meeting", "mode": "baseline"}
    folder = tmp_path / "logs" / "guardium-chat-dispatch"
    folder.mkdir(parents=True)
    old_pending = {**pending, "mode": "protected"}
    name = hashlib.sha256(json.dumps(old_pending, sort_keys=True).encode()).hexdigest()
    (folder / name).write_text("api_accepted\n")
    session = chat.create_session("baseline")
    session.pending_reply = pending
    with pytest.raises(ValueError, match="already reserved"):
        chat.approve_reply(session)


@pytest.mark.parametrize("allowed", [False, True])
def test_guard_decides_reply_before_proposal_boundary(allowed):
    model = Mock()
    proposal = {"tool_calls": [{"function": {"name": "reply_email",
                 "arguments": {"original_id": "id", "body": "I will attend."}}}]}
    model.chat.side_effect = [OllamaChatResult({"message": proposal}, 0),
                              OllamaChatResult({"message": {"content": "Done."}}, 0)]
    client = Mock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.discover_tools = AsyncMock(return_value=[DiscoveredMCPTool("reply_email", "Reply", {
        "type": "object", "properties": {},
    })])
    client.call_tool = AsyncMock(return_value={"success": False, "status": "awaiting_operator_approval"})
    guard = Mock()
    guard.begin_request.return_value = RequestAuthorization(True, "ok")
    guard.authorize_tool.return_value = ToolAuthorization(allowed, "test decision")
    guard.check_response.return_value = ResponseAuthorization(True, "Done.", None)
    agent = GmailMCPAgent(model_client=model, mcp_client=client, guardrails=guard)
    result = asyncio.run(agent.run("Read and reply that I will attend.", session_id="session"))
    assert client.call_tool.await_count == int(allowed)
    assert result.tool_records[0].executed is allowed
    assert guard.authorize_tool.call_args.kwargs["tool_name"] == "reply_email"
