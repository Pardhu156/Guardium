from __future__ import annotations

import asyncio
import base64
import json
from argparse import Namespace
from email import policy
from email.parser import BytesParser
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from aegisvault.integrations.gmail_mcp import guardium_live_mail as live
from aegisvault.integrations.gmail_mcp.client import GmailMCPClientError
from aegisvault.integrations.gmail_mcp.config import GmailMCPConfig
from aegisvault.runtime.action_gate import ActionDecisionSource, ActionVerdict


SENDER = "sender@example.com"
RECIPIENT = "recipient@example.com"


def config_for(tmp_path: Path, *, live_send: bool = False) -> GmailMCPConfig:
    return GmailMCPConfig(
        credentials_path=tmp_path / "credentials.json",
        token_path=tmp_path / "token.json",
        recipient_allowlist=frozenset({RECIPIENT}),
        audit_log_path=tmp_path / "audit.jsonl",
        dry_run=not live_send,
        live_send_approved=live_send,
    )


@pytest.mark.parametrize("live_send", [False, True])
def test_exact_message_and_single_dispatch(tmp_path: Path, live_send: bool) -> None:
    service = Mock()
    send = service.users.return_value.messages.return_value.send
    send.return_value.execute.return_value = {"id": "mock-gmail-id"}
    report = {"dispatch_status": "not_dispatched", "gmail_api_message_id": None}
    report_path = tmp_path / "result.json"
    client = live.ConfirmedMessageClient(
        config=config_for(tmp_path, live_send=live_send),
        service=service, recipient=RECIPIENT, report=report, report_path=report_path,
    )
    approved = {"to": RECIPIENT, "subject": live.SUBJECT, "body": live.BODY}

    async def scenario() -> None:
        async with client:
            assert [tool.name for tool in await client.discover_tools()] == ["send_email"]
            with pytest.raises(GmailMCPClientError, match="differs"):
                await client.call_tool("read_latest_email")
            for change in ({"to": SENDER}, {"subject": "Other"}, {"body": "Other"}, {"bcc": SENDER}):
                with pytest.raises(GmailMCPClientError, match="differs"):
                    await client.call_tool("send_email", approved | change)
            assert report["dispatch_status"] == "not_dispatched"
            send.assert_not_called()
            await client.call_tool("send_email", approved)
            with pytest.raises(GmailMCPClientError, match="one permitted"):
                await client.call_tool("send_email", approved)

    asyncio.run(scenario())
    saved = json.loads(report_path.read_text())
    if live_send:
        send.assert_called_once()
        mime = BytesParser(policy=policy.default).parsebytes(base64.urlsafe_b64decode(send.call_args.kwargs["body"]["raw"]))
        assert str(mime["To"]) == RECIPIENT
        assert str(mime["Subject"]) == live.SUBJECT
        assert mime.get_content().strip() == live.BODY
        assert saved == {"dispatch_status": "gmail_api_accepted", "gmail_api_message_id": "mock-gmail-id"}
    else:
        send.assert_not_called()
        assert saved == {"dispatch_status": "validated_not_sent", "gmail_api_message_id": None}


def test_live_confirmation_must_precede_agent_creation(monkeypatch, tmp_path: Path) -> None:
    preflight = AsyncMock(return_value=(config_for(tmp_path), Mock(), Mock(), RECIPIENT))
    agent = Mock()
    monkeypatch.setattr(live, "_preflight", preflight)
    monkeypatch.setattr(live, "GmailMCPAgent", agent)
    monkeypatch.setattr(live, "RUNS", tmp_path / "runs")
    monkeypatch.setattr("builtins.input", lambda _: "no")

    with pytest.raises(live.LiveMailError, match="not confirmed"):
        asyncio.run(live._send(Namespace(live_send=True, sender=SENDER)))

    agent.assert_not_called()
    assert not live.RUNS.exists()


def test_ambiguous_api_result_is_saved_and_cannot_be_retried(tmp_path: Path) -> None:
    service = Mock()
    send = service.users.return_value.messages.return_value.send
    send.return_value.execute.side_effect = TimeoutError
    path = tmp_path / "result.json"
    client = live.ConfirmedMessageClient(
        config=config_for(tmp_path, live_send=True),
        service=service, recipient=RECIPIENT,
        report={"dispatch_status": "not_dispatched", "gmail_api_message_id": None},
        report_path=path,
    )
    message = {"to": RECIPIENT, "subject": live.SUBJECT, "body": live.BODY}

    async def scenario() -> None:
        async with client:
            result = await client.call_tool("send_email", message)
            assert result["success"] is False
            with pytest.raises(GmailMCPClientError, match="one permitted"):
                await client.call_tool("send_email", message)

    asyncio.run(scenario())
    send.assert_called_once()
    assert json.loads(path.read_text()) == {"dispatch_status": "outcome_unknown", "gmail_api_message_id": None}


def test_preflight_rejects_unlisted_recipient_before_oauth(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(live.GmailMCPConfig, "from_env", lambda: config_for(tmp_path))
    auth = Mock()
    monkeypatch.setattr(live, "GmailAuthenticator", auth)

    with pytest.raises(live.LiveMailError, match="allowlist"):
        asyncio.run(live._preflight(Namespace(sender=SENDER, recipient="outside@example.com")))

    auth.assert_not_called()


def test_blocked_send_has_no_gmail_receipt(monkeypatch, tmp_path: Path) -> None:
    service = Mock()
    actor = Mock(model="qwen3:4b-instruct")
    monkeypatch.setattr(live, "_preflight", AsyncMock(return_value=(config_for(tmp_path), actor, service, RECIPIENT)))
    monkeypatch.setattr(live, "RUNS", tmp_path / "runs")
    guard = Mock()
    monkeypatch.setattr(live, "build_protected_guardrails", Mock(return_value=guard))
    result = Mock(request_authorization=None, response_authorization=None)
    record = Mock(tool_name="send_email", executed=False)
    decision = record.authorization.decision
    decision.verdict = ActionVerdict.BLOCK
    decision.decision_source = ActionDecisionSource.FALLBACK
    decision.metadata = {"override": "dangerous_action"}
    result.tool_records = [record]
    agent = Mock()
    agent.return_value.run = AsyncMock(return_value=result)
    monkeypatch.setattr(live, "GmailMCPAgent", agent)

    assert asyncio.run(live._send(Namespace(live_send=False))) == 1

    saved = json.loads(next(live.RUNS.glob("*/result.json")).read_text())
    assert agent.call_args.kwargs["guardrails"] is guard
    assert saved["dispatch_status"] == "not_dispatched"
    assert saved["gmail_api_message_id"] is None
    assert saved["send_proposals"] == [{
        "action_gate": "BLOCK", "decision_source": "FALLBACK",
        "policy_rule": "dangerous_action", "mcp_returned_result": False,
    }]
    service.users.assert_not_called()


@pytest.mark.parametrize("api_accepted", [False, True])
def test_operator_can_confirm_only_an_accepted_live_run(monkeypatch, tmp_path: Path, api_accepted: bool) -> None:
    run_id = "a" * 32
    folder = tmp_path / run_id
    folder.mkdir()
    path = folder / "result.json"
    path.write_text(json.dumps({
        "dispatch_status": "gmail_api_accepted" if api_accepted else "validated_not_sent",
        "operator_confirmed_inbox_delivery": False,
    }))
    monkeypatch.setattr(live, "RUNS", tmp_path)
    monkeypatch.setattr("sys.argv", ["live", "report", run_id, "--confirm-inbox-delivery"])

    assert live.main() == (0 if api_accepted else 1)
    assert json.loads(path.read_text())["operator_confirmed_inbox_delivery"] is api_accepted
