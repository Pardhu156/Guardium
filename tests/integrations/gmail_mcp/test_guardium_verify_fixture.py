from __future__ import annotations

import base64
from email.header import Header
from unittest.mock import MagicMock

import pytest

from aegisvault.integrations.gmail_mcp import guardium_verify_fixture as fixture


RUN_ID = "7d74b74ce2a8"


def test_preview_prints_benign_meeting_without_authentication(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["fixture", "preview", RUN_ID])
    auth = MagicMock(side_effect=AssertionError("Preview must not authenticate"))
    monkeypatch.setattr(fixture, "GmailAuthenticator", auth)

    assert fixture.main() == 0

    output = capsys.readouterr().out
    assert output == (
        f"Subject: Guardium Demo — Project Review Meeting [GUARDIUM{RUN_ID}]\n\n"
        f"{fixture.MEETING_BODY}\n"
    )
    assert "Friday at 3:00 PM" in output
    auth.assert_not_called()


def test_rejects_run_ids_that_can_change_the_search_query() -> None:
    with pytest.raises(ValueError, match="12 lowercase hexadecimal characters"):
        fixture.fixture_subject("a from:someone")


def _message(subject: str, *, body: str = fixture.MEETING_BODY) -> dict:
    return {
        "id": "matched",
        "labelIds": ["INBOX"],
        "payload": {
            "mimeType": "text/plain",
            "headers": [{"name": "Subject", "value": subject}],
            "body": {"data": base64.urlsafe_b64encode(body.encode()).decode()},
        },
    }


@pytest.mark.parametrize("encode_subject", [False, True])
def test_verifier_reads_only_exact_subject_body_and_returns_id(encode_subject: bool) -> None:
    service = MagicMock()
    messages = service.users().messages()
    messages.list.return_value.execute.return_value = {
        "messages": [{"id": "similar"}, {"id": "matched"}],
    }
    subject = fixture.fixture_subject(RUN_ID)
    if encode_subject:
        subject = Header(subject, "utf-8").encode()
    messages.get.return_value.execute.side_effect = [_message("Re: " + subject), _message(subject), _message(subject)]

    assert fixture.verify_fixture(service, run_id=RUN_ID) == "matched"

    messages.list.assert_called_once_with(
        userId="me", labelIds=["INBOX"], q=f"subject:GUARDIUM{RUN_ID}",
        maxResults=10, fields="messages(id),nextPageToken",
    )
    assert [(call.kwargs["id"], call.kwargs["format"]) for call in messages.get.call_args_list] == [
        ("similar", "metadata"), ("matched", "metadata"), ("matched", "full"),
    ]
    messages.send.assert_not_called()


def test_verifier_handles_empty_search_without_fetching_bodies() -> None:
    service = MagicMock()
    messages = service.users().messages()
    messages.list.return_value.execute.return_value = {}

    with pytest.raises(fixture.FixtureVerificationError, match="no Inbox candidates") as error:
        fixture.verify_fixture(service, run_id=RUN_ID)

    assert fixture.fixture_subject(RUN_ID) in str(error.value)
    messages.get.assert_not_called()


def test_verifier_reports_subject_mismatch_without_fetching_body() -> None:
    service = MagicMock()
    messages = service.users().messages()
    messages.list.return_value.execute.return_value = {"messages": [{"id": "reply"}]}
    subject = fixture.fixture_subject(RUN_ID)
    messages.get.return_value.execute.return_value = _message("Re: " + subject)

    with pytest.raises(fixture.FixtureVerificationError, match="none had this exact Inbox subject") as error:
        fixture.verify_fixture(service, run_id=RUN_ID)

    assert subject in str(error.value)
    assert [call.kwargs["format"] for call in messages.get.call_args_list] == ["metadata"]


def test_verifier_rejects_ambiguous_results_without_fetching_bodies() -> None:
    service = MagicMock()
    messages = service.users().messages()
    messages.list.return_value.execute.return_value = {"messages": [{"id": "one"}, {"id": "two"}]}
    messages.get.return_value.execute.return_value = _message(fixture.fixture_subject(RUN_ID))

    with pytest.raises(fixture.FixtureVerificationError, match="Multiple messages"):
        fixture.verify_fixture(service, run_id=RUN_ID)

    assert [call.kwargs["format"] for call in messages.get.call_args_list] == ["metadata", "metadata"]


def test_verifier_rejects_an_unexpected_body() -> None:
    service = MagicMock()
    messages = service.users().messages()
    messages.list.return_value.execute.return_value = {"messages": [{"id": "matched"}]}
    subject = fixture.fixture_subject(RUN_ID)
    messages.get.return_value.execute.side_effect = [_message(subject), _message(subject, body="Different content")]

    with pytest.raises(fixture.FixtureVerificationError, match="body differs"):
        fixture.verify_fixture(service, run_id=RUN_ID)

    messages.send.assert_not_called()


def test_comparison_selection_accepts_untrusted_body_but_requires_unique_subject() -> None:
    service = MagicMock()
    messages = service.users().messages()
    subject = fixture.fixture_subject(RUN_ID)
    message = _message(subject, body="Untrusted test content")
    messages.list.return_value.execute.return_value = {"messages": [{"id": "matched"}]}
    messages.get.return_value.execute.return_value = message
    assert fixture.verify_fixture(service, run_id=RUN_ID, require_benign=False) == "matched"
    messages.list.return_value.execute.return_value = {"messages": [{"id": "one"}, {"id": "two"}]}
    with pytest.raises(fixture.FixtureVerificationError, match="Multiple messages"):
        fixture.verify_fixture(service, run_id=RUN_ID, require_benign=False)


def test_gmail_client_reads_exact_fixture_and_marks_it_untrusted() -> None:
    from aegisvault.integrations.gmail_mcp.gmail_api import GmailAPIClient

    service = MagicMock()
    messages = service.users().messages()
    message = _message(fixture.fixture_subject(RUN_ID))
    messages.list.return_value.execute.return_value = {"messages": [{"id": "matched"}]}
    messages.get.return_value.execute.return_value = message
    result = GmailAPIClient(service=service, recipient_allowlist=frozenset()).read_meeting_fixture(RUN_ID)
    assert result["message_id"] == "matched"
    assert result["body"] == fixture.MEETING_BODY
    assert result["content_trust"] == "untrusted_external_data"
    messages.send.assert_not_called()


@pytest.mark.parametrize("stored,accepted", [(fixture.FOOTER_BODY, True), (fixture.MEETING_BODY, False),
                                             (fixture.FOOTER_BODY + " extra", False)])
def test_footer_requires_exact_body_and_preserves_untrusted_content(stored, accepted):
    from aegisvault.integrations.gmail_mcp.gmail_api import GmailAPIClient, GmailAPIError

    service = MagicMock()
    messages = service.users().messages()
    messages.list.return_value.execute.return_value = {"messages": [{"id": "matched"}]}
    messages.get.return_value.execute.return_value = _message(fixture.fixture_subject(RUN_ID), body=stored)
    client = GmailAPIClient(service=service, recipient_allowlist=frozenset())
    if accepted:
        result = client.read_meeting_fixture(RUN_ID, footer=True)
        assert result["body"] == stored and result["content_trust"] == "untrusted_external_data"
    else:
        with pytest.raises(GmailAPIError):
            client.read_meeting_fixture(RUN_ID, footer=True)
    messages.send.assert_not_called()


def test_fixture_mode_routes_mcp_read_to_selected_fixture(tmp_path) -> None:
    import asyncio
    from aegisvault.integrations.gmail_mcp.client import GmailMCPClient
    from aegisvault.integrations.gmail_mcp.config import GmailMCPConfig
    from aegisvault.integrations.gmail_mcp.server import create_server

    config = GmailMCPConfig(credentials_path=tmp_path / "credentials.json", token_path=tmp_path / "token.json",
                             recipient_allowlist=frozenset(), audit_log_path=tmp_path / "audit.jsonl", fixture_run_id=RUN_ID)
    gmail = MagicMock()
    gmail.read_meeting_fixture.return_value = {"success": True, "message_id": "matched", "content_trust": "untrusted_external_data"}

    async def read():
        async with GmailMCPClient(target=create_server(config=config, gmail_client=gmail), config=config) as client:
            return await client.call_tool("read_latest_email", {})

    assert asyncio.run(read())["message_id"] == "matched"
    gmail.read_meeting_fixture.assert_called_once_with(RUN_ID)
    gmail.read_latest_email.assert_not_called()


def test_verify_does_not_search_a_different_account(monkeypatch, capsys) -> None:
    monkeypatch.setattr("sys.argv", ["fixture", "verify", RUN_ID, "--account", "test@example.com"])
    auth = MagicMock()
    authenticated = auth.return_value.authenticate.return_value
    authenticated.email_address = "different@example.com"
    monkeypatch.setattr(fixture, "GmailAuthenticator", auth)

    assert fixture.main() == 1
    assert "differs from the configured test account" in capsys.readouterr().err
    auth.return_value.authenticate.assert_called_once_with(interactive=False)
    authenticated.service.users.assert_not_called()
