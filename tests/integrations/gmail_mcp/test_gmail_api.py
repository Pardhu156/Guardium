from __future__ import annotations

import base64
from typing import Any

import pytest

from aegisvault.integrations.gmail_mcp.config import GmailConfigError, GmailMCPConfig
from aegisvault.integrations.gmail_mcp.gmail_api import (
    GmailAPIClient,
    GmailRecipientBlocked,
)


class Request:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    def execute(self) -> dict[str, Any]:
        return self.result


class Messages:
    def __init__(self, *, listing: dict[str, Any], message: dict[str, Any]) -> None:
        self.listing = listing
        self.message = message
        self.send_calls: list[dict[str, Any]] = []

    def list(self, **kwargs: Any) -> Request:
        assert kwargs == {"userId": "me", "labelIds": ["INBOX"], "maxResults": 1}
        return Request(self.listing)

    def get(self, **kwargs: Any) -> Request:
        assert kwargs == {"userId": "me", "id": "m1", "format": "full"}
        return Request(self.message)

    def send(self, **kwargs: Any) -> Request:
        self.send_calls.append(kwargs)
        return Request({"id": "sent-1"})


class Users:
    def __init__(self, messages: Messages) -> None:
        self._messages = messages

    def messages(self) -> Messages:
        return self._messages


class Service:
    def __init__(self, messages: Messages) -> None:
        self._users = Users(messages)

    def users(self) -> Users:
        return self._users


def encoded(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode()).decode().rstrip("=")


def test_read_latest_email_returns_headers_body_and_untrusted_marker() -> None:
    messages = Messages(
        listing={"messages": [{"id": "m1"}]},
        message={
            "id": "m1",
            "payload": {
                "mimeType": "text/plain",
                "headers": [
                    {"name": "Subject", "value": "Status"},
                    {"name": "From", "value": "Sender <sender@example.com>"},
                ],
                "body": {"data": encoded("Everything is ready.")},
            },
        },
    )

    result = GmailAPIClient(service=Service(messages), recipient_allowlist=frozenset()).read_latest_email()

    assert result == {
        "success": True,
        "empty": False,
        "message_id": "m1",
        "subject": "Status",
        "sender": "Sender <sender@example.com>",
        "body": "Everything is ready.",
        "content_trust": "untrusted_external_data",
    }


def test_read_latest_email_handles_empty_inbox() -> None:
    messages = Messages(listing={"messages": []}, message={})

    result = GmailAPIClient(service=Service(messages), recipient_allowlist=frozenset()).read_latest_email()

    assert result == {"success": True, "empty": True, "message": "Inbox is empty."}


def test_read_latest_email_prefers_plain_text_in_multipart_message() -> None:
    messages = Messages(
        listing={"messages": [{"id": "m1"}]},
        message={
            "id": "m1",
            "payload": {
                "mimeType": "multipart/alternative",
                "headers": [{"name": "Subject", "value": "Multipart"}],
                "parts": [
                    {"mimeType": "text/html", "body": {"data": encoded("<b>HTML</b>")}},
                    {"mimeType": "text/plain", "body": {"data": encoded("Plain text")}},
                ],
            },
        },
    )

    result = GmailAPIClient(service=Service(messages), recipient_allowlist=frozenset()).read_latest_email()

    assert result["body"] == "Plain text"


def test_send_email_dry_run_validates_but_never_calls_gmail() -> None:
    messages = Messages(listing={}, message={})
    client = GmailAPIClient(
        service=Service(messages),
        recipient_allowlist=frozenset({"allowed@example.com"}),
        dry_run=True,
    )

    result = client.send_email(to="Allowed <allowed@example.com>", subject="Test", body="Body")

    assert result["dry_run"] is True
    assert result["status"] == "validated_not_sent"
    assert messages.send_calls == []


def test_send_email_rejects_recipient_outside_server_allowlist() -> None:
    client = GmailAPIClient(
        service=None,
        recipient_allowlist=frozenset({"allowed@example.com"}),
        dry_run=True,
    )

    with pytest.raises(GmailRecipientBlocked, match="allowlist"):
        client.send_email(to="other@example.com", subject="Test", body="Body")


def test_configuration_rejects_live_send_without_explicit_acknowledgement(tmp_path: Any) -> None:
    with pytest.raises(GmailConfigError, match="acknowledgement"):
        GmailMCPConfig(
            credentials_path=tmp_path / "credentials.json",
            token_path=tmp_path / "token.json",
            recipient_allowlist=frozenset({"allowed@example.com"}),
            dry_run=False,
        )
