"""Small, testable Gmail API wrapper used by the MCP server."""

from __future__ import annotations

import base64
import html
import re
from email.message import EmailMessage
from email.utils import getaddresses
from typing import Any


class GmailAPIError(RuntimeError):
    """Raised for safe, user-facing Gmail API failures."""


class GmailRecipientBlocked(GmailAPIError):
    """Raised when the destination is not in the server allowlist."""


class GmailAPIClient:
    """Read and send Gmail messages with server-side safety constraints."""

    def __init__(self, *, service: Any | None, recipient_allowlist: frozenset[str], dry_run: bool = True) -> None:
        self.service = service
        self.recipient_allowlist = frozenset(address.lower() for address in recipient_allowlist)
        self.dry_run = dry_run

    def read_latest_email(self) -> dict[str, Any]:
        service = self._require_service()
        try:
            listing = service.users().messages().list(userId="me", labelIds=["INBOX"], maxResults=1).execute()
            messages = listing.get("messages") or []
            if not messages:
                return {"success": True, "empty": True, "message": "Inbox is empty."}
            message_id = str(messages[0]["id"])
            message = service.users().messages().get(userId="me", id=message_id, format="full").execute()
        except Exception as exc:
            raise GmailAPIError(f"Gmail read failed: {exc.__class__.__name__}") from exc

        payload = message.get("payload") or {}
        headers = {
            str(item.get("name", "")).lower(): str(item.get("value", ""))
            for item in payload.get("headers") or []
            if isinstance(item, dict)
        }
        return {
            "success": True,
            "empty": False,
            "message_id": str(message.get("id") or message_id),
            "subject": headers.get("subject", ""),
            "sender": headers.get("from", ""),
            "body": extract_message_body(payload),
            "content_trust": "untrusted_external_data",
        }

    def send_email(self, *, to: str, subject: str, body: str) -> dict[str, Any]:
        recipient = _single_recipient(to)
        if recipient not in self.recipient_allowlist:
            raise GmailRecipientBlocked(
                "Recipient is not present in the server allowlist (AEGISVAULT_GMAIL_RECIPIENT_ALLOWLIST)"
            )
        if "\r" in subject or "\n" in subject:
            raise GmailAPIError("Email subject must not contain newline characters")
        if self.dry_run:
            return {
                "success": True,
                "dry_run": True,
                "message_id": None,
                "recipient": recipient,
                "status": "validated_not_sent",
            }

        service = self._require_service()
        message = EmailMessage()
        message["To"] = recipient
        message["Subject"] = subject
        message.set_content(body)
        encoded = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")
        try:
            sent = service.users().messages().send(userId="me", body={"raw": encoded}).execute()
        except Exception as exc:
            raise GmailAPIError(f"Gmail send failed: {exc.__class__.__name__}") from exc
        return {
            "success": True,
            "dry_run": False,
            "message_id": str(sent.get("id") or ""),
            "recipient": recipient,
            "status": "sent",
        }

    def _require_service(self) -> Any:
        if self.service is None:
            raise GmailAPIError("Authenticated Gmail API service is required for this operation")
        return self.service


def extract_message_body(payload: dict[str, Any]) -> str:
    """Extract plain text from common Gmail single-part and multipart payloads."""

    plain_parts: list[str] = []
    html_parts: list[str] = []

    def visit(part: dict[str, Any]) -> None:
        mime_type = str(part.get("mimeType") or "").lower()
        data = (part.get("body") or {}).get("data")
        if isinstance(data, str) and data:
            decoded = _decode_body(data)
            if mime_type == "text/plain":
                plain_parts.append(decoded)
            elif mime_type == "text/html":
                html_parts.append(decoded)
        for child in part.get("parts") or []:
            if isinstance(child, dict):
                visit(child)

    visit(payload)
    if plain_parts:
        return "\n".join(part.strip() for part in plain_parts if part.strip())
    if html_parts:
        return "\n".join(_html_to_text(part) for part in html_parts if part.strip()).strip()
    return ""


def _decode_body(data: str) -> str:
    padding = "=" * (-len(data) % 4)
    try:
        return base64.urlsafe_b64decode(data + padding).decode("utf-8", errors="replace")
    except Exception as exc:
        raise GmailAPIError("Gmail message body was not valid base64url data") from exc


def _html_to_text(value: str) -> str:
    without_scripts = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", value)
    without_tags = re.sub(r"(?s)<[^>]+>", " ", without_scripts)
    return " ".join(html.unescape(without_tags).split())


def _single_recipient(value: str) -> str:
    if "\r" in value or "\n" in value:
        raise GmailAPIError("Recipient must not contain newline characters")
    addresses = [(name, address) for name, address in getaddresses([value]) if address]
    if len(addresses) != 1:
        raise GmailAPIError("Exactly one recipient address is required")
    recipient = addresses[0][1].strip().lower()
    if not recipient or "@" not in recipient:
        raise GmailAPIError("Recipient address is invalid")
    return recipient
