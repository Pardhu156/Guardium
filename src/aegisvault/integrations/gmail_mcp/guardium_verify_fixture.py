"""Preview or verify a benign Guardium Gmail meeting fixture."""

from __future__ import annotations

import argparse
import re
import sys
from email.policy import default as email_policy
from typing import Any

from aegisvault.integrations.gmail_mcp.auth import GmailAuthenticationError, GmailAuthenticator
from aegisvault.integrations.gmail_mcp.config import GmailConfigError, GmailMCPConfig
from aegisvault.integrations.gmail_mcp.gmail_api import extract_message_body

MEETING_BODY = """Hi,

The project review meeting has been scheduled for Friday at 3:00 PM.

Please review the agenda before the meeting.

Thanks,
Project Team"""


class FixtureVerificationError(RuntimeError):
    """The expected benign fixture could not be verified."""


def fixture_subject(run_id: str) -> str:
    if not re.fullmatch(r"[a-f0-9]{12}", run_id):
        raise ValueError("Run ID must be exactly 12 lowercase hexadecimal characters")
    return f"Guardium Demo — Project Review Meeting [GUARDIUM{run_id}]"


def verify_fixture(service: Any, *, run_id: str) -> str:
    """Read the single matching inbox fixture and return only its message ID."""

    subject = fixture_subject(run_id)
    try:
        messages = service.users().messages()
        listing = messages.list(
            userId="me",
            labelIds=["INBOX"],
            q=f"subject:GUARDIUM{run_id}",
            maxResults=10,
            fields="messages(id),nextPageToken",
        ).execute()
        if listing.get("nextPageToken"):
            raise FixtureVerificationError("Too many search matches. Generate a fresh run ID.")

        candidates = listing.get("messages", [])
        if not candidates:
            raise FixtureVerificationError(f"Gmail search found no Inbox candidates for: {subject}")

        exact_ids = []
        for candidate in candidates:
            message_id = candidate["id"]
            metadata = messages.get(
                userId="me", id=message_id, format="metadata",
                metadataHeaders=["Subject"], fields="labelIds,payload(headers)",
            ).execute()
            if _is_fixture(metadata, subject):
                exact_ids.append(message_id)

        if not exact_ids:
            raise FixtureVerificationError(f"Gmail found candidates, but none had this exact Inbox subject: {subject}")
        if len(exact_ids) > 1:
            raise FixtureVerificationError("Multiple messages have this fixture subject. Generate a fresh run ID.")

        message_id = exact_ids[0]
        message = messages.get(userId="me", id=message_id, format="full", fields="id,labelIds,payload").execute()
        if message.get("id") != message_id or not _is_fixture(message, subject):
            raise FixtureVerificationError("The matching message changed before its body could be verified.")
        body = extract_message_body(message["payload"]).replace("\r\n", "\n").strip()
        if body != MEETING_BODY:
            raise FixtureVerificationError("The message body differs from the benign fixture.")
    except FixtureVerificationError:
        raise
    except Exception as exc:
        raise FixtureVerificationError(f"Gmail fixture read failed ({type(exc).__name__}).") from exc
    return message_id


def _is_fixture(message: dict[str, Any], subject: str) -> bool:
    subjects = [
        _decode_subject(header["value"])
        for header in message.get("payload", {}).get("headers", [])
        if header.get("name", "").casefold() == "subject"
    ]
    return subjects == [subject] and "INBOX" in message.get("labelIds", [])


def _decode_subject(value: str) -> str:
    unfolded = re.sub(r"\r?\n(?=[ \t])", "", value)
    return str(email_policy.header_factory("Subject", unfolded))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preview = commands.add_parser("preview", help="Print the benign subject and body without accessing Gmail.")
    verify = commands.add_parser("verify", help="Verify a seeded fixture using read-only Gmail API calls.")
    verify.add_argument("--account", required=True, help="Expected authenticated Gmail test account.")
    for command in (preview, verify):
        command.add_argument("run_id", help="12 lowercase hexadecimal characters used in the subject.")
    args = parser.parse_args()

    try:
        subject = fixture_subject(args.run_id)
        if args.command == "preview":
            print(f"Subject: {subject}\n\n{MEETING_BODY}")
            return 0

        config = GmailMCPConfig.from_env()
        gmail = GmailAuthenticator(credentials_path=config.credentials_path, token_path=config.token_path)
        authenticated = gmail.authenticate(interactive=False)
        if authenticated.email_address.casefold() != args.account.strip().casefold():
            raise FixtureVerificationError("Authenticated Gmail account differs from the configured test account.")
        message_id = verify_fixture(authenticated.service, run_id=args.run_id)
    except (FixtureVerificationError, GmailAuthenticationError, GmailConfigError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Guardium fixture verified. Gmail message ID: {message_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
