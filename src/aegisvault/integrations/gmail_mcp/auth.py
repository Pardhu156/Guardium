"""OAuth Desktop authentication for the Gmail MCP demo."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from aegisvault.integrations.gmail_mcp.config import GmailMCPConfig

GMAIL_SCOPES = (
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
)


class GmailAuthenticationError(RuntimeError):
    """Raised when Gmail credentials cannot be loaded or authorized."""


@dataclass(frozen=True, slots=True)
class AuthenticatedGmail:
    credentials: Any
    service: Any
    email_address: str


class GmailAuthenticator:
    """Load, refresh, or interactively create local Gmail OAuth credentials."""

    def __init__(
        self,
        *,
        credentials_path: Path,
        token_path: Path,
        credentials_cls: Any | None = None,
        request_factory: Callable[[], Any] | None = None,
        flow_cls: Any | None = None,
        build_service: Callable[..., Any] | None = None,
    ) -> None:
        if credentials_cls is None:
            from google.oauth2.credentials import Credentials

            credentials_cls = Credentials
        if request_factory is None:
            from google.auth.transport.requests import Request

            request_factory = Request
        if flow_cls is None:
            from google_auth_oauthlib.flow import InstalledAppFlow

            flow_cls = InstalledAppFlow
        if build_service is None:
            from googleapiclient.discovery import build

            build_service = build
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.credentials_cls = credentials_cls
        self.request_factory = request_factory
        self.flow_cls = flow_cls
        self.build_service = build_service

    def authenticate(self, *, interactive: bool = True) -> AuthenticatedGmail:
        credentials = self._load_credentials(interactive=interactive)
        try:
            service = self.build_service("gmail", "v1", credentials=credentials, cache_discovery=False)
            profile = service.users().getProfile(userId="me").execute()
            email_address = str(profile.get("emailAddress") or "").strip()
        except Exception as exc:
            raise GmailAuthenticationError(f"Gmail account verification failed: {exc.__class__.__name__}") from exc
        if not email_address:
            raise GmailAuthenticationError("Gmail account verification returned no email address")
        return AuthenticatedGmail(credentials=credentials, service=service, email_address=email_address)

    def _load_credentials(self, *, interactive: bool) -> Any:
        credentials = None
        if self.token_path.exists():
            try:
                credentials = self.credentials_cls.from_authorized_user_file(str(self.token_path), list(GMAIL_SCOPES))
            except Exception as exc:
                raise GmailAuthenticationError(
                    f"Could not load OAuth token file {self.token_path}; remove it and authenticate again."
                ) from exc

        if credentials is not None and credentials.valid:
            return credentials
        if credentials is not None and credentials.expired and credentials.refresh_token:
            try:
                credentials.refresh(self.request_factory())
            except Exception as exc:
                raise GmailAuthenticationError("Gmail OAuth token refresh failed") from exc
            self._persist_token(credentials)
            return credentials

        if not interactive:
            raise GmailAuthenticationError(
                "Interactive Gmail authorization is required. Run: "
                "python -m aegisvault.integrations.gmail_mcp.auth"
            )
        if not self.credentials_path.is_file():
            raise GmailAuthenticationError(
                f"OAuth Desktop credentials were not found at {self.credentials_path}. "
                "Set AEGISVAULT_GMAIL_CREDENTIALS_PATH to the downloaded credentials file."
            )
        try:
            flow = self.flow_cls.from_client_secrets_file(str(self.credentials_path), list(GMAIL_SCOPES))
            credentials = flow.run_local_server(port=0, open_browser=True)
        except Exception as exc:
            raise GmailAuthenticationError(f"Interactive Gmail authorization failed: {exc.__class__.__name__}") from exc
        self._persist_token(credentials)
        return credentials

    def _persist_token(self, credentials: Any) -> None:
        try:
            self.token_path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(self.token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                os.fchmod(handle.fileno(), 0o600)
                handle.write(credentials.to_json())
        except Exception as exc:
            raise GmailAuthenticationError(f"Could not persist OAuth token at {self.token_path}") from exc


def main() -> int:
    parser = argparse.ArgumentParser(description="Authorize the AegisVault Gmail MCP demo.")
    parser.parse_args()
    config = GmailMCPConfig.from_env()
    authenticated = GmailAuthenticator(
        credentials_path=config.credentials_path,
        token_path=config.token_path,
    ).authenticate(interactive=True)
    print(f"Authenticated Gmail account: {authenticated.email_address}")
    print(f"OAuth token stored locally at: {config.token_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
