from __future__ import annotations

from pathlib import Path

import pytest

from aegisvault.integrations.gmail_mcp.auth import GmailAuthenticationError, GmailAuthenticator


class FakeCredentials:
    loaded: "FakeCredentials | None" = None

    def __init__(self, *, valid: bool, expired: bool, refresh_token: str | None = None) -> None:
        self.valid = valid
        self.expired = expired
        self.refresh_token = refresh_token
        self.refreshed = False

    @classmethod
    def from_authorized_user_file(cls, path: str, scopes: list[str]) -> "FakeCredentials":
        assert Path(path).exists()
        assert len(scopes) == 2
        assert cls.loaded is not None
        return cls.loaded

    def refresh(self, request: object) -> None:
        assert request == "request"
        self.valid = True
        self.expired = False
        self.refreshed = True

    def to_json(self) -> str:
        return '{"local_test_token":"redacted"}'


class FakeRequest:
    def execute(self) -> dict[str, str]:
        return {"emailAddress": "owner@example.com"}


class FakeUsers:
    def getProfile(self, *, userId: str) -> FakeRequest:
        assert userId == "me"
        return FakeRequest()


class FakeService:
    def users(self) -> FakeUsers:
        return FakeUsers()


def _build_service(name: str, version: str, **kwargs: object) -> FakeService:
    assert (name, version) == ("gmail", "v1")
    assert "credentials" in kwargs
    return FakeService()


def test_oauth_loads_valid_credentials_without_interactive_flow(tmp_path: Path) -> None:
    token = tmp_path / "token.json"
    token.write_text("{}", encoding="utf-8")
    FakeCredentials.loaded = FakeCredentials(valid=True, expired=False)
    auth = GmailAuthenticator(
        credentials_path=tmp_path / "credentials.json",
        token_path=token,
        credentials_cls=FakeCredentials,
        request_factory=lambda: "request",
        flow_cls=object,
        build_service=_build_service,
    )

    result = auth.authenticate(interactive=False)

    assert result.email_address == "owner@example.com"
    assert result.credentials is FakeCredentials.loaded


def test_oauth_refreshes_expired_credentials_and_persists_token(tmp_path: Path) -> None:
    token = tmp_path / "token.json"
    token.write_text("{}", encoding="utf-8")
    credentials = FakeCredentials(valid=False, expired=True, refresh_token="refresh-present")
    FakeCredentials.loaded = credentials
    auth = GmailAuthenticator(
        credentials_path=tmp_path / "credentials.json",
        token_path=token,
        credentials_cls=FakeCredentials,
        request_factory=lambda: "request",
        flow_cls=object,
        build_service=_build_service,
    )

    result = auth.authenticate(interactive=False)

    assert result.email_address == "owner@example.com"
    assert credentials.refreshed is True
    assert token.read_text(encoding="utf-8") == '{"local_test_token":"redacted"}'
    assert token.stat().st_mode & 0o777 == 0o600


def test_oauth_noninteractive_mode_reports_required_manual_action(tmp_path: Path) -> None:
    auth = GmailAuthenticator(
        credentials_path=tmp_path / "missing.json",
        token_path=tmp_path / "missing-token.json",
        credentials_cls=FakeCredentials,
        request_factory=lambda: "request",
        flow_cls=object,
        build_service=_build_service,
    )

    with pytest.raises(GmailAuthenticationError, match="Interactive Gmail authorization is required"):
        auth.authenticate(interactive=False)
