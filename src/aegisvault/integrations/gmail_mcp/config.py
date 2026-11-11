"""Environment-backed configuration for the Gmail MCP demo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


LIVE_SEND_ACK = "I_EXPLICITLY_APPROVE_LIVE_SEND"


class GmailConfigError(ValueError):
    """Raised when Gmail MCP configuration is missing or unsafe."""


@dataclass(frozen=True, slots=True)
class GmailMCPConfig:
    credentials_path: Path
    token_path: Path
    recipient_allowlist: frozenset[str]
    dry_run: bool = True
    audit_log_path: Path = Path("logs/gmail_mcp.jsonl")
    live_send_approved: bool = False

    def __post_init__(self) -> None:
        if not self.dry_run and not self.live_send_approved:
            raise GmailConfigError("Live Gmail sending requires an explicit local acknowledgement")

    @classmethod
    def from_env(cls) -> "GmailMCPConfig":
        dry_run = _env_bool("AEGISVAULT_GMAIL_DRY_RUN", default=True)
        if not dry_run and os.getenv("AEGISVAULT_GMAIL_LIVE_SEND_ACK") != LIVE_SEND_ACK:
            raise GmailConfigError(
                "Live Gmail sending is disabled. Keep AEGISVAULT_GMAIL_DRY_RUN=true, or provide the explicit "
                "live-send acknowledgement only after user approval."
            )
        allowlist = frozenset(
            address.strip().lower()
            for address in os.getenv("AEGISVAULT_GMAIL_RECIPIENT_ALLOWLIST", "").split(",")
            if address.strip()
        )
        return cls(
            credentials_path=Path(os.getenv("AEGISVAULT_GMAIL_CREDENTIALS_PATH", "credentials.json")),
            token_path=Path(os.getenv("AEGISVAULT_GMAIL_TOKEN_PATH", "token.json")),
            recipient_allowlist=allowlist,
            dry_run=dry_run,
            audit_log_path=Path(os.getenv("AEGISVAULT_GMAIL_AUDIT_LOG", "logs/gmail_mcp.jsonl")),
            live_send_approved=not dry_run,
        )

    def server_environment(self) -> dict[str, str]:
        """Return the non-secret configuration required by a stdio server child."""

        values = {
            "AEGISVAULT_GMAIL_CREDENTIALS_PATH": str(self.credentials_path),
            "AEGISVAULT_GMAIL_TOKEN_PATH": str(self.token_path),
            "AEGISVAULT_GMAIL_RECIPIENT_ALLOWLIST": ",".join(sorted(self.recipient_allowlist)),
            "AEGISVAULT_GMAIL_DRY_RUN": "true" if self.dry_run else "false",
            "AEGISVAULT_GMAIL_AUDIT_LOG": str(self.audit_log_path),
        }
        if not self.dry_run:
            values["AEGISVAULT_GMAIL_LIVE_SEND_ACK"] = LIVE_SEND_ACK
        return values


def _env_bool(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise GmailConfigError(f"{name} must be a boolean value")
