"""Sanitized audit helpers for Gmail MCP tool activity."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any, Mapping
from uuid import uuid4

from aegisvault.audit import AuditSink


class GmailSanitizingAuditSink(AuditSink):
    """Redact Gmail addresses and message bodies from existing guard audit events."""

    def __init__(self, sink: AuditSink) -> None:
        self.sink = sink

    def record(self, event: dict[str, Any]) -> None:
        self.sink.record(_sanitize_nested(event))


def record_tool_event(
    sink: AuditSink,
    *,
    event_type: str,
    tool_name: str,
    arguments: Mapping[str, Any],
    status: str,
    result: Mapping[str, Any] | None = None,
    session_id: str | None = None,
) -> None:
    event = {
        "event_id": str(uuid4()),
        "timestamp": datetime.now(UTC).isoformat(),
        "event_type": event_type,
        "session_id": session_id,
        "tool": tool_name,
        "arguments": sanitize_arguments(arguments),
        "status": status,
        "result": sanitize_result(result),
    }
    try:
        sink.record(event)
    except Exception:
        return None


def sanitize_arguments(arguments: Mapping[str, Any]) -> dict[str, Any]:
    value = _sanitize_nested(dict(arguments))
    return value if isinstance(value, dict) else {}


def sanitize_result(result: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    allowed = ("success", "empty", "dry_run", "status", "message_id", "content_trust")
    return {key: result.get(key) for key in allowed if key in result}


def _address_fingerprint(value: str) -> str:
    digest = hashlib.sha256(value.strip().lower().encode("utf-8")).hexdigest()[:12]
    return f"sha256:{digest}"


def _sanitize_nested(value: Any, *, key: str | None = None) -> Any:
    lowered = (key or "").lower()
    if lowered == "body":
        return f"<redacted:{len(str(value))} chars>"
    if lowered in {"to", "recipient", "recipients"}:
        return _address_fingerprint(str(value))
    if lowered == "subject":
        return str(value)[:120]
    if isinstance(value, Mapping):
        return {str(child_key): _sanitize_nested(child, key=str(child_key)) for child_key, child in value.items()}
    if isinstance(value, list | tuple):
        return [_sanitize_nested(child) for child in value]
    return value
