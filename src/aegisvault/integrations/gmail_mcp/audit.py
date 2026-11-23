"""Sanitized audit helpers for Gmail MCP tool activity."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any, Mapping
from uuid import uuid4

from aegisvault.audit import AuditSink, JsonLineAuditSink
from aegisvault.types import GateDecision, Verdict


class ChatAuditSink(JsonLineAuditSink):
    """Persist chat decisions and execution facts without model-controlled text or arguments."""

    def record(self, event: dict[str, Any]) -> None:
        fields = {
            "event_type", "event", "timestamp", "event_id", "session_id", "step_index",
            "decision", "verdict", "sentinel_decision", "decision_source", "source",
            "category", "original_verdict", "confidence", "latency_ms", "fused_risk",
            "ema_risk", "action_drift", "intent_drift", "reasoning_drift",
            "action_signal_available", "intent_signal_available", "reasoning_signal_available",
            "backend_type", "ollama_called", "executed", "success", "status", "message_retrieved",
            "recipient_listed", "subject_chars", "body_chars",
            "error_code",
        }
        safe = {key: value for key, value in event.items() if key in fields}
        for key in ("tool", "tool_name"):
            if key in event:
                safe[key] = event[key] if event[key] in ("read_latest_email", "send_email", "reply_email", "read_confidential_profile") else "unknown"
        super().record(safe)

    def record_gate(self, decision: GateDecision) -> None:
        source = decision.evaluator.partition(":")[0]
        if source not in {"ollama", "disabled", "deterministic", "fallback"}:
            source = "other"
        event = {"event_type": "guardium_chat.gate_outcome", "event": f"{decision.gate.value}_gate",
                 "decision": decision.verdict.value, "source": source}
        if decision.confidence is not None:
            event["confidence"] = decision.confidence
        original = decision.metadata.get("original_verdict")
        if original in (Verdict.ALLOW.value, Verdict.BLOCK.value):
            event.update(category="low_confidence", original_verdict=original)
        else:
            event["category"] = "evaluator_failure" if source == "fallback" else "gate_verdict"
        self.record(event)


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
