"""Tracing and report models for Stage 4.0 runtime."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class TraceEvent:
    event_type: str
    timestamp: str
    order: int
    latency_ms: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(slots=True)
class AgentTrace:
    trace_id: str
    model: str
    user_prompt: str
    started_at: str
    completed_at: str | None = None
    events: list[TraceEvent] = field(default_factory=list)
    final_response: str | None = None
    total_latency_ms: float | None = None
    token_usage: dict[str, Any] | None = None

    @classmethod
    def start(cls, *, model: str, user_prompt: str) -> "AgentTrace":
        return cls(trace_id=str(uuid4()), model=model, user_prompt=user_prompt, started_at=utc_now())

    def add_event(
        self,
        event_type: str,
        *,
        payload: dict[str, Any] | None = None,
        latency_ms: float | None = None,
        error: str | None = None,
    ) -> None:
        self.events.append(
            TraceEvent(
                event_type=event_type,
                timestamp=utc_now(),
                order=len(self.events) + 1,
                latency_ms=latency_ms,
                payload=payload or {},
                error=error,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JsonlTraceLogger:
    """Append-only JSONL trace logger."""

    def __init__(self, path: str | Path = "logs/agent_runtime_traces.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, trace: AgentTrace) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
