"""Typed Sentinel runtime monitoring models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class SentinelDecisionLevel(str, Enum):
    """Sentinel signal levels."""

    ALLOW = "allow"
    OBSERVE = "observe"
    REVIEW = "review"
    BLOCK = "block"


@dataclass(frozen=True, slots=True)
class ToolCallState:
    """Structured proposed tool call consumed by Sentinel."""

    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", _freeze_mapping(self.arguments))


@dataclass(frozen=True, slots=True)
class SentinelExecutionState:
    """Structured execution object consumed by Sentinel."""

    session_id: str | None = None
    reasoning: str | None = None
    current_intent: str | None = None
    tool_call: ToolCallState | None = None
    step_index: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class MonitorResult:
    """One Sentinel monitor result."""

    similarity: float | None
    drift: float | None
    available: bool
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class SentinelConfig:
    """Configurable Sentinel weights, thresholds, and EMA behavior."""

    reasoning_weight: float = 0.20
    intent_weight: float = 0.35
    action_weight: float = 0.45
    ema_alpha: float = 0.40
    allow_threshold: float = 0.25
    observe_threshold: float = 0.45
    review_threshold: float = 0.65

    def __post_init__(self) -> None:
        for name in ("reasoning_weight", "intent_weight", "action_weight", "ema_alpha"):
            value = float(getattr(self, name))
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")
        if not (0.0 <= self.allow_threshold <= self.observe_threshold <= self.review_threshold <= 1.0):
            raise ValueError("Sentinel thresholds must be ordered within 0.0 and 1.0")


@dataclass(frozen=True, slots=True)
class SentinelDecision:
    """Final Sentinel runtime signal."""

    session_id: str
    reasoning_similarity: float | None
    intent_similarity: float | None
    action_similarity: float | None
    reasoning_drift: float | None
    intent_drift: float | None
    action_drift: float | None
    fused_risk: float
    ema_risk: float
    confidence: float
    decision: SentinelDecisionLevel
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


def thaw_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Convert frozen mappings back to mutable JSON-like dictionaries."""

    return {str(key): _thaw(raw_value) for key, raw_value in dict(value).items()}


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): _freeze(raw_value) for key, raw_value in dict(value).items()})


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return thaw_mapping(value)
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value
