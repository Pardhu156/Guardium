"""Typed models for Stage 3.2 Action Gate."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from aegisvault.runtime.action_gate.exceptions import ActionGateValidationError


class ActionVerdict(str, Enum):
    """Supported Action Gate verdicts."""

    EXECUTE = "EXECUTE"
    JUSTIFY = "JUSTIFY"
    BLOCK = "BLOCK"


class ActionDecisionSource(str, Enum):
    """Source that produced the Action Gate decision."""

    COSINE = "COSINE"
    OLLAMA = "OLLAMA"
    FALLBACK = "FALLBACK"


class SideEffectLevel(str, Enum):
    """Coarse side-effect level for a proposed tool."""

    READ = "read"
    WRITE = "write"
    NETWORK = "network"
    SYSTEM = "system"


@dataclass(frozen=True, slots=True)
class ActionGateConfig:
    """Configurable thresholds for Action Gate decisions."""

    high_similarity: float = 0.82
    low_similarity: float = 0.35
    minimum_llm_confidence: float = 0.75
    fallback_verdict: ActionVerdict = ActionVerdict.BLOCK

    def __post_init__(self) -> None:
        for field_name in ("high_similarity", "low_similarity", "minimum_llm_confidence"):
            value = getattr(self, field_name)
            if value < -1.0 or value > 1.0:
                raise ActionGateValidationError(f"{field_name} must be between -1.0 and 1.0")
        if self.low_similarity >= self.high_similarity:
            raise ActionGateValidationError("low_similarity must be lower than high_similarity")
        if self.minimum_llm_confidence < 0.0 or self.minimum_llm_confidence > 1.0:
            raise ActionGateValidationError("minimum_llm_confidence must be between 0.0 and 1.0")


@dataclass(frozen=True, slots=True)
class ProposedToolAction:
    """A proposed tool call before execution."""

    tool_name: str
    tool_description: str
    tool_arguments: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tool_name.strip():
            raise ActionGateValidationError("tool_name must not be empty")
        if not self.tool_description.strip():
            raise ActionGateValidationError("tool_description must not be empty")
        object.__setattr__(self, "tool_arguments", _freeze_mapping(self.tool_arguments))


@dataclass(frozen=True, slots=True)
class ToolMetadata:
    """Metadata describing the risk and permission profile of a tool."""

    risk_level: str
    allowed_domains: tuple[str, ...] = ()
    required_permissions: tuple[str, ...] = ()
    side_effect_level: SideEffectLevel = SideEffectLevel.READ

    def __post_init__(self) -> None:
        if not self.risk_level.strip():
            raise ActionGateValidationError("risk_level must not be empty")
        object.__setattr__(self, "allowed_domains", tuple(str(item) for item in self.allowed_domains))
        object.__setattr__(self, "required_permissions", tuple(str(item) for item in self.required_permissions))
        if not isinstance(self.side_effect_level, SideEffectLevel):
            object.__setattr__(self, "side_effect_level", SideEffectLevel(str(self.side_effect_level)))


@dataclass(frozen=True, slots=True)
class ActionRuntimeContext:
    """Lightweight context available at the proposed action boundary."""

    reasoning_summary: str | None = None
    previous_approved_action: str | None = None
    session_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_metadata", _freeze_mapping(self.session_metadata))


@dataclass(frozen=True, slots=True)
class ActionGateDecision:
    """Structured Action Gate decision."""

    tool_name: str
    tool_arguments: Mapping[str, Any]
    goal_similarity: float | None
    decision_source: ActionDecisionSource
    verdict: ActionVerdict
    confidence: float | None
    reason: str
    latency_ms: float
    ollama_called: bool
    goal_session: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_arguments", _freeze_mapping(self.tool_arguments))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    """Result returned by a protected tool callable."""

    decision: ActionGateDecision
    executed: bool
    result: Any = None


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): _freeze_value(raw_value) for key, raw_value in dict(value).items()})


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_value(item) for item in value))
    return value


def thaw_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _thaw_value(raw_value) for key, raw_value in dict(value).items()}


def _thaw_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return thaw_mapping(value)
    if isinstance(value, tuple):
        return [_thaw_value(item) for item in value]
    return value
