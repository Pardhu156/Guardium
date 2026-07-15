"""Typed models for the AgentDojo adapter boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Mapping

from aegisvault.layer0 import Layer0Decision
from aegisvault.runtime.action_gate import ActionGateDecision, SideEffectLevel


@dataclass(frozen=True, slots=True)
class AgentDojoTaskView:
    """Normalized view of an AgentDojo benchmark task."""

    task_id: str
    suite_name: str
    objective: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AgentDojoToolSpec:
    """Normalized tool definition consumed by the adapter."""

    name: str
    description: str
    parameters: Mapping[str, Any]
    function: Callable[..., Any]
    risk_level: str = "medium"
    side_effect_level: SideEffectLevel = SideEffectLevel.READ
    allowed_domains: tuple[str, ...] = ()
    required_permissions: tuple[str, ...] = ()
    requires_approval: bool = False


@dataclass(slots=True)
class AgentDojoToolResult:
    """Tool result returned to the AgentDojo agent."""

    tool_name: str
    arguments: dict[str, Any]
    executed: bool
    result: Any = None
    error: str | None = None
    layer0_decision: Layer0Decision | None = None
    action_decision: ActionGateDecision | None = None


@dataclass(slots=True)
class AgentDojoAdapterResult:
    """Final adapter result returned to the AgentDojo benchmark runner."""

    task: AgentDojoTaskView
    session_id: str
    request_allowed: bool
    goal_initialized: bool
    agent_executed: bool
    agent_result: Any = None
    tool_results: list[AgentDojoToolResult] = field(default_factory=list)
    request_layer0_decision: Layer0Decision | None = None
    stopped_by: str | None = None
    error: str | None = None

