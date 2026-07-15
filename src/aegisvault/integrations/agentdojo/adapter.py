"""AgentDojo compatibility adapter for AegisVault runtime security.

The adapter intentionally bypasses AegisVault's semantic Request Gate and
Response Gate. It uses only deterministic Layer 0 sanity checks, Goal Vault,
Sentinel, Action Gate, and protected tool execution.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Mapping

from aegisvault.audit import AuditSink, NullAuditSink
from aegisvault.layer0 import Layer0Validator
from aegisvault.policy.models import DomainPolicy
from aegisvault.runtime.action_gate import (
    ActionGate,
    ActionGateConfig,
    ActionRuntimeContext,
    ActionVerdict,
    SideEffectLevel,
    ToolMetadata,
)
from aegisvault.runtime.action_gate.evaluators import ActionEvaluator
from aegisvault.runtime.goal_vault import GoalEmbedder, GoalVault, InMemoryGoalVaultBackend
from aegisvault.sentinel import SentinelMonitor

from aegisvault.integrations.agentdojo.models import (
    AgentDojoAdapterResult,
    AgentDojoTaskView,
    AgentDojoToolResult,
    AgentDojoToolSpec,
)


@dataclass(frozen=True, slots=True)
class AgentDojoAdapterConfig:
    """Runtime configuration for the AgentDojo adapter."""

    suite_name: str
    domain: str
    audit_only_layer0_tools: bool = False


class AgentDojoToolExecutor:
    """Tool execution hook passed to AgentDojo-compatible agents."""

    def __init__(self, *, adapter: "AgentDojoAegisVaultAdapter", session_id: str) -> None:
        self.adapter = adapter
        self.session_id = session_id
        self.records: list[AgentDojoToolResult] = []

    def execute_tool(
        self,
        name: str,
        arguments: Mapping[str, Any] | None = None,
        *,
        reasoning: str | None = None,
        intent: str | None = None,
        step_index: int | None = None,
    ) -> AgentDojoToolResult:
        """Execute one proposed tool call through Layer 0, Sentinel, and Action Gate."""

        record = self.adapter.execute_tool(
            session_id=self.session_id,
            name=name,
            arguments=dict(arguments or {}),
            reasoning=reasoning,
            intent=intent,
            step_index=step_index,
        )
        self.records.append(record)
        return record


class AgentDojoAegisVaultAdapter:
    """Route AgentDojo benchmark execution through AegisVault runtime security."""

    def __init__(
        self,
        *,
        policy: DomainPolicy,
        config: AgentDojoAdapterConfig,
        tools: list[AgentDojoToolSpec],
        goal_vault: GoalVault | None = None,
        embedder: GoalEmbedder | None = None,
        action_evaluator: ActionEvaluator | None = None,
        sentinel_monitor: SentinelMonitor | None = None,
        action_config: ActionGateConfig | None = None,
        audit_sink: AuditSink | None = None,
    ) -> None:
        self.policy = policy
        self.config = config
        self.audit_sink = audit_sink or NullAuditSink()
        self.tool_specs = {tool.name: tool for tool in tools}
        self.tool_catalog = {tool.name: {"description": tool.description, "parameters": dict(tool.parameters)} for tool in tools}
        self.goal_vault = goal_vault or GoalVault(backend=InMemoryGoalVaultBackend(), embedder=embedder, audit_sink=self.audit_sink)
        self.embedder = embedder or self.goal_vault.embedder
        self.layer0 = Layer0Validator(policy=policy, audit_sink=self.audit_sink, tool_catalog=self.tool_catalog)
        self.sentinel_monitor = sentinel_monitor
        self.action_gate = ActionGate(
            goal_vault=self.goal_vault,
            embedder=self.embedder,
            evaluator=action_evaluator,
            config=action_config,
            audit_sink=self.audit_sink,
        )

    def run_task(self, task: Any, agent: Any) -> AgentDojoAdapterResult:
        """Run an AgentDojo-like task through AegisVault and return control to the caller."""

        task_view = normalize_task(task, suite_name=self.config.suite_name)
        session_id = task_view.task_id
        request_decision = self.layer0.validate_request(
            session_id=session_id,
            request_text=task_view.objective,
            domain=self.config.domain,
            metadata=task_view.metadata,
        )
        if not request_decision.allowed:
            return AgentDojoAdapterResult(
                task=task_view,
                session_id=session_id,
                request_allowed=False,
                goal_initialized=False,
                agent_executed=False,
                request_layer0_decision=request_decision,
                stopped_by="LAYER0_REQUEST",
            )

        self.goal_vault.commit_goal(
            session_id=session_id,
            application_name=self.policy.application.name,
            goal=task_view.objective,
            metadata={"suite": task_view.suite_name},
        )
        executor = AgentDojoToolExecutor(adapter=self, session_id=session_id)
        try:
            agent_result = _call_agent(agent, task, executor)
        except Exception as exc:
            return AgentDojoAdapterResult(
                task=task_view,
                session_id=session_id,
                request_allowed=True,
                goal_initialized=True,
                agent_executed=True,
                tool_results=executor.records,
                stopped_by="AGENT_RUNTIME",
                error=f"{exc.__class__.__name__}: {exc}",
            )
        return AgentDojoAdapterResult(
            task=task_view,
            session_id=session_id,
            request_allowed=True,
            goal_initialized=True,
            agent_executed=True,
            agent_result=agent_result,
            tool_results=executor.records,
        )

    def execute_tool(
        self,
        *,
        session_id: str,
        name: str,
        arguments: dict[str, Any],
        reasoning: str | None,
        intent: str | None,
        step_index: int | None,
    ) -> AgentDojoToolResult:
        """Execute a proposed AgentDojo tool call through the protected tool path."""

        spec = self.tool_specs.get(name)
        if spec is None:
            return self._blocked_unknown_tool(session_id=session_id, name=name, arguments=arguments)
        protected = self.action_gate.protect_tool(
            spec.function,
            tool_metadata=ToolMetadata(
                risk_level=spec.risk_level,
                allowed_domains=spec.allowed_domains,
                required_permissions=spec.required_permissions,
                side_effect_level=spec.side_effect_level,
                requires_approval=spec.requires_approval,
            ),
            policy=self.policy,
            tool_name=spec.name,
            tool_description=spec.description,
            layer0_validator=self.layer0,
            tool_catalog=self.tool_catalog,
            sentinel_monitor=self.sentinel_monitor,
        )
        runtime_context = ActionRuntimeContext(
            qwen_reasoning=reasoning,
            current_intent=intent,
            step_index=step_index,
            session_metadata={"suite": self.config.suite_name},
        )
        result = protected(session_id=session_id, runtime_context=runtime_context, **arguments)
        return AgentDojoToolResult(
            tool_name=name,
            arguments=dict(arguments),
            executed=result.executed,
            result=result.result,
            error=None if result.executed else result.decision.reason,
            action_decision=result.decision,
        )

    def _blocked_unknown_tool(self, *, session_id: str, name: str, arguments: dict[str, Any]) -> AgentDojoToolResult:
        decision = self.layer0.validate_tool_call(
            session_id=session_id,
            tool_name=name,
            arguments=arguments,
            domain=self.config.domain,
            tool_catalog=self.tool_catalog,
        )
        return AgentDojoToolResult(
            tool_name=name,
            arguments=dict(arguments),
            executed=False,
            error=decision.reason,
            layer0_decision=decision,
        )


def normalize_task(task: Any, *, suite_name: str) -> AgentDojoTaskView:
    """Normalize mapping/object AgentDojo task shapes into a stable task view."""

    task_id = _first_value(task, ("id", "task_id", "uid", "name"))
    objective = _first_value(task, ("objective", "goal", "instruction", "prompt", "user_task"))
    metadata = _first_value(task, ("metadata",), default={})
    if not isinstance(metadata, Mapping):
        metadata = {}
    return AgentDojoTaskView(
        task_id=str(task_id or ""),
        suite_name=suite_name,
        objective=str(objective or ""),
        metadata=dict(metadata),
    )


def _call_agent(agent: Any, task: Any, executor: AgentDojoToolExecutor) -> Any:
    target = agent.run if hasattr(agent, "run") else agent
    if not callable(target):
        raise TypeError("AgentDojo agent must be callable or expose run()")
    signature = inspect.signature(target)
    if "tool_executor" in signature.parameters:
        return target(task, tool_executor=executor)
    if "executor" in signature.parameters:
        return target(task, executor=executor)
    return target(task, executor)


def _first_value(task: Any, names: tuple[str, ...], default: Any = None) -> Any:
    if isinstance(task, Mapping):
        for name in names:
            if name in task:
                return task[name]
        return default
    for name in names:
        if hasattr(task, name):
            return getattr(task, name)
    return default
