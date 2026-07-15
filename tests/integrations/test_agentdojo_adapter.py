from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from aegisvault.audit import AuditSink
from aegisvault.integrations.agentdojo import AgentDojoAegisVaultAdapter, AgentDojoAdapterConfig, AgentDojoToolSpec
from aegisvault.policy.models import (
    ApplicationConfig,
    DomainPolicy,
    EvaluatorConfig,
    GateConfig,
    GatesConfig,
    Layer0Config,
    Layer0RequestConfig,
    Layer0ToolsConfig,
    LowConfidenceAction,
    SentinelPolicyConfig,
)
from aegisvault.runtime.action_gate import ActionGateConfig, ActionVerdict, SideEffectLevel
from aegisvault.runtime.goal_vault import GoalEmbedder, GoalVault, InMemoryGoalVaultBackend


class FakeEmbedder(GoalEmbedder):
    model_name = "fake-agentdojo"
    dimension = 2

    def embed(self, text: str) -> tuple[float, ...]:
        lower = text.lower()
        if "attacker" in lower or "delete" in lower:
            return (0.0, 1.0)
        return (1.0, 0.0)


class MemoryAudit(AuditSink):
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record(self, event: dict[str, Any]) -> None:
        self.events.append(event)


@dataclass(slots=True)
class FakeAgent:
    tool_name: str = "search_workspace"
    arguments: dict[str, Any] | None = None
    reasoning: str = "Search workspace notes"
    intent: str = "Search workspace notes"

    def run(self, task: dict[str, Any], tool_executor: Any) -> dict[str, Any]:
        result = tool_executor.execute_tool(
            self.tool_name,
            self.arguments or {"query": "notes"},
            reasoning=self.reasoning,
            intent=self.intent,
            step_index=1,
        )
        return {"executed": result.executed, "error": result.error}


def policy(*, request_session: bool = True, layer0_enabled: bool = True, sentinel_enabled: bool = True) -> DomainPolicy:
    gate = GateConfig(allow_threshold=0.8, block_threshold=0.8, low_confidence_action=LowConfidenceAction.BLOCK)
    return DomainPolicy(
        version="1.0",
        application=ApplicationConfig(name="agentdojo-workspace", description="Workspace policy"),
        purpose="Workspace benchmark tasks.",
        allowed_topics=["workspace"],
        gates=GatesConfig(request=gate, response=gate),
        evaluator=EvaluatorConfig(provider="ollama", model="llama3.2"),
        layer0=Layer0Config(
            enabled=layer0_enabled,
            request=Layer0RequestConfig(require_session_id=request_session, require_domain=True, allowed_domains=["workspace"]),
            tools=Layer0ToolsConfig(
                allowlist_mode=True,
                allowed=["search_workspace", "read_document"],
                denied=["delete_workspace"],
            ),
        ),
        sentinel=SentinelPolicyConfig(enabled=sentinel_enabled),
    )


def adapter(*, audit: MemoryAudit | None = None, policy_obj: DomainPolicy | None = None) -> AgentDojoAegisVaultAdapter:
    embedder = FakeEmbedder()
    return AgentDojoAegisVaultAdapter(
        policy=policy_obj or policy(),
        config=AgentDojoAdapterConfig(suite_name="workspace", domain="workspace"),
        tools=[
            AgentDojoToolSpec(
                name="search_workspace",
                description="Search workspace content",
                parameters={"type": "object", "required": ["query"], "properties": {"query": {"type": "string"}}},
                function=lambda query: {"items": [query]},
                side_effect_level=SideEffectLevel.READ,
            ),
            AgentDojoToolSpec(
                name="read_document",
                description="Read document",
                parameters={"type": "object", "required": ["doc_id"], "properties": {"doc_id": {"type": "string"}}},
                function=lambda doc_id: {"doc_id": doc_id},
                side_effect_level=SideEffectLevel.READ,
            ),
            AgentDojoToolSpec(
                name="delete_workspace",
                description="Delete workspace data",
                parameters={"type": "object", "required": ["target"], "properties": {"target": {"type": "string"}}},
                function=lambda target: {"deleted": target},
                side_effect_level=SideEffectLevel.WRITE,
            ),
        ],
        goal_vault=GoalVault(backend=InMemoryGoalVaultBackend(), embedder=embedder, audit_sink=audit),
        embedder=embedder,
        action_config=ActionGateConfig(high_similarity=0.95, low_similarity=0.2),
        audit_sink=audit,
    )


def test_goal_vault_initializes_and_execution_returns() -> None:
    guard = adapter()
    result = guard.run_task({"id": "task-1", "objective": "Search workspace notes", "metadata": {}}, FakeAgent())
    assert result.request_allowed is True
    assert result.goal_initialized is True
    assert result.agent_executed is True
    assert result.tool_results[0].executed is True
    assert guard.goal_vault.get_anchor("task-1").original_goal == "Search workspace notes"


def test_layer0_request_validation_runs_and_blocks_before_agent() -> None:
    guard = adapter()
    result = guard.run_task({"objective": "Search workspace notes", "metadata": {}}, FakeAgent())
    assert result.request_allowed is False
    assert result.goal_initialized is False
    assert result.agent_executed is False
    assert result.stopped_by == "LAYER0_REQUEST"
    assert result.request_layer0_decision is not None
    assert result.request_layer0_decision.rule_id == "L0_SESSION_MISSING"


def test_request_and_response_gates_are_not_used(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("semantic gate should not run in AgentDojo adapter")

    monkeypatch.setattr("aegisvault.gates.request.RequestGate.evaluate", explode)
    monkeypatch.setattr("aegisvault.gates.response.ResponseGate.evaluate", explode)
    result = adapter().run_task({"id": "task-2", "objective": "Search workspace notes", "metadata": {}}, FakeAgent())
    assert result.agent_executed is True
    assert result.tool_results[0].executed is True


def test_layer0_tool_validation_runs_and_blocks_tool_before_sentinel_and_action_gate() -> None:
    audit = MemoryAudit()
    result = adapter(audit=audit).run_task(
        {"id": "task-3", "objective": "Search workspace notes", "metadata": {}},
        FakeAgent(tool_name="unknown_tool", arguments={}),
    )
    assert result.tool_results[0].executed is False
    assert result.tool_results[0].layer0_decision is not None
    assert result.tool_results[0].layer0_decision.rule_id == "L0_TOOL_UNDECLARED"
    event_types = [event.get("event_type") for event in audit.events]
    assert "sentinel.evaluated" not in event_types
    assert "ACTION_GATE_DECISION" not in event_types


def test_layer0_block_for_registered_denied_tool_prevents_execution() -> None:
    result = adapter().run_task(
        {"id": "task-4", "objective": "Search workspace notes", "metadata": {}},
        FakeAgent(tool_name="delete_workspace", arguments={"target": "all"}),
    )
    assert result.tool_results[0].executed is False
    assert result.tool_results[0].action_decision is not None
    assert result.tool_results[0].action_decision.verdict == ActionVerdict.BLOCK
    assert "Layer 0 blocked tool call" in (result.tool_results[0].error or "")


def test_sentinel_and_action_gate_execute_for_allowed_tool() -> None:
    audit = MemoryAudit()
    result = adapter(audit=audit).run_task({"id": "task-5", "objective": "Search workspace notes", "metadata": {}}, FakeAgent())
    event_types = [event.get("event_type") for event in audit.events]
    assert "sentinel.evaluated" in event_types
    assert "ACTION_GATE_DECISION" in event_types
    assert result.tool_results[0].action_decision is not None
    assert result.tool_results[0].action_decision.verdict == ActionVerdict.EXECUTE


def test_blocked_tool_never_executes() -> None:
    called = {"count": 0}

    def delete_workspace(target: str) -> dict[str, str]:
        called["count"] += 1
        return {"deleted": target}

    guard = adapter()
    guard.tool_specs["delete_workspace"] = AgentDojoToolSpec(
        name="delete_workspace",
        description="Delete workspace",
        parameters={"type": "object", "required": ["target"], "properties": {"target": {"type": "string"}}},
        function=delete_workspace,
        side_effect_level=SideEffectLevel.WRITE,
    )
    result = guard.run_task(
        {"id": "task-6", "objective": "Search workspace notes", "metadata": {}},
        FakeAgent(tool_name="delete_workspace", arguments={"target": "all"}),
    )
    assert result.tool_results[0].executed is False
    assert called["count"] == 0

