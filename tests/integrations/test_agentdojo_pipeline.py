from __future__ import annotations

from typing import Any

from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime, make_function
from agentdojo.types import ChatAssistantMessage

from aegisvault.integrations.agentdojo import AgentDojoAdapterConfig
from aegisvault.integrations.agentdojo.pipeline import AegisVaultAgentDojoToolsExecutor
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
from aegisvault.runtime.action_gate import (
    ActionDecisionSource,
    ActionGate,
    ActionGateConfig,
    ActionGateDecision,
    ActionRuntimeContext,
    ActionVerdict,
    ProposedToolAction,
    SideEffectLevel,
    ToolMetadata,
)
from aegisvault.runtime.action_gate.evaluators import ActionEvaluator
from aegisvault.runtime.goal_vault import GoalAnchor, GoalEmbedder, GoalVault, InMemoryGoalVaultBackend
from aegisvault.sentinel import SentinelConfig, SentinelDecision, SentinelDecisionLevel, SentinelExecutionState, SentinelMonitor


class FakeEmbedder(GoalEmbedder):
    model_name = "fake-agentdojo-pipeline"
    dimension = 2

    def embed(self, text: str) -> tuple[float, ...]:
        if "attacker" in text.lower() or "delete" in text.lower():
            return (0.0, 1.0)
        return (1.0, 0.0)


class MidSimilarityEmbedder(GoalEmbedder):
    model_name = "mid-similarity-test"
    dimension = 2

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, text: str) -> tuple[float, ...]:
        self.calls += 1
        if self.calls == 1:
            return (1.0, 0.0)
        return (0.5, 0.8660254038)


class CountingSentinel(SentinelMonitor):
    def __init__(self, level: SentinelDecisionLevel = SentinelDecisionLevel.ALLOW) -> None:
        self.calls = 0
        self.last_execution: SentinelExecutionState | None = None
        self.level = level
        super().__init__(embedder=FakeEmbedder(), config=SentinelConfig())

    def analyze(self, *, session_id: str, trusted_goal: str, execution: SentinelExecutionState) -> SentinelDecision:
        self.calls += 1
        self.last_execution = execution
        risk = 0.7 if self.level == SentinelDecisionLevel.BLOCK else 0.1
        return SentinelDecision(
            session_id=session_id,
            reasoning_similarity=None,
            intent_similarity=1.0,
            action_similarity=1.0,
            reasoning_drift=None,
            intent_drift=0.0,
            action_drift=0.0,
            fused_risk=risk,
            ema_risk=risk,
            confidence=0.9,
            decision=self.level,
            reason=f"fake {self.level.value}",
            metadata={"available_monitors": ["intent", "action"]},
        )


def search_workspace(query: str) -> dict[str, Any]:
    """Search workspace content.

    :param query: Search query.
    """

    return {"query": query}


def delete_workspace(target: str) -> dict[str, Any]:
    """Delete workspace content.

    :param target: Target to delete.
    """

    raise AssertionError("blocked tool should not execute")


def policy() -> DomainPolicy:
    gate = GateConfig(allow_threshold=0.8, block_threshold=0.8, low_confidence_action=LowConfidenceAction.BLOCK)
    return DomainPolicy(
        version="1.0",
        application=ApplicationConfig(name="agentdojo-workspace", description="Workspace policy"),
        purpose="Workspace tasks.",
        allowed_topics=["workspace"],
        gates=GatesConfig(request=gate, response=gate),
        evaluator=EvaluatorConfig(provider="ollama", model="llama3.2"),
        layer0=Layer0Config(
            enabled=True,
            request=Layer0RequestConfig(require_session_id=True, require_domain=True, allowed_domains=["workspace"]),
            tools=Layer0ToolsConfig(allowlist_mode=True, allowed=["search_workspace"], denied=["delete_workspace"]),
        ),
        sentinel=SentinelPolicyConfig(enabled=True),
    )


def executor(sentinel: CountingSentinel | None = None) -> AegisVaultAgentDojoToolsExecutor:
    embedder = FakeEmbedder()
    return AegisVaultAgentDojoToolsExecutor(
        policy=policy(),
        config=AgentDojoAdapterConfig(suite_name="workspace", domain="workspace"),
        goal_vault=GoalVault(backend=InMemoryGoalVaultBackend(), embedder=embedder),
        embedder=embedder,
        sentinel_monitor=sentinel or CountingSentinel(),
        action_config=ActionGateConfig(high_similarity=0.95, low_similarity=0.2),
    )


def assistant_message(tool_name: str, args: dict[str, Any]) -> ChatAssistantMessage:
    return ChatAssistantMessage(
        role="assistant",
        content=None,
        tool_calls=[FunctionCall(function=tool_name, args=args, id="call-1")],
    )


def test_real_agentdojo_tool_executor_runs_allowed_tool() -> None:
    sentinel = CountingSentinel()
    runtime = FunctionsRuntime([make_function(search_workspace)])
    _, _, _, messages, _ = executor(sentinel).query(
        "Search workspace notes",
        runtime,
        messages=[assistant_message("search_workspace", {"query": "notes"})],
        extra_args={"task_id": "workspace-user-task-1"},
    )
    tool_message = messages[-1]
    assert tool_message["role"] == "tool"
    assert tool_message["error"] is None
    assert "notes" in tool_message["content"][0]["content"]
    assert sentinel.calls == 1
    assert sentinel.last_execution is not None
    assert sentinel.last_execution.tool_call is not None
    assert sentinel.last_execution.tool_call.name == "search_workspace"


def test_real_agentdojo_sentinel_receives_observable_message_context() -> None:
    sentinel = CountingSentinel()
    runtime = FunctionsRuntime([make_function(search_workspace)])
    messages = [
        {"role": "assistant", "content": "I will search the workspace notes before answering.", "tool_calls": None},
        assistant_message("search_workspace", {"query": "notes"}),
    ]
    _, _, _, result_messages, _ = executor(sentinel).query(
        "Search workspace notes",
        runtime,
        messages=messages,
        extra_args={"task_id": "workspace-user-task-context"},
    )
    assert result_messages[-1]["error"] is None
    assert sentinel.last_execution is not None
    assert sentinel.last_execution.reasoning is not None
    assert "search the workspace notes" in sentinel.last_execution.reasoning
    assert sentinel.last_execution.metadata["trusted_goal"] == "Search workspace notes"


def test_real_agentdojo_layer0_tool_block_prevents_runtime_execution() -> None:
    runtime = FunctionsRuntime([make_function(delete_workspace)])
    _, _, _, messages, _ = executor().query(
        "Search workspace notes",
        runtime,
        messages=[assistant_message("delete_workspace", {"target": "all"})],
        extra_args={"task_id": "workspace-user-task-2"},
    )
    tool_message = messages[-1]
    assert tool_message["role"] == "tool"
    assert tool_message["error"] is not None
    assert "Tool is denied" in tool_message["error"]


def test_real_agentdojo_pipeline_derives_session_when_runner_does_not_pass_task_id() -> None:
    runtime = FunctionsRuntime([make_function(search_workspace)])
    protected = executor()
    _, _, _, messages, _ = protected.query(
        "Search workspace notes",
        runtime,
        messages=[assistant_message("search_workspace", {"query": "notes"})],
        extra_args={},
    )
    assert messages[-1]["error"] is None
    assert protected.goal_vault.get_anchor("agentdojo-" + str(abs(hash("Search workspace notes")))).original_goal == "Search workspace notes"


class RecordingEvaluator(ActionEvaluator):
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(
        self,
        *,
        goal_anchor: GoalAnchor,
        action: ProposedToolAction,
        tool_metadata: ToolMetadata,
        policy: DomainPolicy,
        runtime_context: ActionRuntimeContext | None = None,
        goal_similarity: float | None = None,
    ) -> ActionGateDecision:
        self.calls += 1
        return ActionGateDecision(
            tool_name=action.tool_name,
            tool_arguments=action.tool_arguments,
            goal_similarity=goal_similarity,
            decision_source=ActionDecisionSource.OLLAMA,
            verdict=ActionVerdict.EXECUTE,
            confidence=0.95,
            reason="verified",
            latency_ms=0.0,
            ollama_called=True,
            goal_session=goal_anchor.session_id,
        )


def test_risky_high_similarity_action_does_not_bypass_action_gate_verifier() -> None:
    embedder = FakeEmbedder()
    vault = GoalVault(backend=InMemoryGoalVaultBackend(), embedder=embedder)
    vault.commit_goal(session_id="s1", application_name="test", goal="Search workspace notes")
    evaluator = RecordingEvaluator()
    gate = ActionGate(
        goal_vault=vault,
        embedder=embedder,
        evaluator=evaluator,
        config=ActionGateConfig(high_similarity=0.8, low_similarity=0.1, force_verifier_for_risky_actions=True),
    )
    decision = gate.evaluate_action(
        session_id="s1",
        action=ProposedToolAction("send_slack_message", "Send message", {"text": "hello"}),
        tool_metadata=ToolMetadata(
            risk_level="high",
            side_effect_level=SideEffectLevel.WRITE,
            requires_approval=True,
        ),
        policy=policy(),
    )
    assert evaluator.calls == 1
    assert decision.decision_source == ActionDecisionSource.OLLAMA
    assert decision.ollama_called is True


def test_read_only_low_risk_action_keeps_cosine_fast_path() -> None:
    embedder = FakeEmbedder()
    vault = GoalVault(backend=InMemoryGoalVaultBackend(), embedder=embedder)
    vault.commit_goal(session_id="s1", application_name="test", goal="Search workspace notes")
    evaluator = RecordingEvaluator()
    gate = ActionGate(
        goal_vault=vault,
        embedder=embedder,
        evaluator=evaluator,
        config=ActionGateConfig(high_similarity=0.8, low_similarity=0.1, force_verifier_for_risky_actions=True),
    )
    decision = gate.evaluate_action(
        session_id="s1",
        action=ProposedToolAction("search_workspace", "Search workspace", {"query": "notes"}),
        tool_metadata=ToolMetadata(risk_level="low", side_effect_level=SideEffectLevel.READ),
        policy=policy(),
    )
    assert evaluator.calls == 0
    assert decision.decision_source == ActionDecisionSource.COSINE


def test_low_risk_read_action_can_use_mid_similarity_fast_path() -> None:
    embedder = MidSimilarityEmbedder()
    vault = GoalVault(backend=InMemoryGoalVaultBackend(), embedder=embedder)
    vault.commit_goal(session_id="s1", application_name="test", goal="Find the calendar event")
    evaluator = RecordingEvaluator()
    gate = ActionGate(
        goal_vault=vault,
        embedder=embedder,
        evaluator=evaluator,
        config=ActionGateConfig(
            high_similarity=0.95,
            low_similarity=0.2,
            force_verifier_for_risky_actions=True,
            allow_low_risk_read_fast_path=True,
        ),
    )
    decision = gate.evaluate_action(
        session_id="s1",
        action=ProposedToolAction("search_calendar_events", "Search calendar", {"query": "Networking event"}),
        tool_metadata=ToolMetadata(risk_level="low", side_effect_level=SideEffectLevel.READ),
        policy=policy(),
    )
    assert evaluator.calls == 0
    assert decision.verdict == ActionVerdict.EXECUTE
    assert decision.decision_source == ActionDecisionSource.COSINE
    assert decision.metadata["threshold_type"] == "low_risk_read_fast_path"
