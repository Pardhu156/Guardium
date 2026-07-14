from __future__ import annotations

from typing import Any

import pytest

from aegisvault.audit import AuditSink
from aegisvault.layer0 import Layer0Validator
from aegisvault.policy import load_policy
from aegisvault.policy.models import (
    ApplicationConfig,
    DomainPolicy,
    EvaluatorConfig,
    GateConfig,
    GatesConfig,
    Layer0Config,
    Layer0ToolsConfig,
    LowConfidenceAction,
    SentinelFailMode,
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
from aegisvault.sentinel import SentinelConfig, SentinelDecision, SentinelDecisionLevel, SentinelExecutionState
from aegisvault.sentinel.models import ToolCallState
from aegisvault.sentinel.service import SentinelMonitor


class FakeEmbedder(GoalEmbedder):
    model_name = "fake-shared"
    dimension = 2

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, text: str) -> tuple[float, ...]:
        self.calls += 1
        lower = text.lower()
        if lower == "use safe tool for email":
            return (1.0, 0.0)
        if "safe_tool" in lower:
            return (0.5, 0.866)
        return (0.0, 1.0) if "drift" in lower or "attacker" in lower else (1.0, 0.0)


class FakeActionEvaluator(ActionEvaluator):
    def __init__(self, verdict: ActionVerdict = ActionVerdict.EXECUTE) -> None:
        self.verdict = verdict
        self.calls = 0
        self.last_context: ActionRuntimeContext | None = None

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
        self.last_context = runtime_context
        return ActionGateDecision(
            tool_name=action.tool_name,
            tool_arguments=action.tool_arguments,
            goal_similarity=goal_similarity,
            decision_source=ActionDecisionSource.OLLAMA,
            verdict=self.verdict,
            confidence=1.0,
            reason="fake evaluator",
            latency_ms=0.0,
            ollama_called=True,
            goal_session=goal_anchor.session_id,
        )


class MemoryAudit(AuditSink):
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record(self, event: dict[str, Any]) -> None:
        self.events.append(event)


class CountingSentinel(SentinelMonitor):
    def __init__(self, decision: SentinelDecisionLevel, *, exc: Exception | None = None) -> None:
        self.calls = 0
        self.executions: list[SentinelExecutionState] = []
        self.exc = exc
        super().__init__(embedder=FakeEmbedder(), config=SentinelConfig())
        self._level = decision

    def analyze(self, *, session_id: str, trusted_goal: str, execution: SentinelExecutionState) -> SentinelDecision:
        self.calls += 1
        self.executions.append(execution)
        if self.exc is not None:
            raise self.exc
        risk = 0.7 if self._level == SentinelDecisionLevel.BLOCK else 0.2
        available = []
        if execution.reasoning:
            available.append("reasoning")
        if execution.current_intent:
            available.append("intent")
        if execution.tool_call:
            available.append("action")
        return SentinelDecision(
            session_id=session_id,
            reasoning_similarity=1.0 if execution.reasoning else None,
            intent_similarity=1.0 if execution.current_intent else None,
            action_similarity=1.0 if execution.tool_call else None,
            reasoning_drift=0.0 if execution.reasoning else None,
            intent_drift=0.0 if execution.current_intent else None,
            action_drift=0.0 if execution.tool_call else None,
            fused_risk=risk,
            ema_risk=risk,
            confidence=0.9,
            decision=self._level,
            reason=f"fake {self._level.value}",
            metadata={"available_monitors": available},
        )


def policy(*, layer0: bool = False, sentinel: bool = False, sentinel_fail_mode: SentinelFailMode = SentinelFailMode.CLOSED) -> DomainPolicy:
    gate = GateConfig(allow_threshold=0.8, block_threshold=0.8, low_confidence_action=LowConfidenceAction.BLOCK)
    return DomainPolicy(
        version="1.0",
        application=ApplicationConfig(name="email-agent", description="Email assistant"),
        purpose="Help with emails.",
        allowed_topics=["email"],
        gates=GatesConfig(request=gate, response=gate),
        evaluator=EvaluatorConfig(provider="ollama", model="llama3.2"),
        layer0=Layer0Config(
            enabled=layer0,
            tools=Layer0ToolsConfig(allowlist_mode=True, allowed=["safe_tool"], denied=["blocked_tool"]),
        ),
        sentinel=SentinelPolicyConfig(enabled=sentinel, fail_mode=sentinel_fail_mode),
    )


def make_gate(
    evaluator: FakeActionEvaluator,
    *,
    audit: MemoryAudit | None = None,
    embedder: FakeEmbedder | None = None,
) -> tuple[ActionGate, GoalVault, FakeEmbedder]:
    shared = embedder or FakeEmbedder()
    vault = GoalVault(backend=InMemoryGoalVaultBackend(), embedder=shared)
    vault.commit_goal(session_id="s1", application_name="email-agent", goal="Use safe tool for email")
    gate = ActionGate(
        goal_vault=vault,
        embedder=shared,
        evaluator=evaluator,
        audit_sink=audit,
        config=ActionGateConfig(high_similarity=1.0, low_similarity=-1.0),
    )
    return gate, vault, shared


def protected_call(
    *,
    gate: ActionGate,
    policy_obj: DomainPolicy,
    sentinel: CountingSentinel | None = None,
    evaluator_tool_name: str = "safe_tool",
    runtime_context: ActionRuntimeContext | None = None,
) -> tuple[bool, Any]:
    executed = {"called": False}

    def tool() -> str:
        executed["called"] = True
        return "done"

    layer0 = Layer0Validator(policy=policy_obj) if policy_obj.layer0.enabled else None
    protected = gate.protect_tool(
        tool,
        tool_metadata=ToolMetadata(risk_level="low", side_effect_level=SideEffectLevel.READ),
        policy=policy_obj,
        tool_name=evaluator_tool_name,
        layer0_validator=layer0,
        sentinel_monitor=sentinel,
    )
    result = protected(session_id="s1", runtime_context=runtime_context or ActionRuntimeContext())
    return executed["called"], result


def test_configuration_a_layer0_and_sentinel_disabled_preserves_stage_4_2_path() -> None:
    evaluator = FakeActionEvaluator()
    gate, _, embedder = make_gate(evaluator)
    sentinel = CountingSentinel(SentinelDecisionLevel.BLOCK)
    before = embedder.calls
    called, result = protected_call(gate=gate, policy_obj=policy(), sentinel=sentinel)
    assert result.executed is True
    assert called is True
    assert evaluator.calls == 1
    assert sentinel.calls == 0
    assert embedder.calls > before


def test_configuration_b_layer0_enabled_sentinel_disabled() -> None:
    evaluator = FakeActionEvaluator()
    gate, _, _ = make_gate(evaluator)
    sentinel = CountingSentinel(SentinelDecisionLevel.BLOCK)
    called, result = protected_call(gate=gate, policy_obj=policy(layer0=True), sentinel=sentinel, evaluator_tool_name="safe_tool")
    assert result.executed is True
    assert sentinel.calls == 0
    assert evaluator.calls == 1


def test_configuration_c_sentinel_enabled_layer0_absent() -> None:
    evaluator = FakeActionEvaluator()
    gate, _, _ = make_gate(evaluator)
    sentinel = CountingSentinel(SentinelDecisionLevel.ALLOW)
    _, result = protected_call(
        gate=gate,
        policy_obj=policy(sentinel=True),
        sentinel=sentinel,
        runtime_context=ActionRuntimeContext(qwen_reasoning="Read email", current_intent="Use safe tool", step_index=2),
    )
    assert result.executed is True
    assert sentinel.calls == 1
    assert evaluator.calls == 1
    assert evaluator.last_context is not None
    assert evaluator.last_context.sentinel_decision is not None


def test_configuration_d_layer0_then_sentinel_then_action_gate() -> None:
    audit = MemoryAudit()
    evaluator = FakeActionEvaluator()
    gate, _, _ = make_gate(evaluator, audit=audit)
    sentinel = CountingSentinel(SentinelDecisionLevel.ALLOW)
    _, result = protected_call(gate=gate, policy_obj=policy(layer0=True, sentinel=True), sentinel=sentinel)
    assert result.executed is True
    assert sentinel.calls == 1
    assert evaluator.calls == 1
    event_types = [event.get("event_type") for event in audit.events]
    assert "sentinel.evaluated" in event_types
    assert "ACTION_GATE_DECISION" in event_types


def test_sentinel_block_prevents_action_gate_and_tool_execution() -> None:
    evaluator = FakeActionEvaluator()
    gate, _, _ = make_gate(evaluator)
    sentinel = CountingSentinel(SentinelDecisionLevel.BLOCK)
    _, result = protected_call(gate=gate, policy_obj=policy(sentinel=True), sentinel=sentinel)
    assert result.executed is False
    assert result.decision.verdict == ActionVerdict.BLOCK
    assert evaluator.calls == 0


def test_action_gate_block_prevents_execution_even_when_sentinel_allows() -> None:
    evaluator = FakeActionEvaluator(ActionVerdict.BLOCK)
    gate, _, _ = make_gate(evaluator)
    sentinel = CountingSentinel(SentinelDecisionLevel.ALLOW)
    _, result = protected_call(gate=gate, policy_obj=policy(sentinel=True), sentinel=sentinel)
    assert result.executed is False
    assert result.decision.verdict == ActionVerdict.BLOCK
    assert sentinel.calls == 1


def test_missing_reasoning_does_not_fail_and_missing_signal_is_audited() -> None:
    audit = MemoryAudit()
    evaluator = FakeActionEvaluator()
    gate, _, _ = make_gate(evaluator, audit=audit)
    sentinel = CountingSentinel(SentinelDecisionLevel.ALLOW)
    _, result = protected_call(
        gate=gate,
        policy_obj=policy(sentinel=True),
        sentinel=sentinel,
        runtime_context=ActionRuntimeContext(current_intent="Use safe tool", step_index=1),
    )
    assert result.executed is True
    assert sentinel.executions[0].reasoning is None
    assert any(event.get("event_type") == "sentinel.signal_missing" for event in audit.events)


def test_missing_trusted_goal_fail_closed_blocks_before_tool_execution() -> None:
    evaluator = FakeActionEvaluator()
    gate = ActionGate(
        goal_vault=GoalVault(backend=InMemoryGoalVaultBackend(), embedder=FakeEmbedder()),
        embedder=FakeEmbedder(),
        evaluator=evaluator,
    )
    sentinel = CountingSentinel(SentinelDecisionLevel.ALLOW)
    _, result = protected_call(gate=gate, policy_obj=policy(sentinel=True), sentinel=sentinel)
    assert result.executed is False
    assert result.decision.verdict == ActionVerdict.BLOCK
    assert evaluator.calls == 0


def test_internal_sentinel_error_fail_open_continues_to_action_gate_but_explicit_block_does_not() -> None:
    evaluator = FakeActionEvaluator()
    gate, _, _ = make_gate(evaluator)
    failing = CountingSentinel(SentinelDecisionLevel.ALLOW, exc=RuntimeError("boom"))
    _, result = protected_call(
        gate=gate,
        policy_obj=policy(sentinel=True, sentinel_fail_mode=SentinelFailMode.OPEN),
        sentinel=failing,
    )
    assert result.executed is True
    assert evaluator.calls == 1

    evaluator2 = FakeActionEvaluator()
    gate2, _, _ = make_gate(evaluator2)
    blocking = CountingSentinel(SentinelDecisionLevel.BLOCK)
    _, blocked = protected_call(
        gate=gate2,
        policy_obj=policy(sentinel=True, sentinel_fail_mode=SentinelFailMode.OPEN),
        sentinel=blocking,
    )
    assert blocked.executed is False
    assert evaluator2.calls == 0


def test_ema_state_persists_and_isolated_with_shared_service() -> None:
    service = SentinelMonitor(embedder=FakeEmbedder())
    aligned = SentinelExecutionState(current_intent="Use safe tool", tool_call=ToolCallState(name="safe_tool", arguments={}))
    drift = SentinelExecutionState(current_intent="weather drift", tool_call=ToolCallState(name="send", arguments={"to": "attacker"}))
    first = service.analyze(session_id="s1", trusted_goal="Use safe tool", execution=aligned)
    second = service.analyze(session_id="s1", trusted_goal="Use safe tool", execution=drift)
    other = service.analyze(session_id="s2", trusted_goal="Use safe tool", execution=aligned)
    assert second.ema_risk > first.ema_risk
    assert other.ema_risk == pytest.approx(first.fused_risk)


def test_audit_logs_do_not_contain_raw_reasoning_or_secret() -> None:
    audit = MemoryAudit()
    evaluator = FakeActionEvaluator()
    gate, _, _ = make_gate(evaluator, audit=audit)
    sentinel = CountingSentinel(SentinelDecisionLevel.ALLOW)
    protected_call(
        gate=gate,
        policy_obj=policy(sentinel=True),
        sentinel=sentinel,
        runtime_context=ActionRuntimeContext(
            qwen_reasoning="raw private reasoning with secret-token",
            current_intent="Use safe tool",
        ),
    )
    serialized = str(audit.events)
    assert "raw private reasoning" not in serialized
    assert "secret-token" not in serialized


def test_stage5_policy_loads_and_enables_both_layers() -> None:
    loaded = load_policy("evaluation/policies/email_assistant_stage5.yaml")
    assert loaded.layer0.enabled is True
    assert loaded.sentinel.enabled is True
