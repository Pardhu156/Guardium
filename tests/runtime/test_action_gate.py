from __future__ import annotations

from typing import Any

import pytest

from aegisvault.audit import AuditSink
from aegisvault.policy import load_policy
from aegisvault.policy.models import DomainPolicy
from aegisvault.runtime.action_gate import (
    ActionDecisionSource,
    ActionEvaluator,
    ActionGate,
    ActionGateConfig,
    ActionGateDecision,
    ActionRuntimeContext,
    ActionVerdict,
    ProposedToolAction,
    SideEffectLevel,
    ToolExecutionResult,
    ToolMetadata,
    build_action_embedding_text,
    cosine_similarity,
)
from aegisvault.runtime.action_gate.evaluators import OllamaActionEvaluator
from aegisvault.runtime.action_gate.exceptions import (
    ActionEvaluatorError,
    ActionGateValidationError,
    MalformedActionEvaluatorResponseError,
)
from aegisvault.runtime.goal_vault import GoalAnchor, GoalEmbedder, GoalVault, InMemoryGoalVaultBackend
from aegisvault.runtime.goal_vault import GoalBackendUnavailableError


class FakeEmbedder(GoalEmbedder):
    model_name = "fake-action-3d"
    dimension = 3

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        if "terminal.execute_python" in text:
            return [-1.0, 0.0, 0.0]
        if "gmail.send" in text:
            return [0.6, 0.8, 0.0]
        return [1.0, 0.0, 0.0]


class FakeActionEvaluator(ActionEvaluator):
    def __init__(
        self,
        verdict: ActionVerdict = ActionVerdict.JUSTIFY,
        confidence: float = 0.9,
        exc: Exception | None = None,
    ) -> None:
        self.verdict = verdict
        self.confidence = confidence
        self.exc = exc
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
        if self.exc is not None:
            raise self.exc
        return ActionGateDecision(
            tool_name=action.tool_name,
            tool_arguments=action.tool_arguments,
            goal_similarity=goal_similarity,
            decision_source=ActionDecisionSource.OLLAMA,
            verdict=self.verdict,
            confidence=self.confidence,
            reason="fake action evaluator",
            latency_ms=2.0,
            ollama_called=True,
            goal_session=goal_anchor.session_id,
        )


class MemoryAuditSink(AuditSink):
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record(self, event: dict[str, Any]) -> None:
        self.events.append(event)


@pytest.fixture
def action_metadata() -> ToolMetadata:
    return ToolMetadata(
        risk_level="low",
        allowed_domains=("email_assistant",),
        required_permissions=("gmail.read",),
        side_effect_level=SideEffectLevel.READ,
    )


@pytest.fixture
def goal_vault() -> GoalVault:
    return GoalVault(backend=InMemoryGoalVaultBackend(), embedder=FakeEmbedder(), default_ttl_seconds=60)


def commit_goal(vault: GoalVault, session_id: str = "action-session") -> GoalAnchor:
    return vault.commit_goal(
        session_id=session_id,
        application_name="email-assistant",
        goal="Summarize unread emails",
    )


def make_gate(
    goal_vault: GoalVault,
    *,
    evaluator: FakeActionEvaluator | None = None,
    audit_sink: MemoryAuditSink | None = None,
    config: ActionGateConfig | None = None,
) -> ActionGate:
    return ActionGate(
        goal_vault=goal_vault,
        embedder=FakeEmbedder(),
        evaluator=evaluator,
        audit_sink=audit_sink,
        config=config or ActionGateConfig(high_similarity=0.8, low_similarity=0.2, minimum_llm_confidence=0.75),
    )


def test_execute_path_uses_high_cosine_shortcut(
    goal_vault: GoalVault,
    policy: DomainPolicy,
    action_metadata: ToolMetadata,
) -> None:
    commit_goal(goal_vault)
    evaluator = FakeActionEvaluator()
    gate = make_gate(goal_vault, evaluator=evaluator)
    action = ProposedToolAction(
        tool_name="gmail.read",
        tool_description="Read unread email messages",
        tool_arguments={"label": "UNREAD"},
    )

    decision = gate.evaluate_action(
        session_id="action-session",
        action=action,
        tool_metadata=action_metadata,
        policy=policy,
    )

    assert decision.verdict == ActionVerdict.EXECUTE
    assert decision.decision_source == ActionDecisionSource.COSINE
    assert decision.ollama_called is False
    assert evaluator.calls == 0
    assert decision.goal_similarity == pytest.approx(1.0)


def test_block_path_uses_low_cosine_shortcut(
    goal_vault: GoalVault,
    policy: DomainPolicy,
    action_metadata: ToolMetadata,
) -> None:
    commit_goal(goal_vault)
    evaluator = FakeActionEvaluator()
    gate = make_gate(goal_vault, evaluator=evaluator)
    action = ProposedToolAction(
        tool_name="terminal.execute_python",
        tool_description="Execute Python code on the local system",
        tool_arguments={"code": "print('not email')"},
    )

    decision = gate.evaluate_action(
        session_id="action-session",
        action=action,
        tool_metadata=action_metadata,
        policy=policy,
    )

    assert decision.verdict == ActionVerdict.BLOCK
    assert decision.decision_source == ActionDecisionSource.COSINE
    assert decision.ollama_called is False
    assert evaluator.calls == 0
    assert decision.goal_similarity == pytest.approx(-1.0)


def test_uncertainty_band_calls_ollama_and_can_justify(
    goal_vault: GoalVault,
    policy: DomainPolicy,
    action_metadata: ToolMetadata,
) -> None:
    commit_goal(goal_vault)
    evaluator = FakeActionEvaluator(ActionVerdict.JUSTIFY, confidence=0.91)
    gate = make_gate(goal_vault, evaluator=evaluator)
    action = ProposedToolAction(
        tool_name="gmail.send",
        tool_description="Send an email message",
        tool_arguments={"to": "customer@example.com", "body": "Here is the summary"},
    )

    decision = gate.evaluate_action(
        session_id="action-session",
        action=action,
        tool_metadata=action_metadata,
        policy=policy,
    )

    assert decision.verdict == ActionVerdict.JUSTIFY
    assert decision.decision_source == ActionDecisionSource.OLLAMA
    assert decision.ollama_called is True
    assert evaluator.calls == 1
    assert decision.goal_similarity == pytest.approx(0.6)


def test_uncertainty_band_low_llm_confidence_becomes_justify(
    goal_vault: GoalVault,
    policy: DomainPolicy,
    action_metadata: ToolMetadata,
) -> None:
    commit_goal(goal_vault)
    evaluator = FakeActionEvaluator(ActionVerdict.EXECUTE, confidence=0.5)
    gate = make_gate(goal_vault, evaluator=evaluator)

    decision = gate.evaluate_action(
        session_id="action-session",
        action=ProposedToolAction("gmail.send", "Send email", {"body": "summary"}),
        tool_metadata=action_metadata,
        policy=policy,
    )

    assert decision.verdict == ActionVerdict.JUSTIFY
    assert decision.metadata["original_verdict"] == "EXECUTE"
    assert decision.metadata["minimum_llm_confidence"] == 0.75


def test_evaluator_failure_returns_fallback(
    goal_vault: GoalVault,
    policy: DomainPolicy,
    action_metadata: ToolMetadata,
) -> None:
    commit_goal(goal_vault)
    evaluator = FakeActionEvaluator(exc=ActionEvaluatorError("ollama unavailable"))
    gate = make_gate(goal_vault, evaluator=evaluator)

    decision = gate.evaluate_action(
        session_id="action-session",
        action=ProposedToolAction("gmail.send", "Send email", {"body": "summary"}),
        tool_metadata=action_metadata,
        policy=policy,
    )

    assert decision.verdict == ActionVerdict.BLOCK
    assert decision.decision_source == ActionDecisionSource.FALLBACK
    assert decision.ollama_called is True


def test_missing_goal_anchor_returns_fallback(
    goal_vault: GoalVault,
    policy: DomainPolicy,
    action_metadata: ToolMetadata,
) -> None:
    gate = make_gate(goal_vault)

    decision = gate.evaluate_action(
        session_id="missing-session",
        action=ProposedToolAction("gmail.read", "Read email", {"label": "UNREAD"}),
        tool_metadata=action_metadata,
        policy=policy,
    )

    assert decision.verdict == ActionVerdict.BLOCK
    assert decision.decision_source == ActionDecisionSource.FALLBACK
    assert decision.ollama_called is False


def test_goal_vault_backend_unavailable_returns_fallback(
    policy: DomainPolicy,
    action_metadata: ToolMetadata,
) -> None:
    class UnavailableGoalVault:
        embedder = FakeEmbedder()

        def get_anchor(self, session_id: str) -> GoalAnchor:
            raise GoalBackendUnavailableError("redis unavailable")

    gate = ActionGate(goal_vault=UnavailableGoalVault(), embedder=FakeEmbedder())

    decision = gate.evaluate_action(
        session_id="redis-down",
        action=ProposedToolAction("gmail.read", "Read email", {"label": "UNREAD"}),
        tool_metadata=action_metadata,
        policy=policy,
    )

    assert decision.verdict == ActionVerdict.BLOCK
    assert decision.decision_source == ActionDecisionSource.FALLBACK
    assert "redis unavailable" in decision.reason


def test_goal_integrity_failure_returns_fallback(
    goal_vault: GoalVault,
    policy: DomainPolicy,
    action_metadata: ToolMetadata,
) -> None:
    anchor = commit_goal(goal_vault)
    tampered = GoalAnchor(
        session_id=anchor.session_id,
        application_name=anchor.application_name,
        original_goal="Different goal",
        normalized_goal=anchor.normalized_goal,
        goal_embedding=anchor.goal_embedding,
        embedding_model=anchor.embedding_model,
        embedding_dimension=anchor.embedding_dimension,
        integrity_hash=anchor.integrity_hash,
        created_at=anchor.created_at,
        expires_at=anchor.expires_at,
    )
    goal_vault.backend.delete_anchor(anchor.session_id)
    goal_vault.backend.commit_anchor(tampered, 60)
    gate = make_gate(goal_vault)

    decision = gate.evaluate_action(
        session_id="action-session",
        action=ProposedToolAction("gmail.read", "Read email", {"label": "UNREAD"}),
        tool_metadata=action_metadata,
        policy=policy,
    )

    assert decision.verdict == ActionVerdict.BLOCK
    assert decision.decision_source == ActionDecisionSource.FALLBACK


def test_protect_tool_executes_only_when_gate_executes(
    goal_vault: GoalVault,
    policy: DomainPolicy,
    action_metadata: ToolMetadata,
) -> None:
    commit_goal(goal_vault)
    gate = make_gate(goal_vault)
    calls = 0

    def read_email(label: str) -> str:
        nonlocal calls
        calls += 1
        return f"read {label}"

    protected = gate.protect_tool(
        read_email,
        tool_metadata=action_metadata,
        policy=policy,
        tool_name="gmail.read",
        tool_description="Read unread email messages",
    )

    result = protected("UNREAD", session_id="action-session")

    assert isinstance(result, ToolExecutionResult)
    assert result.executed is True
    assert result.result == "read UNREAD"
    assert calls == 1


def test_protect_tool_does_not_execute_block_or_justify(
    goal_vault: GoalVault,
    policy: DomainPolicy,
    action_metadata: ToolMetadata,
) -> None:
    commit_goal(goal_vault)
    gate = make_gate(goal_vault)
    calls = 0

    def execute_python(code: str) -> str:
        nonlocal calls
        calls += 1
        return code

    protected = gate.protect_tool(
        execute_python,
        tool_metadata=action_metadata,
        policy=policy,
        tool_name="terminal.execute_python",
        tool_description="Execute Python code on the local system",
    )

    result = protected("print(1)", session_id="action-session")

    assert result.executed is False
    assert result.decision.verdict == ActionVerdict.BLOCK
    assert calls == 0


def test_protect_tool_pauses_on_justify(
    goal_vault: GoalVault,
    policy: DomainPolicy,
    action_metadata: ToolMetadata,
) -> None:
    commit_goal(goal_vault)
    gate = make_gate(goal_vault, evaluator=FakeActionEvaluator(ActionVerdict.JUSTIFY))
    calls = 0

    def send_email(body: str) -> str:
        nonlocal calls
        calls += 1
        return body

    protected = gate.protect_tool(
        send_email,
        tool_metadata=action_metadata,
        policy=policy,
        tool_name="gmail.send",
        tool_description="Send an email message",
    )

    result = protected("summary", session_id="action-session")

    assert result.executed is False
    assert result.decision.verdict == ActionVerdict.JUSTIFY
    assert calls == 0


def test_audit_event_contains_required_fields(
    goal_vault: GoalVault,
    policy: DomainPolicy,
    action_metadata: ToolMetadata,
) -> None:
    commit_goal(goal_vault)
    audit = MemoryAuditSink()
    gate = make_gate(goal_vault, audit_sink=audit)

    gate.evaluate_action(
        session_id="action-session",
        action=ProposedToolAction("gmail.read", "Read email", {"label": "UNREAD"}),
        tool_metadata=action_metadata,
        policy=policy,
    )

    event = audit.events[0]
    assert event["event_type"] == "ACTION_GATE_DECISION"
    assert event["session_id"] == "action-session"
    assert event["tool"] == "gmail.read"
    assert event["arguments"] == {"label": "UNREAD"}
    assert event["similarity"] == pytest.approx(1.0)
    assert event["ollama_called"] is False
    assert event["decision_source"] == "COSINE"
    assert event["verdict"] == "EXECUTE"
    assert event["latency_ms"] >= 0


def test_action_embedding_text_uses_all_required_inputs(
    goal_vault: GoalVault,
    policy: DomainPolicy,
    action_metadata: ToolMetadata,
) -> None:
    anchor = commit_goal(goal_vault)
    context = ActionRuntimeContext(
        reasoning_summary="Need to read unread messages",
        previous_approved_action="gmail.list",
        session_metadata={"case_id": "abc"},
    )
    action = ProposedToolAction("gmail.read", "Read email", {"label": "UNREAD"})

    text = build_action_embedding_text(
        goal_anchor=anchor,
        action=action,
        tool_metadata=action_metadata,
        policy=policy,
        runtime_context=context,
    )

    assert "Summarize unread emails" in text
    assert "gmail.read" in text
    assert "Read email" in text
    assert "UNREAD" in text
    assert "required_permissions" in text
    assert "Need to read unread messages" in text
    assert policy.purpose in text


def test_action_gate_works_with_loaded_policy_file(
    goal_vault: GoalVault,
    action_metadata: ToolMetadata,
) -> None:
    commit_goal(goal_vault)
    policy = load_policy("evaluation/policies/email_assistant.yaml")
    gate = make_gate(goal_vault)

    decision = gate.evaluate_action(
        session_id="action-session",
        action=ProposedToolAction("gmail.read", "Read email", {"label": "UNREAD"}),
        tool_metadata=action_metadata,
        policy=policy,
    )

    assert decision.verdict == ActionVerdict.EXECUTE


def test_tool_metadata_validation() -> None:
    with pytest.raises(ActionGateValidationError):
        ToolMetadata(risk_level="")


def test_config_validation() -> None:
    with pytest.raises(ActionGateValidationError):
        ActionGateConfig(high_similarity=0.2, low_similarity=0.3)


def test_action_evaluator_repairs_missing_reason() -> None:
    evaluator = OllamaActionEvaluator(model="fake")

    output, metadata = evaluator._parse_model_json('{"verdict":"EXECUTE","confidence":0.88}')

    assert output.verdict == ActionVerdict.EXECUTE
    assert output.confidence == pytest.approx(0.88)
    assert output.reason
    assert metadata["output_repaired"] is True
    assert metadata["repair_reason"] == "missing_reason"


def test_action_evaluator_invalid_verdict_still_rejected() -> None:
    evaluator = OllamaActionEvaluator(model="fake")

    with pytest.raises(MalformedActionEvaluatorResponseError):
        evaluator._parse_model_json('{"verdict":"ALLOW","confidence":0.88}')


def test_action_evaluator_invalid_confidence_still_rejected() -> None:
    evaluator = OllamaActionEvaluator(model="fake")

    with pytest.raises(MalformedActionEvaluatorResponseError):
        evaluator._parse_model_json('{"verdict":"EXECUTE","confidence":1.4}')


def test_action_evaluator_does_not_repair_extra_malformed_fields() -> None:
    evaluator = OllamaActionEvaluator(model="fake")

    with pytest.raises(MalformedActionEvaluatorResponseError):
        evaluator._parse_model_json('{"verdict":"EXECUTE","confidence":0.88,"extra":"not allowed"}')


def test_cosine_similarity_validation() -> None:
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
    with pytest.raises(ActionGateValidationError):
        cosine_similarity([1], [1, 0])
    with pytest.raises(ActionGateValidationError):
        cosine_similarity([0, 0], [1, 0])
