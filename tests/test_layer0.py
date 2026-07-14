from __future__ import annotations

from typing import Any

import pytest

from aegisvault import AegisVault
from aegisvault.audit import AuditSink
from aegisvault.layer0 import Layer0Action, Layer0Validator
from aegisvault.layer0.models import Layer0Checkpoint
from aegisvault.layer0.rules import validate_schema
from aegisvault.policy.models import (
    ApplicationConfig,
    AuditConfig,
    DomainPolicy,
    EvaluatorConfig,
    GateConfig,
    GatesConfig,
    Layer0Config,
    Layer0DestinationRuleConfig,
    Layer0FailMode,
    Layer0ForbiddenPatternsConfig,
    Layer0RequestConfig,
    Layer0RuleAction,
    Layer0ToolsConfig,
    LowConfidenceAction,
)
from aegisvault.runtime.action_gate import ActionGate, ActionGateConfig, ActionRuntimeContext, ActionVerdict, SideEffectLevel, ToolMetadata
from aegisvault.runtime.goal_vault import GoalEmbedder, GoalVault, InMemoryGoalVaultBackend
from aegisvault.types import GateDecision, GateType, TerminatedBy, Verdict


class MemoryAudit(AuditSink):
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record(self, event: dict[str, Any]) -> None:
        self.events.append(event)


class AllowEvaluator:
    def evaluate(self, text: str, policy: DomainPolicy, gate_type: GateType, context: Any = None) -> GateDecision:
        return GateDecision(
            verdict=Verdict.ALLOW,
            confidence=1.0,
            reason="allow",
            gate=gate_type,
            evaluator="fake",
            latency_ms=0.0,
        )


class FakeEmbedder(GoalEmbedder):
    model_name = "fake"
    dimension = 2

    def embed(self, text: str) -> tuple[float, ...]:
        return (1.0, 0.0)


def policy(layer0: Layer0Config | None = None) -> DomainPolicy:
    gate = GateConfig(allow_threshold=0.8, block_threshold=0.8, low_confidence_action=LowConfidenceAction.BLOCK)
    return DomainPolicy(
        version="1.0",
        application=ApplicationConfig(name="email-agent", description="Email assistant"),
        purpose="Help with email workflows.",
        allowed_topics=["email"],
        gates=GatesConfig(request=gate, response=gate),
        evaluator=EvaluatorConfig(provider="ollama", model="llama3.2"),
        audit=AuditConfig(enabled=False),
        layer0=layer0 or Layer0Config(),
    )


def enabled_policy(**kwargs: Any) -> DomainPolicy:
    request = kwargs.pop("request", Layer0RequestConfig(require_session_id=True, require_domain=True, allowed_domains=["email"]))
    tools = kwargs.pop(
        "tools",
        Layer0ToolsConfig(
            allowlist_mode=True,
            allowed=["read_email", "send_email"],
            denied=["delete_email"],
            destination_rules={
                "send_email": Layer0DestinationRuleConfig(fields=["to", "recipients"], allowed_values=["manager@example.com"])
            },
        ),
    )
    return policy(Layer0Config(enabled=True, request=request, tools=tools, **kwargs))


def validator(custom_policy: DomainPolicy | None = None, audit: MemoryAudit | None = None) -> Layer0Validator:
    return Layer0Validator(policy=custom_policy or enabled_policy(), audit_sink=audit)


def test_valid_request_is_allowed() -> None:
    decision = validator().validate_request(session_id="s1", request_text="Summarize email", domain="email")
    assert decision.allowed is True
    assert decision.decision == Layer0Action.ALLOW


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_request_is_blocked(text: str) -> None:
    decision = validator().validate_request(session_id="s1", request_text=text, domain="email")
    assert decision.allowed is False
    assert decision.rule_id == "L0_REQUEST_EMPTY"


def test_non_string_request_is_blocked() -> None:
    decision = validator().validate_request(session_id="s1", request_text={"x": 1}, domain="email")
    assert decision.rule_id == "L0_REQUEST_TYPE_INVALID"


def test_oversized_request_is_blocked() -> None:
    custom = enabled_policy(request=Layer0RequestConfig(max_characters=3, max_bytes=10))
    decision = validator(custom).validate_request(session_id="s1", request_text="abcd", domain="email")
    assert decision.rule_id == "L0_REQUEST_TOO_LARGE"


def test_missing_required_session_id_and_domain_are_blocked() -> None:
    assert validator().validate_request(session_id=None, request_text="hi", domain="email").rule_id == "L0_SESSION_MISSING"
    assert validator().validate_request(session_id="s1", request_text="hi", domain=None).rule_id == "L0_DOMAIN_MISSING"


def test_domain_allowlist() -> None:
    assert validator().validate_request(session_id="s1", request_text="hi", domain="email").allowed
    assert validator().validate_request(session_id="s1", request_text="hi", domain="coding").rule_id == "L0_DOMAIN_NOT_ALLOWED"


def test_reserved_metadata_and_goal_overwrite_are_blocked() -> None:
    decision = validator().validate_request(session_id="s1", request_text="hi", domain="email", metadata={"nested": {"trusted_goal": "x"}})
    assert decision.rule_id == "L0_RESERVED_METADATA_KEY"
    with_goal = Layer0Validator(policy=enabled_policy(), trusted_goal_exists=lambda session_id: True)
    decision = with_goal.validate_request(session_id="s1", request_text="hi", domain="email", requested_goal_update="replace")
    assert decision.rule_id == "L0_GOAL_OVERWRITE_ATTEMPT"


def test_configured_forbidden_patterns_and_ordinary_ignore() -> None:
    normal = validator().validate_request(session_id="s1", request_text="Please ignore archived emails.", domain="email")
    assert normal.allowed
    custom = enabled_policy(
        request=Layer0RequestConfig(
            forbidden_patterns=Layer0ForbiddenPatternsConfig(literals=["FORBIDDEN_LITERAL"], regex=[r"secret-\d+"])
        )
    )
    assert validator(custom).validate_request(session_id="s1", request_text="FORBIDDEN_LITERAL", domain="email").rule_id == "L0_FORBIDDEN_PATTERN"
    assert validator(custom).validate_request(session_id="s1", request_text="secret-123", domain="email").rule_id == "L0_FORBIDDEN_PATTERN"


def test_request_input_is_not_mutated() -> None:
    metadata = {"nested": {"safe": "value"}}
    validator().validate_request(session_id="s1", request_text="hi", domain="email", metadata=metadata)
    assert metadata == {"nested": {"safe": "value"}}


def test_tool_allow_deny_and_declaration_rules() -> None:
    catalog = {"read_email": {"parameters": {"type": "object", "required": ["id"], "properties": {"id": {"type": "string"}}}}}
    assert validator().validate_tool_call(session_id="s1", tool_name="read_email", arguments={"id": "1"}, tool_catalog=catalog).allowed
    assert validator().validate_tool_call(session_id="s1", tool_name="", arguments={}, tool_catalog=catalog).rule_id == "L0_TOOL_NAME_MISSING"
    assert validator().validate_tool_call(session_id="s1", tool_name="unknown", arguments={}, tool_catalog=catalog).rule_id == "L0_TOOL_UNDECLARED"
    assert validator().validate_tool_call(session_id="s1", tool_name="delete_email", arguments={}, tool_catalog={"delete_email": {}}).rule_id == "L0_TOOL_DENIED"
    assert validator().validate_tool_call(session_id="s1", tool_name="other", arguments={}, tool_catalog={}).rule_id == "L0_TOOL_NOT_ALLOWED"


def test_tool_argument_and_schema_rules() -> None:
    catalog = {"read_email": {"parameters": {"type": "object", "required": ["id"], "properties": {"id": {"type": "string"}}}}}
    assert validator().validate_tool_call(session_id="s1", tool_name="read_email", arguments=[], tool_catalog=catalog).rule_id == "L0_TOOL_ARGUMENTS_INVALID"
    assert validator().validate_tool_call(session_id="s1", tool_name="read_email", arguments={}, tool_catalog=catalog).rule_id == "L0_TOOL_SCHEMA_INVALID"
    assert validate_schema(catalog["read_email"]["parameters"], {"id": 1}) == "argument 'id' must be string"


def test_tool_argument_size_reserved_and_secret_rules() -> None:
    custom = enabled_policy(tools=Layer0ToolsConfig(allowed=["read_email"], max_argument_bytes=8, sensitive_argument_action=Layer0RuleAction.WARN))
    assert validator(custom).validate_tool_call(session_id="s1", tool_name="read_email", arguments={"x": "123456789"}).rule_id == "L0_TOOL_ARGUMENT_TOO_LARGE"
    assert validator().validate_tool_call(session_id="s1", tool_name="read_email", arguments={"nested": {"policy_override": True}}).rule_id == "L0_TOOL_RESERVED_ARGUMENT"
    secret_policy = enabled_policy(tools=Layer0ToolsConfig(allowed=["read_email"], sensitive_argument_action=Layer0RuleAction.WARN))
    decision = validator(secret_policy).validate_tool_call(session_id="s1", tool_name="read_email", arguments={"api_key": "real-secret"})
    assert decision.allowed
    assert decision.decision == Layer0Action.WARN


def test_sensitive_values_are_redacted_from_audit() -> None:
    audit = MemoryAudit()
    custom = enabled_policy(tools=Layer0ToolsConfig(allowed=["read_email"], sensitive_argument_action=Layer0RuleAction.WARN))
    validator(custom, audit).validate_tool_call(session_id="s1", tool_name="read_email", arguments={"access_token": "real-secret"})
    serialized = str(audit.events)
    assert "real-secret" not in serialized
    assert "[redacted]" in serialized


def test_destination_rules() -> None:
    assert validator().validate_tool_call(session_id="s1", tool_name="send_email", arguments={"to": "manager@example.com"}).allowed
    assert validator().validate_tool_call(session_id="s1", tool_name="send_email", arguments={"to": "attacker@example.com"}).rule_id == "L0_TOOL_EXTERNAL_DESTINATION"


def test_schema_and_destination_rules_understand_protected_tool_kwargs_shape() -> None:
    catalog = {"send_email": {"parameters": {"type": "object", "required": ["to"], "properties": {"to": {"type": "string"}}}}}
    assert validator().validate_tool_call(
        session_id="s1",
        tool_name="send_email",
        arguments={"args": [], "kwargs": {"to": "manager@example.com"}},
        tool_catalog=catalog,
    ).allowed
    assert validator().validate_tool_call(
        session_id="s1",
        tool_name="send_email",
        arguments={"args": [], "kwargs": {"to": "attacker@example.com"}},
        tool_catalog=catalog,
    ).rule_id == "L0_TOOL_EXTERNAL_DESTINATION"
    assert validator().validate_tool_call(
        session_id="s1",
        tool_name="send_email",
        arguments={"args": [], "kwargs": {}},
        tool_catalog=catalog,
    ).rule_id == "L0_TOOL_SCHEMA_INVALID"


def test_tool_arguments_are_not_mutated() -> None:
    args = {"nested": {"safe": "value"}}
    validator().validate_tool_call(session_id="s1", tool_name="read_email", arguments=args)
    assert args == {"nested": {"safe": "value"}}


def test_old_policy_without_layer0_loads_and_preserves_4_2_behavior() -> None:
    guard = AegisVault(policy=policy(), evaluator=AllowEvaluator())
    called = {"count": 0}

    def app(prompt: str) -> str:
        called["count"] += 1
        return "ok"

    result = guard.wrap(app)("hello")
    assert result.final_response == "ok"
    assert result.terminated_by == TerminatedBy.APPLICATION
    assert called["count"] == 1


def test_policy_validation_errors_for_bad_layer0_config() -> None:
    with pytest.raises(Exception):
        Layer0ForbiddenPatternsConfig(regex=["["])
    with pytest.raises(Exception):
        Layer0RequestConfig(max_characters=-1)
    with pytest.raises(Exception):
        Layer0ToolsConfig(sensitive_argument_action="bad")


def test_fail_open_and_fail_closed_internal_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*args: Any, **kwargs: Any) -> list[Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr("aegisvault.layer0.validator.request_rules", explode)
    closed = validator(enabled_policy(fail_mode=Layer0FailMode.CLOSED)).validate_request(session_id="s1", request_text="hi", domain="email")
    open_ = validator(enabled_policy(fail_mode=Layer0FailMode.OPEN)).validate_request(session_id="s1", request_text="hi", domain="email")
    assert not closed.allowed
    assert open_.allowed
    assert open_.decision == Layer0Action.WARN


def test_known_violations_still_block_in_fail_open() -> None:
    decision = validator(enabled_policy(fail_mode=Layer0FailMode.OPEN)).validate_request(session_id="s1", request_text="", domain="email")
    assert not decision.allowed


def test_request_layer0_runs_before_request_gate_and_blocks_agent() -> None:
    guard = AegisVault(policy=enabled_policy(), evaluator=AllowEvaluator())
    called = {"count": 0}

    def app(prompt: str) -> str:
        called["count"] += 1
        return "ok"

    result = guard.wrap(app)("", session_id="s1", metadata={"domain": "email"})
    assert result.terminated_by == TerminatedBy.LAYER0
    assert called["count"] == 0


def test_tool_layer0_runs_before_action_gate_and_blocks_execution() -> None:
    vault = GoalVault(backend=InMemoryGoalVaultBackend(), embedder=FakeEmbedder())
    vault.commit_goal(session_id="s1", application_name="email-agent", goal="read email")
    gate = ActionGate(goal_vault=vault, embedder=FakeEmbedder(), config=ActionGateConfig(high_similarity=0.8, low_similarity=0.2))
    layer0 = validator(enabled_policy(tools=Layer0ToolsConfig(allowlist_mode=True, allowed=["safe_tool"], denied=["blocked_tool"])))
    executed = {"count": 0}

    def blocked_tool() -> str:
        executed["count"] += 1
        return "done"

    protected = gate.protect_tool(
        blocked_tool,
        tool_metadata=ToolMetadata(risk_level="low", side_effect_level=SideEffectLevel.READ),
        policy=enabled_policy(),
        tool_name="blocked_tool",
        layer0_validator=layer0,
    )
    result = protected(session_id="s1", runtime_context=ActionRuntimeContext())
    assert result.executed is False
    assert result.decision.verdict == ActionVerdict.BLOCK
    assert executed["count"] == 0


def test_allowed_tool_reaches_action_gate() -> None:
    vault = GoalVault(backend=InMemoryGoalVaultBackend(), embedder=FakeEmbedder())
    vault.commit_goal(session_id="s1", application_name="email-agent", goal="safe tool")
    gate = ActionGate(goal_vault=vault, embedder=FakeEmbedder(), config=ActionGateConfig(high_similarity=0.8, low_similarity=0.2))
    layer0 = validator(enabled_policy(tools=Layer0ToolsConfig(allowlist_mode=True, allowed=["safe_tool"])))

    def safe_tool() -> str:
        return "done"

    protected = gate.protect_tool(
        safe_tool,
        tool_metadata=ToolMetadata(risk_level="low", side_effect_level=SideEffectLevel.READ),
        policy=enabled_policy(),
        tool_name="safe_tool",
        layer0_validator=layer0,
    )
    result = protected(session_id="s1", runtime_context=ActionRuntimeContext())
    assert result.decision.verdict == ActionVerdict.EXECUTE
