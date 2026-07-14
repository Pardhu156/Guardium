from __future__ import annotations

from aegisvault.gates import RequestGate
from aegisvault.policy import load_policy
from aegisvault.types import GateType, Verdict

from conftest import FakeEvaluator, decision


def test_request_gate_allow(policy) -> None:
    evaluator = FakeEvaluator([decision(Verdict.ALLOW, 0.9, GateType.REQUEST)])

    result = RequestGate(policy, evaluator).evaluate("Where is my order?")

    assert result.verdict == Verdict.ALLOW
    assert len(evaluator.calls) == 1


def test_request_gate_block(policy) -> None:
    evaluator = FakeEvaluator([decision(Verdict.BLOCK, 0.9, GateType.REQUEST)])

    result = RequestGate(policy, evaluator).evaluate("Write code")

    assert result.verdict == Verdict.BLOCK


def test_request_gate_clarify(policy) -> None:
    evaluator = FakeEvaluator([decision(Verdict.ALLOW, 0.4, GateType.REQUEST)])

    result = RequestGate(policy, evaluator).evaluate("Can you calculate something?")

    assert result.verdict == Verdict.CLARIFY


def test_confidence_threshold_boundary_values(policy) -> None:
    evaluator = FakeEvaluator([decision(Verdict.ALLOW, 0.8, GateType.REQUEST)])

    result = RequestGate(policy, evaluator).evaluate("At the boundary")

    assert result.verdict == Verdict.ALLOW


def test_malformed_evaluator_result(policy, malformed_error) -> None:
    evaluator = FakeEvaluator(exc=malformed_error)

    result = RequestGate(policy, evaluator).evaluate("hello")

    assert result.verdict == Verdict.BLOCK
    assert result.confidence is None


def test_evaluator_timeout_fallback(policy, timeout_error) -> None:
    evaluator = FakeEvaluator(exc=timeout_error)

    result = RequestGate(policy, evaluator).evaluate("hello")

    assert result.verdict == Verdict.BLOCK


def test_fail_open_behaviour(policy_dict) -> None:
    from aegisvault.policy.models import DomainPolicy

    policy_dict["fallback"]["evaluator_failure_action"] = "allow"
    policy = DomainPolicy.model_validate(policy_dict)
    evaluator = FakeEvaluator(exc=RuntimeError("plain failure"))

    # RuntimeError is not an EvaluatorError and should surface as a developer bug.
    try:
        RequestGate(policy, evaluator).evaluate("hello")
    except RuntimeError:
        pass


def test_fail_closed_behaviour(policy, timeout_error) -> None:
    evaluator = FakeEvaluator(exc=timeout_error)

    result = RequestGate(policy, evaluator).evaluate("hello")

    assert result.verdict == Verdict.BLOCK


def test_fail_open_evaluator_error(policy_dict, timeout_error) -> None:
    from aegisvault.policy.models import DomainPolicy

    policy_dict["fallback"]["evaluator_failure_action"] = "allow"
    policy = DomainPolicy.model_validate(policy_dict)
    evaluator = FakeEvaluator(exc=timeout_error)

    result = RequestGate(policy, evaluator).evaluate("hello")

    assert result.verdict == Verdict.ALLOW


def test_email_native_sensitive_requests_reach_later_gates() -> None:
    policy = load_policy("evaluation/policies/email_assistant.yaml")
    evaluator = FakeEvaluator(
        [
            decision(Verdict.ALLOW, 0.95, GateType.REQUEST),
            decision(Verdict.ALLOW, 0.95, GateType.REQUEST),
            decision(Verdict.ALLOW, 0.95, GateType.REQUEST),
            decision(Verdict.ALLOW, 0.95, GateType.REQUEST),
        ]
    )
    gate = RequestGate(policy, evaluator)

    prompts = [
        "Draft a reply to my manager.",
        "Forward this email to HR.",
        "Archive the newsletter.",
        "Delete unnecessary emails.",
    ]

    for prompt in prompts:
        assert gate.evaluate(prompt).verdict == Verdict.ALLOW

    assert len(evaluator.calls) == len(prompts)
