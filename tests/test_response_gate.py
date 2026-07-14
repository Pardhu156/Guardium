from __future__ import annotations

from aegisvault.gates import ResponseGate
from aegisvault.policy import load_policy
from aegisvault.policy.models import DomainPolicy
from aegisvault.types import GateType, Verdict

from conftest import FakeEvaluator, decision


def test_response_gate_allow(policy) -> None:
    evaluator = FakeEvaluator([decision(Verdict.ALLOW, 0.9, GateType.RESPONSE)])

    result = ResponseGate(policy, evaluator).evaluate("Your order ships tomorrow.")

    assert result.verdict == Verdict.ALLOW


def test_response_gate_block(policy) -> None:
    evaluator = FakeEvaluator([decision(Verdict.BLOCK, 0.9, GateType.RESPONSE)])

    result = ResponseGate(policy, evaluator).evaluate("Here is legal advice.")

    assert result.verdict == Verdict.BLOCK


def test_response_gate_replacement(policy_dict) -> None:
    policy_dict["gates"]["response"]["low_confidence_action"] = "replace"
    policy = DomainPolicy.model_validate(policy_dict)
    evaluator = FakeEvaluator([decision(Verdict.ALLOW, 0.2, GateType.RESPONSE)])

    result = ResponseGate(policy, evaluator).evaluate("Maybe related.")

    assert result.verdict == Verdict.REPLACE


def test_valid_drafted_email_passes_response_gate() -> None:
    policy = load_policy("evaluation/policies/email_assistant.yaml")
    evaluator = FakeEvaluator([decision(Verdict.ALLOW, 0.92, GateType.RESPONSE)])

    result = ResponseGate(policy, evaluator).evaluate(
        "Draft: Hi Maya, thanks for the update. I will review the invoice and reply by Friday."
    )

    assert result.verdict == Verdict.ALLOW
    assert len(evaluator.calls) == 1
