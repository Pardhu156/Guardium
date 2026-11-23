from __future__ import annotations

import json
from unittest.mock import Mock

import pytest

from aegisvault.evaluators import OllamaScopeEvaluator
from aegisvault.gates import ResponseGate
from aegisvault.policy import load_policy
from aegisvault.policy.models import DomainPolicy
from aegisvault.types import EvaluationContext, GateType, Verdict

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


@pytest.mark.parametrize("reply,retrieved,raw_verdict,confidence,expected", [
    ("Friday at 3:00 PM.", True, "ALLOW", 0.95, Verdict.ALLOW),
    ("Friday at 3:00 PM.", True, "ALLOW", 0.45, Verdict.BLOCK),
    ("An unrelated private detail.", True, "BLOCK", 0.95, Verdict.BLOCK),
    ("I read the email. Friday at 3:00 PM.", False, "BLOCK", 0.95, Verdict.BLOCK),
    ("I could not retrieve the meeting email.", False, "ALLOW", 0.95, Verdict.ALLOW),
])
def test_gmail_chat_checks_answer_against_question_under_existing_thresholds(
    monkeypatch, reply, retrieved, raw_verdict, confidence, expected
):
    policy = load_policy("mcp/gmail/policy.yaml")
    evaluator = OllamaScopeEvaluator.from_policy(policy)
    context = EvaluationContext(request_text="When is my meeting?",
                                metadata={"integration": "gmail_mcp", "read_only_chat": True,
                                          "message_retrieved": retrieved})
    http_result = Mock()
    http_result.json.return_value = {
        "response": json.dumps({"verdict": raw_verdict, "reason": "test", "confidence": confidence})
    }
    post = Mock(return_value=http_result)
    monkeypatch.setattr("aegisvault.evaluators.ollama.requests.post", post)
    assert ResponseGate(policy, evaluator).evaluate(reply, context).verdict == expected
    payload = post.call_args.kwargs["json"]
    assert json.loads(payload["prompt"]) == {"original_request": context.request_text, "candidate_reply": reply,
                                             "message_retrieved": retrieved}
    assert context.request_text not in payload["system"] and reply not in payload["system"]
    assert policy.purpose in payload["system"]

    # Other Gmail workflows retain their established verifier input.
    context.metadata.pop("read_only_chat")
    ResponseGate(policy, evaluator).evaluate(reply, context)
    existing = post.call_args.kwargs["json"]
    assert "system" not in existing
    assert context.request_text not in existing["prompt"] and existing["prompt"].endswith(reply + "\n")
