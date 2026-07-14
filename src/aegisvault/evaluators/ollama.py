"""Ollama HTTP scope evaluator."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from aegisvault.evaluators.base import ScopeEvaluator
from aegisvault.exceptions import EvaluatorError, EvaluatorTimeoutError, MalformedEvaluatorResponseError
from aegisvault.policy.models import DomainPolicy
from aegisvault.types import EvaluationContext, GateDecision, GateType, Verdict

logger = logging.getLogger(__name__)


class _EvaluatorOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)

    @field_validator("verdict")
    @classmethod
    def validate_verdict(cls, value: Verdict) -> Verdict:
        if value not in {Verdict.ALLOW, Verdict.BLOCK}:
            raise ValueError("evaluator verdict must be ALLOW or BLOCK")
        return value


class OllamaScopeEvaluator(ScopeEvaluator):
    """Scope evaluator backed by Ollama's local HTTP API."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_seconds: float = 30,
        temperature: float = 0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.temperature = temperature
        self.name = f"ollama:{model}"

    @classmethod
    def from_policy(cls, policy: DomainPolicy) -> "OllamaScopeEvaluator":
        """Create an Ollama evaluator from policy configuration."""

        return cls(
            model=policy.evaluator.model,
            base_url=policy.evaluator.base_url,
            timeout_seconds=policy.evaluator.timeout_seconds,
            temperature=policy.evaluator.temperature,
        )

    def evaluate(
        self,
        text: str,
        policy: DomainPolicy,
        gate_type: GateType,
        context: EvaluationContext | None = None,
    ) -> GateDecision:
        """Evaluate text with Ollama and return a schema-validated decision."""

        prompt = self._build_prompt(text=text, policy=policy, gate_type=gate_type)
        started = time.perf_counter()
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": self.temperature},
                },
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.Timeout as exc:
            raise EvaluatorTimeoutError(f"Ollama evaluation timed out after {self.timeout_seconds} seconds") from exc
        except requests.RequestException as exc:
            logger.warning("Ollama evaluator request failed: %s", exc)
            raise EvaluatorError(f"Ollama evaluator request failed: {exc}") from exc

        latency_ms = (time.perf_counter() - started) * 1000
        try:
            envelope = response.json()
        except ValueError as exc:
            raise MalformedEvaluatorResponseError("Ollama returned non-JSON HTTP response") from exc

        if not isinstance(envelope, dict) or not isinstance(envelope.get("response"), str):
            raise MalformedEvaluatorResponseError("Ollama response must contain a string 'response' field")

        output = self._parse_model_json(envelope["response"])
        return GateDecision(
            verdict=output.verdict,
            confidence=output.confidence,
            reason=output.reason,
            gate=gate_type,
            evaluator=self.name,
            latency_ms=latency_ms,
            metadata={"provider": "ollama"},
        )

    def _parse_model_json(self, raw: str) -> _EvaluatorOutput:
        try:
            payload: Any = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MalformedEvaluatorResponseError(f"Ollama model returned malformed JSON: {exc}") from exc

        try:
            return _EvaluatorOutput.model_validate(payload)
        except ValidationError as exc:
            raise MalformedEvaluatorResponseError(f"Ollama model output failed schema validation: {exc}") from exc

    def _build_prompt(self, *, text: str, policy: DomainPolicy, gate_type: GateType) -> str:
        allowed = "\n".join(f"- {topic}" for topic in policy.allowed_topics)
        blocked = "\n".join(f"- {topic}" for topic in policy.blocked_topics) or "- none declared"
        gate_label = "user request" if gate_type == GateType.REQUEST else "generated response"
        gate_guidance = (
            "For request evaluation, decide whether the user is asking for help within the application domain. "
            "Do not block email-native operations only because they are risky, destructive, underspecified, or involve "
            "external recipients; if they are email operations, return ALLOW so later gates can evaluate action risk."
            if gate_type == GateType.REQUEST
            else "For response evaluation, allow valid in-domain assistant outputs such as email drafts, reply text, "
            "summaries, search results, action items, classifications, and concise status messages about simulated "
            "email actions. Block only responses that provide content outside the declared application purpose."
        )
        return f"""You are AegisVault, a strict domain-scope evaluator.

Application name: {policy.application.name}
Application description: {policy.application.description}
Application purpose: {policy.purpose}

Allowed topics:
{allowed}

Blocked topics:
{blocked}

You are evaluating a {gate_label}.

Gate-specific instruction:
{gate_guidance}

Verdict definitions:
- ALLOW: The text is within the application's declared purpose and allowed topics.
- BLOCK: The text is outside the application's purpose, asks for a blocked topic, or causes the application to drift outside scope.

Return JSON only. Do not include markdown, commentary, or extra fields.
Required JSON schema:
{{"verdict":"ALLOW|BLOCK","confidence":0.0-1.0,"reason":"brief reason"}}

Text to evaluate:
{text}
"""
