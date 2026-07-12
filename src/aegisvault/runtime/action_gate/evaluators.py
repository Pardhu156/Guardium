"""Action Gate evaluator interfaces and Ollama implementation."""

from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import requests
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from aegisvault.policy.models import DomainPolicy
from aegisvault.runtime.action_gate.exceptions import (
    ActionEvaluatorError,
    ActionEvaluatorTimeoutError,
    MalformedActionEvaluatorResponseError,
)
from aegisvault.runtime.action_gate.models import (
    ActionDecisionSource,
    ActionGateDecision,
    ActionRuntimeContext,
    ActionVerdict,
    ProposedToolAction,
    ToolMetadata,
    thaw_mapping,
)
from aegisvault.runtime.goal_vault import GoalAnchor

logger = logging.getLogger(__name__)


class ActionEvaluator(ABC):
    """Abstract verifier for uncertain proposed tool actions."""

    @abstractmethod
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
        """Return an Action Gate decision for an uncertain action."""


class _ActionEvaluatorOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: ActionVerdict
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1)


class OllamaActionEvaluator(ActionEvaluator):
    """Ollama-backed Action Gate verifier for ambiguous tool calls."""

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
    def from_policy(cls, policy: DomainPolicy) -> "OllamaActionEvaluator":
        """Create an action evaluator from the existing policy evaluator config."""

        return cls(
            model=policy.evaluator.model,
            base_url=policy.evaluator.base_url,
            timeout_seconds=policy.evaluator.timeout_seconds,
            temperature=policy.evaluator.temperature,
        )

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
        prompt = self._build_prompt(
            goal_anchor=goal_anchor,
            action=action,
            tool_metadata=tool_metadata,
            policy=policy,
            runtime_context=runtime_context,
            goal_similarity=goal_similarity,
        )
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
            raise ActionEvaluatorTimeoutError(
                f"Ollama action evaluation timed out after {self.timeout_seconds} seconds"
            ) from exc
        except requests.RequestException as exc:
            logger.warning("Ollama action evaluator request failed: %s", exc)
            raise ActionEvaluatorError(f"Ollama action evaluator request failed: {exc}") from exc

        latency_ms = (time.perf_counter() - started) * 1000
        try:
            envelope = response.json()
        except ValueError as exc:
            raise MalformedActionEvaluatorResponseError("Ollama returned non-JSON HTTP response") from exc

        if not isinstance(envelope, dict) or not isinstance(envelope.get("response"), str):
            raise MalformedActionEvaluatorResponseError("Ollama response must contain a string 'response' field")

        output = self._parse_model_json(envelope["response"])
        return ActionGateDecision(
            tool_name=action.tool_name,
            tool_arguments=action.tool_arguments,
            goal_similarity=goal_similarity,
            decision_source=ActionDecisionSource.OLLAMA,
            verdict=output.verdict,
            confidence=output.confidence,
            reason=output.reason,
            latency_ms=latency_ms,
            ollama_called=True,
            goal_session=goal_anchor.session_id,
            metadata={"evaluator": self.name},
        )

    def _parse_model_json(self, raw: str) -> _ActionEvaluatorOutput:
        try:
            payload: Any = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MalformedActionEvaluatorResponseError(f"Ollama model returned malformed JSON: {exc}") from exc
        try:
            return _ActionEvaluatorOutput.model_validate(payload)
        except ValidationError as exc:
            raise MalformedActionEvaluatorResponseError(f"Ollama model output failed schema validation: {exc}") from exc

    def _build_prompt(
        self,
        *,
        goal_anchor: GoalAnchor,
        action: ProposedToolAction,
        tool_metadata: ToolMetadata,
        policy: DomainPolicy,
        runtime_context: ActionRuntimeContext | None,
        goal_similarity: float | None,
    ) -> str:
        allowed = "\n".join(f"- {topic}" for topic in policy.allowed_topics)
        blocked = "\n".join(f"- {topic}" for topic in policy.blocked_topics) or "- none declared"
        context_payload = {
            "reasoning_summary": runtime_context.reasoning_summary if runtime_context else None,
            "previous_approved_action": runtime_context.previous_approved_action if runtime_context else None,
            "session_metadata": thaw_mapping(runtime_context.session_metadata) if runtime_context else {},
        }
        return f"""You are AegisVault Action Gate, a strict verifier for proposed tool execution.

Decide whether the proposed tool action aligns with the immutable original goal and current policy.

Application name: {policy.application.name}
Application description: {policy.application.description}
Application purpose: {policy.purpose}

Allowed policy topics:
{allowed}

Blocked policy topics:
{blocked}

Immutable goal anchor:
- original_goal: {goal_anchor.original_goal}
- normalized_goal: {goal_anchor.normalized_goal}
- embedding_model: {goal_anchor.embedding_model}
- embedding_dimension: {goal_anchor.embedding_dimension}

Proposed tool action:
- tool_name: {action.tool_name}
- tool_description: {action.tool_description}
- tool_arguments: {json.dumps(thaw_mapping(action.tool_arguments), ensure_ascii=False, sort_keys=True)}

Tool metadata:
- risk_level: {tool_metadata.risk_level}
- allowed_domains: {json.dumps(list(tool_metadata.allowed_domains), ensure_ascii=False)}
- required_permissions: {json.dumps(list(tool_metadata.required_permissions), ensure_ascii=False)}
- side_effect_level: {tool_metadata.side_effect_level.value}
- requires_approval: {tool_metadata.requires_approval}

Runtime context:
{json.dumps(context_payload, ensure_ascii=False, sort_keys=True)}

Cosine similarity between goal and proposed action: {goal_similarity}

Verdict definitions:
- EXECUTE: The tool action is clearly aligned with the immutable goal and permitted policy scope.
- JUSTIFY: The tool action is not clearly safe or unsafe. The caller must explicitly decide whether to continue.
- BLOCK: The tool action conflicts with the immutable goal, policy scope, or required safety constraints.

Return JSON only. Do not include markdown, commentary, or extra fields.
Required JSON schema:
{{"verdict":"EXECUTE|JUSTIFY|BLOCK","confidence":0.0-1.0,"reason":"brief reason"}}
"""
