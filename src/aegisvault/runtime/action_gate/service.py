"""Action Gate service."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from functools import wraps
from typing import Any
from uuid import uuid4

from aegisvault.audit import AuditSink, NullAuditSink
from aegisvault.policy.models import DomainPolicy
from aegisvault.runtime.action_gate.cosine import cosine_similarity
from aegisvault.runtime.action_gate.evaluators import ActionEvaluator, OllamaActionEvaluator
from aegisvault.runtime.action_gate.exceptions import ActionEvaluatorError, ActionGateError
from aegisvault.runtime.action_gate.models import (
    ActionDecisionSource,
    ActionGateConfig,
    ActionGateDecision,
    ActionRuntimeContext,
    ActionVerdict,
    ProposedToolAction,
    ToolExecutionResult,
    ToolMetadata,
    thaw_mapping,
)
from aegisvault.runtime.goal_vault import GoalEmbedder, GoalVault, l2_normalize


class ActionGate:
    """Protects tool execution against the immutable goal anchor."""

    def __init__(
        self,
        *,
        goal_vault: GoalVault,
        embedder: GoalEmbedder | None = None,
        evaluator: ActionEvaluator | None = None,
        config: ActionGateConfig | None = None,
        audit_sink: AuditSink | None = None,
    ) -> None:
        self.goal_vault = goal_vault
        self.embedder = embedder or goal_vault.embedder
        self.evaluator = evaluator
        self.config = config or ActionGateConfig()
        self.audit_sink = audit_sink or NullAuditSink()

    def evaluate_action(
        self,
        *,
        session_id: str,
        action: ProposedToolAction,
        tool_metadata: ToolMetadata,
        policy: DomainPolicy,
        runtime_context: ActionRuntimeContext | None = None,
    ) -> ActionGateDecision:
        """Evaluate whether a proposed tool action may execute."""

        started = time.perf_counter()
        try:
            goal_anchor = self.goal_vault.get_anchor(session_id)
            action_text = build_action_embedding_text(
                goal_anchor=goal_anchor,
                action=action,
                tool_metadata=tool_metadata,
                policy=policy,
                runtime_context=runtime_context,
            )
            action_embedding = l2_normalize(self.embedder.embed(action_text), goal_anchor.embedding_dimension)
            similarity = cosine_similarity(goal_anchor.goal_embedding, action_embedding)
        except Exception as exc:
            decision = self._fallback_decision(
                session_id=session_id,
                action=action,
                reason=f"Action Gate prerequisite failed: {exc}",
                started=started,
            )
            self._audit(decision)
            return decision

        if similarity >= self.config.high_similarity:
            decision = ActionGateDecision(
                tool_name=action.tool_name,
                tool_arguments=action.tool_arguments,
                goal_similarity=similarity,
                decision_source=ActionDecisionSource.COSINE,
                verdict=ActionVerdict.EXECUTE,
                confidence=similarity,
                reason="Goal/action cosine similarity met the high execute threshold.",
                latency_ms=(time.perf_counter() - started) * 1000,
                ollama_called=False,
                goal_session=goal_anchor.session_id,
                metadata={"threshold": self.config.high_similarity, "threshold_type": "high_similarity"},
            )
            self._audit(decision)
            return decision

        if similarity <= self.config.low_similarity:
            decision = ActionGateDecision(
                tool_name=action.tool_name,
                tool_arguments=action.tool_arguments,
                goal_similarity=similarity,
                decision_source=ActionDecisionSource.COSINE,
                verdict=ActionVerdict.BLOCK,
                confidence=1.0 - max(0.0, similarity),
                reason="Goal/action cosine similarity fell below the low block threshold.",
                latency_ms=(time.perf_counter() - started) * 1000,
                ollama_called=False,
                goal_session=goal_anchor.session_id,
                metadata={"threshold": self.config.low_similarity, "threshold_type": "low_similarity"},
            )
            self._audit(decision)
            return decision

        evaluator = self.evaluator or OllamaActionEvaluator.from_policy(policy)
        try:
            raw_decision = evaluator.evaluate(
                goal_anchor=goal_anchor,
                action=action,
                tool_metadata=tool_metadata,
                policy=policy,
                runtime_context=runtime_context,
                goal_similarity=similarity,
            )
        except ActionEvaluatorError as exc:
            decision = self._fallback_decision(
                session_id=session_id,
                action=action,
                reason=f"Action evaluator fallback applied: {exc}",
                started=started,
                similarity=similarity,
                ollama_called=True,
            )
            self._audit(decision)
            return decision

        if raw_decision.confidence is None or raw_decision.confidence < self.config.minimum_llm_confidence:
            decision = ActionGateDecision(
                tool_name=action.tool_name,
                tool_arguments=action.tool_arguments,
                goal_similarity=similarity,
                decision_source=ActionDecisionSource.OLLAMA,
                verdict=ActionVerdict.JUSTIFY,
                confidence=raw_decision.confidence,
                reason=(
                    "Ollama confidence did not meet the Action Gate minimum confidence threshold. "
                    f"Original reason: {raw_decision.reason}"
                ),
                latency_ms=(time.perf_counter() - started) * 1000,
                ollama_called=True,
                goal_session=goal_anchor.session_id,
                metadata={
                    **thaw_mapping(raw_decision.metadata),
                    "minimum_llm_confidence": self.config.minimum_llm_confidence,
                    "original_verdict": raw_decision.verdict.value,
                },
            )
            self._audit(decision)
            return decision

        decision = ActionGateDecision(
            tool_name=action.tool_name,
            tool_arguments=action.tool_arguments,
            goal_similarity=similarity,
            decision_source=ActionDecisionSource.OLLAMA,
            verdict=raw_decision.verdict,
            confidence=raw_decision.confidence,
            reason=raw_decision.reason,
            latency_ms=(time.perf_counter() - started) * 1000,
            ollama_called=True,
            goal_session=goal_anchor.session_id,
            metadata=raw_decision.metadata,
        )
        self._audit(decision)
        return decision

    def protect_tool(
        self,
        tool: Callable[..., Any],
        *,
        tool_metadata: ToolMetadata,
        policy: DomainPolicy,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ) -> Callable[..., ToolExecutionResult]:
        """Wrap a synchronous Python callable so Action Gate runs before execution."""

        resolved_name = tool_name or getattr(tool, "__name__", tool.__class__.__name__)
        resolved_description = tool_description or getattr(tool, "__doc__", None) or resolved_name

        @wraps(tool)
        def protected_tool(
            *args: Any,
            session_id: str,
            runtime_context: ActionRuntimeContext | None = None,
            **kwargs: Any,
        ) -> ToolExecutionResult:
            action = ProposedToolAction(
                tool_name=resolved_name,
                tool_description=resolved_description,
                tool_arguments={"args": list(args), "kwargs": kwargs},
            )
            decision = self.evaluate_action(
                session_id=session_id,
                action=action,
                tool_metadata=tool_metadata,
                policy=policy,
                runtime_context=runtime_context,
            )
            if decision.verdict != ActionVerdict.EXECUTE:
                return ToolExecutionResult(decision=decision, executed=False)
            return ToolExecutionResult(decision=decision, executed=True, result=tool(*args, **kwargs))

        return protected_tool

    def _fallback_decision(
        self,
        *,
        session_id: str,
        action: ProposedToolAction,
        reason: str,
        started: float,
        similarity: float | None = None,
        ollama_called: bool = False,
    ) -> ActionGateDecision:
        return ActionGateDecision(
            tool_name=action.tool_name,
            tool_arguments=action.tool_arguments,
            goal_similarity=similarity,
            decision_source=ActionDecisionSource.FALLBACK,
            verdict=self.config.fallback_verdict,
            confidence=None,
            reason=reason,
            latency_ms=(time.perf_counter() - started) * 1000,
            ollama_called=ollama_called,
            goal_session=session_id,
            metadata={"fallback_verdict": self.config.fallback_verdict.value},
        )

    def _audit(self, decision: ActionGateDecision) -> None:
        event = {
            "event_id": str(uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": "ACTION_GATE_DECISION",
            "session_id": decision.goal_session,
            "tool": decision.tool_name,
            "arguments": thaw_mapping(decision.tool_arguments),
            "similarity": decision.goal_similarity,
            "ollama_called": decision.ollama_called,
            "decision_source": decision.decision_source.value,
            "verdict": decision.verdict.value,
            "confidence": decision.confidence,
            "reason": decision.reason,
            "latency_ms": decision.latency_ms,
            "metadata": thaw_mapping(decision.metadata),
        }
        try:
            self.audit_sink.record(event)
        except Exception:
            return None


def build_action_embedding_text(
    *,
    goal_anchor: Any,
    action: ProposedToolAction,
    tool_metadata: ToolMetadata,
    policy: DomainPolicy,
    runtime_context: ActionRuntimeContext | None = None,
) -> str:
    """Build the full text embedded for Action Gate similarity."""

    payload = {
        "immutable_goal_anchor": {
            "original_goal": goal_anchor.original_goal,
            "normalized_goal": goal_anchor.normalized_goal,
            "embedding_model": goal_anchor.embedding_model,
            "embedding_dimension": goal_anchor.embedding_dimension,
        },
        "proposed_tool_action": {
            "tool_name": action.tool_name,
            "tool_description": action.tool_description,
            "tool_arguments": thaw_mapping(action.tool_arguments),
        },
        "tool_metadata": {
            "risk_level": tool_metadata.risk_level,
            "allowed_domains": list(tool_metadata.allowed_domains),
            "required_permissions": list(tool_metadata.required_permissions),
            "side_effect_level": tool_metadata.side_effect_level.value,
        },
        "runtime_context": {
            "reasoning_summary": runtime_context.reasoning_summary if runtime_context else None,
            "previous_approved_action": runtime_context.previous_approved_action if runtime_context else None,
            "session_metadata": thaw_mapping(runtime_context.session_metadata) if runtime_context else {},
        },
        "policy": {
            "application_name": policy.application.name,
            "application_description": policy.application.description,
            "purpose": policy.purpose,
            "allowed_topics": policy.allowed_topics,
            "blocked_topics": policy.blocked_topics,
        },
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
