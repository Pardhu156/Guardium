"""Action Gate service."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from functools import wraps
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from aegisvault.audit import AuditSink, NullAuditSink
from aegisvault.layer0 import Layer0Validator
from aegisvault.policy.models import DomainPolicy
from aegisvault.policy.models import SentinelFailMode
from aegisvault.runtime.action_gate.cosine import cosine_similarity
from aegisvault.runtime.action_gate.evaluators import ActionEvaluator, OllamaActionEvaluator
from aegisvault.runtime.action_gate.exceptions import ActionEvaluatorError
from aegisvault.runtime.action_gate.models import (
    ActionDecisionSource,
    ActionGateConfig,
    ActionGateDecision,
    ActionRuntimeContext,
    ActionVerdict,
    ProposedToolAction,
    SideEffectLevel,
    ToolExecutionResult,
    ToolMetadata,
    thaw_mapping,
)
from aegisvault.runtime.goal_vault import GoalEmbedder, GoalVault, l2_normalize

if TYPE_CHECKING:
    from aegisvault.sentinel import SentinelDecision, SentinelMonitor


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
            policy_override = _forced_metadata_decision(
                session_id=goal_anchor.session_id,
                action=action,
                tool_metadata=tool_metadata,
                similarity=similarity,
                started=started,
            )
            if policy_override is not None:
                self._audit(policy_override)
                return policy_override
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
        layer0_validator: Layer0Validator | None = None,
        tool_catalog: dict[str, Any] | None = None,
        sentinel_monitor: SentinelMonitor | None = None,
    ) -> Callable[..., ToolExecutionResult]:
        """Wrap a synchronous Python callable so Action Gate runs before execution."""

        resolved_name = tool_name or getattr(tool, "__name__", tool.__class__.__name__)
        resolved_description = tool_description or getattr(tool, "__doc__", None) or resolved_name
        resolved_sentinel = sentinel_monitor
        if resolved_sentinel is None and policy.sentinel.enabled:
            from aegisvault.sentinel import SentinelMonitor

            resolved_sentinel = SentinelMonitor(embedder=self.embedder, config=_sentinel_config_from_policy(policy))

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
            if layer0_validator is not None and layer0_validator.enabled:
                layer0_decision = layer0_validator.validate_tool_call(
                    session_id=session_id,
                    tool_name=resolved_name,
                    arguments=action.tool_arguments,
                    domain=policy.application.name,
                    metadata=thaw_mapping(runtime_context.session_metadata) if runtime_context else {},
                    tool_catalog=tool_catalog,
                )
                if not layer0_decision.allowed:
                    decision = ActionGateDecision(
                        tool_name=action.tool_name,
                        tool_arguments=action.tool_arguments,
                        goal_similarity=None,
                        decision_source=ActionDecisionSource.FALLBACK,
                        verdict=ActionVerdict.BLOCK,
                        confidence=None,
                        reason=f"Layer 0 blocked tool call: {layer0_decision.reason}",
                        latency_ms=0.0,
                        ollama_called=False,
                        goal_session=session_id,
                        metadata={
                            "layer0_tool_decision": {
                                "decision": layer0_decision.decision.value,
                                "rule_id": layer0_decision.rule_id,
                                "matched_rule_ids": [rule.rule_id for rule in layer0_decision.matched_rules],
                            }
                        },
                    )
                    self._audit(decision)
                    return ToolExecutionResult(decision=decision, executed=False)
            sentinel_decision: SentinelDecision | None = None
            if resolved_sentinel is not None and policy.sentinel.enabled and policy.sentinel.runtime.evaluate_before_every_tool:
                try:
                    from aegisvault.sentinel import SentinelDecisionLevel, SentinelExecutionState, ToolCallState

                    goal_anchor = self.goal_vault.get_anchor(session_id)
                    execution = SentinelExecutionState(
                        session_id=session_id,
                        reasoning=(runtime_context.qwen_reasoning if runtime_context else None)
                        if policy.sentinel.signals.reasoning
                        else None,
                        current_intent=((runtime_context.current_intent if runtime_context else None) or resolved_description)
                        if policy.sentinel.signals.intent
                        else None,
                        tool_call=ToolCallState(name=resolved_name, arguments=action.tool_arguments)
                        if policy.sentinel.signals.action
                        else None,
                        step_index=runtime_context.step_index if runtime_context else None,
                    )
                    sentinel_decision = resolved_sentinel.analyze(
                        session_id=session_id,
                        trusted_goal=goal_anchor.original_goal,
                        execution=execution,
                    )
                    self._audit_sentinel(
                        "sentinel.evaluated",
                        session_id=session_id,
                        tool_name=resolved_name,
                        sentinel_decision=sentinel_decision,
                        step_index=runtime_context.step_index if runtime_context else None,
                    )
                    if policy.sentinel.runtime.audit_missing_signals:
                        missing = _missing_sentinel_signals(sentinel_decision)
                        if missing:
                            self._audit_sentinel(
                                "sentinel.signal_missing",
                                session_id=session_id,
                                tool_name=resolved_name,
                                sentinel_decision=sentinel_decision,
                                step_index=runtime_context.step_index if runtime_context else None,
                                metadata={"missing_signals": missing},
                            )
                    if (
                        sentinel_decision.decision == SentinelDecisionLevel.BLOCK
                        and policy.sentinel.enforcement.block_on_sentinel_block
                    ):
                        decision = _sentinel_block_decision(session_id, action, sentinel_decision)
                        self._audit_sentinel(
                            "sentinel.blocked",
                            session_id=session_id,
                            tool_name=resolved_name,
                            sentinel_decision=sentinel_decision,
                            step_index=runtime_context.step_index if runtime_context else None,
                        )
                        self._audit(decision)
                        return ToolExecutionResult(decision=decision, executed=False)
                except Exception as exc:
                    self._audit_sentinel_error(session_id=session_id, tool_name=resolved_name, exc=exc)
                    if policy.sentinel.fail_mode == SentinelFailMode.CLOSED:
                        decision = _sentinel_error_decision(session_id, action, exc)
                        self._audit(decision)
                        return ToolExecutionResult(decision=decision, executed=False)

            runtime_context_for_gate = _context_with_sentinel(runtime_context, sentinel_decision)
            decision = self.evaluate_action(
                session_id=session_id,
                action=action,
                tool_metadata=tool_metadata,
                policy=policy,
                runtime_context=runtime_context_for_gate,
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

    def _audit_sentinel(
        self,
        event_type: str,
        *,
        session_id: str,
        tool_name: str,
        sentinel_decision: SentinelDecision,
        step_index: int | None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        available = set(sentinel_decision.metadata.get("available_monitors", []))
        event = {
            "event_id": str(uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "session_id": session_id,
            "step_index": step_index,
            "tool": tool_name,
            "reasoning_signal_available": "reasoning" in available,
            "intent_signal_available": "intent" in available,
            "action_signal_available": "action" in available,
            "reasoning_drift": sentinel_decision.reasoning_drift,
            "intent_drift": sentinel_decision.intent_drift,
            "action_drift": sentinel_decision.action_drift,
            "fused_risk": sentinel_decision.fused_risk,
            "ema_risk": sentinel_decision.ema_risk,
            "confidence": sentinel_decision.confidence,
            "sentinel_decision": sentinel_decision.decision.value,
            "metadata": metadata or {},
        }
        try:
            self.audit_sink.record(event)
        except Exception:
            return None

    def _audit_sentinel_error(self, *, session_id: str, tool_name: str, exc: Exception) -> None:
        event = {
            "event_id": str(uuid4()),
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": "sentinel.error",
            "session_id": session_id,
            "tool": tool_name,
            "error_type": exc.__class__.__name__,
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
            "requires_approval": tool_metadata.requires_approval,
        },
        "runtime_context": {
            "reasoning_summary": runtime_context.reasoning_summary if runtime_context else None,
            "previous_approved_action": runtime_context.previous_approved_action if runtime_context else None,
            "current_intent": runtime_context.current_intent if runtime_context else None,
            "step_index": runtime_context.step_index if runtime_context else None,
            "sentinel_decision": _sentinel_summary(runtime_context.sentinel_decision if runtime_context else None),
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


def _sentinel_config_from_policy(policy: DomainPolicy) -> SentinelConfig:
    from aegisvault.sentinel import SentinelConfig

    return SentinelConfig(
        reasoning_weight=policy.sentinel.reasoning_weight,
        intent_weight=policy.sentinel.intent_weight,
        action_weight=policy.sentinel.action_weight,
        ema_alpha=policy.sentinel.ema_alpha,
        allow_threshold=policy.sentinel.allow_threshold,
        observe_threshold=policy.sentinel.observe_threshold,
        review_threshold=policy.sentinel.review_threshold,
    )


def _context_with_sentinel(
    runtime_context: ActionRuntimeContext | None,
    sentinel_decision: SentinelDecision | None,
) -> ActionRuntimeContext | None:
    if sentinel_decision is None:
        return runtime_context
    metadata = thaw_mapping(runtime_context.session_metadata) if runtime_context else {}
    metadata["sentinel"] = _sentinel_summary(sentinel_decision)
    return ActionRuntimeContext(
        reasoning_summary=runtime_context.reasoning_summary if runtime_context else None,
        previous_approved_action=runtime_context.previous_approved_action if runtime_context else None,
        qwen_reasoning=runtime_context.qwen_reasoning if runtime_context else None,
        current_intent=runtime_context.current_intent if runtime_context else None,
        step_index=runtime_context.step_index if runtime_context else None,
        sentinel_decision=sentinel_decision,
        session_metadata=metadata,
    )


def _sentinel_summary(decision: SentinelDecision | None) -> dict[str, Any] | None:
    if decision is None:
        return None
    available = decision.metadata.get("available_monitors", [])
    return {
        "decision": decision.decision.value,
        "fused_risk": decision.fused_risk,
        "ema_risk": decision.ema_risk,
        "confidence": decision.confidence,
        "available_signals": list(available) if isinstance(available, list | tuple) else [],
        "reason": decision.reason,
    }


def _missing_sentinel_signals(decision: SentinelDecision) -> list[str]:
    available = set(decision.metadata.get("available_monitors", []))
    return [name for name in ("reasoning", "intent", "action") if name not in available]


def _sentinel_block_decision(session_id: str, action: ProposedToolAction, sentinel_decision: SentinelDecision) -> ActionGateDecision:
    return ActionGateDecision(
        tool_name=action.tool_name,
        tool_arguments=action.tool_arguments,
        goal_similarity=None,
        decision_source=ActionDecisionSource.FALLBACK,
        verdict=ActionVerdict.BLOCK,
        confidence=sentinel_decision.confidence,
        reason=f"Sentinel blocked automatic tool execution: {sentinel_decision.reason}",
        latency_ms=0.0,
        ollama_called=False,
        goal_session=session_id,
        metadata={"sentinel": _sentinel_summary(sentinel_decision), "blocked_by": "sentinel"},
    )


def _sentinel_error_decision(session_id: str, action: ProposedToolAction, exc: Exception) -> ActionGateDecision:
    return ActionGateDecision(
        tool_name=action.tool_name,
        tool_arguments=action.tool_arguments,
        goal_similarity=None,
        decision_source=ActionDecisionSource.FALLBACK,
        verdict=ActionVerdict.BLOCK,
        confidence=None,
        reason=f"Sentinel failed closed before tool execution: {exc.__class__.__name__}",
        latency_ms=0.0,
        ollama_called=False,
        goal_session=session_id,
        metadata={"blocked_by": "sentinel_error", "error_type": exc.__class__.__name__},
    )


def _forced_metadata_decision(
    *,
    session_id: str,
    action: ProposedToolAction,
    tool_metadata: ToolMetadata,
    similarity: float,
    started: float,
) -> ActionGateDecision | None:
    """Apply generic safety metadata before high-similarity auto-execute."""

    arguments_text = json.dumps(thaw_mapping(action.tool_arguments), ensure_ascii=False, sort_keys=True).lower()
    action_text = f"{action.tool_name} {action.tool_description} {arguments_text}".lower()
    dangerous_signals = (
        "rm -rf",
        "attacker",
        "api_key",
        "secret",
        "password",
        "delete all",
        "dump",
        "export all",
        "external",
        "upload_document",
        "investment",
        "medical",
        "legal.give",
        "crypto.buy",
    )
    if tool_metadata.side_effect_level == SideEffectLevel.SYSTEM or any(signal in action_text for signal in dangerous_signals):
        return ActionGateDecision(
            tool_name=action.tool_name,
            tool_arguments=action.tool_arguments,
            goal_similarity=similarity,
            decision_source=ActionDecisionSource.FALLBACK,
            verdict=ActionVerdict.BLOCK,
            confidence=1.0,
            reason="Generic safety metadata or dangerous arguments require blocking before auto-execution.",
            latency_ms=(time.perf_counter() - started) * 1000,
            ollama_called=False,
            goal_session=session_id,
            metadata={"policy_forced": True, "override": "dangerous_action"},
        )
    if tool_metadata.requires_approval or tool_metadata.risk_level.lower() in {"high", "critical"}:
        return ActionGateDecision(
            tool_name=action.tool_name,
            tool_arguments=action.tool_arguments,
            goal_similarity=similarity,
            decision_source=ActionDecisionSource.FALLBACK,
            verdict=ActionVerdict.JUSTIFY,
            confidence=similarity,
            reason="Tool metadata requires explicit approval before execution.",
            latency_ms=(time.perf_counter() - started) * 1000,
            ollama_called=False,
            goal_session=session_id,
            metadata={"policy_forced": True, "override": "requires_approval"},
        )
    return None
