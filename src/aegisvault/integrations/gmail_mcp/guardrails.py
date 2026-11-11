"""Adapters connecting Gmail MCP proposals to existing AegisVault guards."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aegisvault.audit import AuditSink, JsonLineAuditSink
from aegisvault.guard import AegisVault
from aegisvault.policy import DomainPolicy, load_policy
from aegisvault.runtime.action_gate import (
    ActionGate,
    ActionGateConfig,
    ActionGateDecision,
    ActionRuntimeContext,
    SideEffectLevel,
    ToolMetadata,
)
from aegisvault.runtime.goal_vault import (
    GoalVault,
    InMemoryGoalVaultBackend,
    RedisGoalVaultBackend,
    SentenceTransformerGoalEmbedder,
)
from aegisvault.sentinel import SentinelConfig, SentinelMonitor
from aegisvault.types import EvaluationContext, GateDecision, Verdict

from aegisvault.integrations.gmail_mcp.audit import GmailSanitizingAuditSink
from aegisvault.integrations.gmail_mcp.client import DiscoveredMCPTool


@dataclass(frozen=True, slots=True)
class RequestAuthorization:
    allowed: bool
    reason: str
    decision: GateDecision | None = None


@dataclass(frozen=True, slots=True)
class ToolAuthorization:
    allowed: bool
    reason: str
    decision: ActionGateDecision | None = None


@dataclass(frozen=True, slots=True)
class ResponseAuthorization:
    allowed: bool
    response: str
    decision: GateDecision


class GmailAegisVaultGuardrails:
    """Run existing AegisVault controls around Gmail MCP agent activity."""

    def __init__(
        self,
        *,
        policy: DomainPolicy,
        guard: AegisVault,
        goal_vault: GoalVault,
        action_gate: ActionGate,
        sentinel_monitor: SentinelMonitor | None = None,
    ) -> None:
        self.policy = policy
        self.guard = guard
        self.goal_vault = goal_vault
        self.action_gate = action_gate
        self.sentinel_monitor = sentinel_monitor
        self._tools: dict[str, DiscoveredMCPTool] = {}

    def set_discovered_tools(self, tools: list[DiscoveredMCPTool]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def begin_request(self, *, session_id: str, prompt: str) -> RequestAuthorization:
        layer0 = self.guard.layer0_validator.validate_request(
            session_id=session_id,
            request_text=prompt,
            domain="email",
            metadata={"integration": "gmail_mcp"},
        )
        if not layer0.allowed:
            return RequestAuthorization(False, f"Layer 0 blocked request: {layer0.reason}")
        context = EvaluationContext(
            request_text=prompt,
            session_id=session_id,
            metadata={"domain": "email", "integration": "gmail_mcp"},
        )
        decision = self.guard.request_gate.evaluate(prompt, context)
        if decision.verdict != Verdict.ALLOW:
            return RequestAuthorization(False, decision.reason, decision)
        self.goal_vault.commit_goal(
            session_id=session_id,
            application_name=self.policy.application.name,
            goal=prompt,
            metadata={"integration": "gmail_mcp"},
        )
        return RequestAuthorization(True, decision.reason, decision)

    def authorize_tool(
        self,
        *,
        session_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        reasoning: str | None,
        current_intent: str | None,
        step_index: int,
    ) -> ToolAuthorization:
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolAuthorization(False, f"Unknown MCP tool {tool_name!r}")
        metadata = _tool_metadata(tool_name)
        catalog = {
            item.name: {"description": item.description, "parameters": item.input_schema}
            for item in self._tools.values()
        }

        def authorization_marker(**_: Any) -> dict[str, bool]:
            return {"authorized": True}

        protected = self.action_gate.protect_tool(
            authorization_marker,
            tool_metadata=metadata,
            policy=self.policy,
            tool_name=tool.name,
            tool_description=tool.description,
            layer0_validator=self.guard.layer0_validator,
            tool_catalog=catalog,
            sentinel_monitor=self.sentinel_monitor,
        )
        result = protected(
            session_id=session_id,
            runtime_context=ActionRuntimeContext(
                qwen_reasoning=reasoning,
                current_intent=current_intent or tool.description,
                step_index=step_index,
                session_metadata={"integration": "gmail_mcp", "transport": "stdio"},
            ),
            **arguments,
        )
        return ToolAuthorization(result.executed, result.decision.reason, result.decision)

    def check_response(self, *, session_id: str, prompt: str, response: str) -> ResponseAuthorization:
        decision = self.guard.response_gate.evaluate(
            response,
            EvaluationContext(
                request_text=prompt,
                response_text=response,
                session_id=session_id,
                metadata={"domain": "email", "integration": "gmail_mcp"},
            ),
        )
        if decision.verdict == Verdict.ALLOW:
            return ResponseAuthorization(True, response, decision)
        fallback = (
            self.policy.messages.response_replaced
            if decision.verdict == Verdict.REPLACE
            else self.policy.messages.response_blocked
        )
        return ResponseAuthorization(False, fallback, decision)


def build_protected_guardrails(
    *,
    policy_path: str | Path,
    audit_log_path: str | Path,
) -> GmailAegisVaultGuardrails:
    """Build the production protected path; Redis is the default Goal Vault backend."""

    policy = load_policy(policy_path)
    audit: AuditSink = GmailSanitizingAuditSink(JsonLineAuditSink(audit_log_path))
    embedder = SentenceTransformerGoalEmbedder(model_name="all-MiniLM-L6-v2", expected_dimension=384)
    backend_name = os.getenv("AEGISVAULT_GMAIL_GOAL_BACKEND", "redis").strip().lower()
    if backend_name == "redis":
        backend = RedisGoalVaultBackend.from_env()
    elif backend_name == "memory":
        backend = InMemoryGoalVaultBackend()
    else:
        raise ValueError("AEGISVAULT_GMAIL_GOAL_BACKEND must be 'redis' or 'memory'")
    goal_vault = GoalVault(backend=backend, embedder=embedder, audit_sink=audit)
    guard = AegisVault(policy=policy, audit_sink=audit)
    sentinel = SentinelMonitor(embedder=embedder, config=_sentinel_config(policy)) if policy.sentinel.enabled else None
    action_gate = ActionGate(
        goal_vault=goal_vault,
        embedder=embedder,
        config=ActionGateConfig(
            force_verifier_for_risky_actions=True,
            allow_low_risk_read_fast_path=True,
        ),
        audit_sink=audit,
    )
    return GmailAegisVaultGuardrails(
        policy=policy,
        guard=guard,
        goal_vault=goal_vault,
        action_gate=action_gate,
        sentinel_monitor=sentinel,
    )


def _tool_metadata(tool_name: str) -> ToolMetadata:
    if tool_name == "read_latest_email":
        return ToolMetadata(
            risk_level="low",
            allowed_domains=("email",),
            required_permissions=("gmail:read",),
            side_effect_level=SideEffectLevel.READ,
        )
    return ToolMetadata(
        risk_level="medium",
        allowed_domains=("email",),
        required_permissions=("gmail:send",),
        side_effect_level=SideEffectLevel.WRITE,
    )


def _sentinel_config(policy: DomainPolicy) -> SentinelConfig:
    return SentinelConfig(
        reasoning_weight=policy.sentinel.reasoning_weight,
        intent_weight=policy.sentinel.intent_weight,
        action_weight=policy.sentinel.action_weight,
        ema_alpha=policy.sentinel.ema_alpha,
        allow_threshold=policy.sentinel.allow_threshold,
        observe_threshold=policy.sentinel.observe_threshold,
        review_threshold=policy.sentinel.review_threshold,
    )
