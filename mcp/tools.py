"""Safe MCP demo tools protected by AegisVault Action Gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from uuid import uuid4

from aegisvault.policy.models import (
    ApplicationConfig,
    DomainPolicy,
    EvaluatorConfig,
    GateConfig,
    GatesConfig,
    LowConfidenceAction,
)
from aegisvault.runtime.action_gate import (
    ActionDecisionSource,
    ActionGate,
    ActionGateConfig,
    ActionRuntimeContext,
    ActionVerdict,
    ProposedToolAction,
    SideEffectLevel,
    ToolMetadata,
)
from aegisvault.runtime.goal_vault import GoalEmbedder, GoalVault, InMemoryGoalVaultBackend


class DemoGoalEmbedder(GoalEmbedder):
    """Deterministic tiny embedder for the optional local MCP demo."""

    model_name = "mcp-demo-deterministic"
    dimension = 2

    def embed(self, text: str) -> tuple[float, float]:
        if "get_policy_summary" in text or "policy summary" in text.lower():
            return (1.0, 0.0)
        return (0.0, 1.0)


@dataclass(frozen=True)
class MCPToolResult:
    """Serializable result for an MCP tool invocation."""

    allowed: bool
    executed: bool
    tool_name: str
    verdict: str
    reason: str
    result: dict[str, Any] | None = None
    action_decision: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MCPToolGuard:
    """Small adapter that authorizes MCP tools with AegisVault Action Gate."""

    def __init__(self, *, policy: DomainPolicy, session_id: str | None = None) -> None:
        self.policy = policy
        self.session_id = session_id or f"mcp-demo-{uuid4().hex[:8]}"
        self.embedder = DemoGoalEmbedder()
        self.goal_vault = GoalVault(
            backend=InMemoryGoalVaultBackend(),
            embedder=self.embedder,
            default_ttl_seconds=600,
        )
        self.goal_vault.commit_goal(
            session_id=self.session_id,
            application_name=policy.application.name,
            goal="Allow read-only MCP clients to inspect an AegisVault policy summary.",
            metadata={"integration": "mcp-demo"},
        )
        self.action_gate = ActionGate(
            goal_vault=self.goal_vault,
            embedder=self.embedder,
            config=ActionGateConfig(
                high_similarity=0.80,
                low_similarity=0.20,
                minimum_llm_confidence=0.75,
                allow_low_risk_read_fast_path=True,
            ),
        )

    def authorize_and_execute(self, tool_name: str, arguments: dict[str, Any] | None = None) -> MCPToolResult:
        """Authorize a known MCP tool and execute only when Action Gate allows it."""

        arguments = dict(arguments or {})
        if tool_name != "get_policy_summary":
            return MCPToolResult(
                allowed=False,
                executed=False,
                tool_name=tool_name,
                verdict=ActionVerdict.BLOCK.value,
                reason=f"Unknown MCP tool {tool_name!r}; no execution occurred.",
            )

        action = ProposedToolAction(
            tool_name="get_policy_summary",
            tool_description="Return a read-only summary of the active AegisVault policy.",
            tool_arguments=arguments,
        )
        metadata = ToolMetadata(
            risk_level="low",
            allowed_domains=("mcp_demo",),
            required_permissions=("policy:read",),
            side_effect_level=SideEffectLevel.READ,
            requires_approval=False,
        )
        runtime_context = ActionRuntimeContext(
            reasoning_summary="MCP client requested a read-only policy summary.",
            current_intent="Inspect policy scope without mutating configuration.",
            session_metadata={"integration": "mcp-demo", "transport": "stdio-jsonrpc"},
        )
        decision = self.action_gate.evaluate_action(
            session_id=self.session_id,
            action=action,
            tool_metadata=metadata,
            policy=self.policy,
            runtime_context=runtime_context,
        )
        serialized_decision = _serialize_action_decision(decision)
        if decision.verdict != ActionVerdict.EXECUTE:
            return MCPToolResult(
                allowed=False,
                executed=False,
                tool_name=tool_name,
                verdict=decision.verdict.value,
                reason=decision.reason,
                action_decision=serialized_decision,
            )
        return MCPToolResult(
            allowed=True,
            executed=True,
            tool_name=tool_name,
            verdict=decision.verdict.value,
            reason=decision.reason,
            result=get_policy_summary(self.policy),
            action_decision=serialized_decision,
        )


def build_demo_guard() -> MCPToolGuard:
    """Create the isolated MCP demo guard."""

    return MCPToolGuard(policy=build_demo_policy())


def build_demo_policy() -> DomainPolicy:
    """Create a small domain-independent policy used only by the MCP demo."""

    request_gate = GateConfig(
        enabled=False,
        allow_threshold=0.8,
        block_threshold=0.8,
        low_confidence_action=LowConfidenceAction.CLARIFY,
    )
    response_gate = GateConfig(
        enabled=False,
        allow_threshold=0.8,
        block_threshold=0.8,
        low_confidence_action=LowConfidenceAction.BLOCK,
    )
    return DomainPolicy(
        version="1.0",
        application=ApplicationConfig(
            name="mcp-policy-summary-demo",
            description="Optional MCP demo exposing read-only AegisVault policy summaries.",
        ),
        purpose="Allow MCP clients to inspect a safe summary of the configured AegisVault policy.",
        allowed_topics=["policy summary", "guardrail configuration overview", "read-only inspection"],
        blocked_topics=["policy mutation", "secret disclosure", "tool execution outside the MCP demo"],
        gates=GatesConfig(request=request_gate, response=response_gate),
        evaluator=EvaluatorConfig(provider="ollama", model="llama3.2"),
    )


def get_policy_summary(policy: DomainPolicy) -> dict[str, Any]:
    """Return a sanitized, read-only policy summary."""

    return {
        "application": {
            "name": policy.application.name,
            "description": policy.application.description,
        },
        "purpose": policy.purpose,
        "allowed_topics": list(policy.allowed_topics),
        "blocked_topics": list(policy.blocked_topics),
        "request_gate_enabled": policy.gates.request.enabled,
        "response_gate_enabled": policy.gates.response.enabled,
    }


def _serialize_action_decision(decision: Any) -> dict[str, Any]:
    source = decision.decision_source
    if isinstance(source, ActionDecisionSource):
        source = source.value
    verdict = decision.verdict
    if isinstance(verdict, ActionVerdict):
        verdict = verdict.value
    return {
        "tool_name": decision.tool_name,
        "verdict": verdict,
        "decision_source": source,
        "goal_similarity": decision.goal_similarity,
        "confidence": decision.confidence,
        "reason": decision.reason,
        "latency_ms": decision.latency_ms,
        "ollama_called": decision.ollama_called,
        "goal_session": decision.goal_session,
    }

