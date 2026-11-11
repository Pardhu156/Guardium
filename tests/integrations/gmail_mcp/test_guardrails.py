from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from aegisvault.evaluators.base import ScopeEvaluator
from aegisvault.guard import AegisVault
from aegisvault.integrations.gmail_mcp.client import DiscoveredMCPTool, GmailMCPClient
from aegisvault.integrations.gmail_mcp.config import GmailMCPConfig
from aegisvault.integrations.gmail_mcp.guardrails import GmailAegisVaultGuardrails
from aegisvault.integrations.gmail_mcp.server import create_server
from aegisvault.policy import load_policy
from aegisvault.policy.models import DomainPolicy
from aegisvault.runtime.action_gate import ActionGate, ActionGateConfig
from aegisvault.runtime.goal_vault import GoalEmbedder, GoalVault, InMemoryGoalVaultBackend
from aegisvault.types import EvaluationContext, GateDecision, GateType, Verdict


ROOT = Path(__file__).resolve().parents[3]


class AllowScopeEvaluator(ScopeEvaluator):
    @property
    def name(self) -> str:
        return "allow-test"

    def evaluate(
        self,
        text: str,
        policy: DomainPolicy,
        gate: GateType,
        context: EvaluationContext | None = None,
    ) -> GateDecision:
        return GateDecision(
            verdict=Verdict.ALLOW,
            confidence=1.0,
            reason="Allowed by test evaluator.",
            gate=gate,
            evaluator=self.name,
            latency_ms=0.0,
        )


class DivergingEmbedder(GoalEmbedder):
    model_name = "diverging-test"
    dimension = 2

    def embed(self, text: str) -> tuple[float, float]:
        if "proposed_tool_action" in text:
            return (0.0, 1.0)
        return (1.0, 0.0)


def test_real_guardrail_adapter_blocks_send_proposal_before_execution() -> None:
    policy = load_policy(ROOT / "mcp/gmail/policy.yaml")
    policy = policy.model_copy(update={"sentinel": policy.sentinel.model_copy(update={"enabled": False})})
    embedder = DivergingEmbedder()
    vault = GoalVault(backend=InMemoryGoalVaultBackend(), embedder=embedder)
    guard = AegisVault(policy=policy, evaluator=AllowScopeEvaluator())
    action_gate = ActionGate(
        goal_vault=vault,
        embedder=embedder,
        config=ActionGateConfig(high_similarity=0.8, low_similarity=0.2),
    )
    adapter = GmailAegisVaultGuardrails(
        policy=policy,
        guard=guard,
        goal_vault=vault,
        action_gate=action_gate,
    )
    adapter.set_discovered_tools(
        [
            DiscoveredMCPTool(
                name="send_email",
                description="Send one plain-text email.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "to": {"type": "string"},
                        "subject": {"type": "string"},
                        "body": {"type": "string"},
                    },
                    "required": ["to", "subject", "body"],
                },
            )
        ]
    )
    request = adapter.begin_request(session_id="guardrail-session", prompt="Send a status email.")

    authorization = adapter.authorize_tool(
        session_id="guardrail-session",
        tool_name="send_email",
        arguments={"to": "allowed@example.com", "subject": "Status", "body": "Ready"},
        reasoning=None,
        current_intent="Send an unrelated message.",
        step_index=0,
    )

    assert request.allowed is True
    assert authorization.allowed is False
    assert authorization.decision is not None
    assert authorization.decision.verdict.value == "BLOCK"


def test_advertised_read_tool_description_allows_protected_read(tmp_path: Path) -> None:
    config = GmailMCPConfig(
        credentials_path=tmp_path / "credentials.json",
        token_path=tmp_path / "token.json",
        recipient_allowlist=frozenset(),
        audit_log_path=tmp_path / "audit.jsonl",
    )

    async def discover() -> list[DiscoveredMCPTool]:
        async with GmailMCPClient(target=create_server(config=config), config=config) as client:
            return await client.discover_tools()

    policy = load_policy(ROOT / "mcp/gmail/policy.yaml")
    embedder = DivergingEmbedder()
    vault = GoalVault(backend=InMemoryGoalVaultBackend(), embedder=embedder)
    adapter = GmailAegisVaultGuardrails(
        policy=policy,
        guard=AegisVault(policy=policy, evaluator=AllowScopeEvaluator()),
        goal_vault=vault,
        action_gate=ActionGate(
            goal_vault=vault,
            config=ActionGateConfig(allow_low_risk_read_fast_path=True),
        ),
    )
    adapter.set_discovered_tools(asyncio.run(discover()))
    assert adapter.begin_request(session_id="read-session", prompt="Read the latest inbox email.").allowed

    authorization = adapter.authorize_tool(
        session_id="read-session",
        tool_name="read_latest_email",
        arguments={},
        reasoning=None,
        current_intent=None,
        step_index=0,
    )

    assert authorization.allowed, authorization.reason
