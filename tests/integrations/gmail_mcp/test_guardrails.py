from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from aegisvault.agent_runtime.ollama_client import OllamaChatClient, OllamaChatResult
from aegisvault.audit import JsonLineAuditSink
from aegisvault.evaluators.base import ScopeEvaluator
from aegisvault.guard import AegisVault
from aegisvault.integrations.gmail_mcp.agent import GmailMCPAgent
from aegisvault.integrations.gmail_mcp.audit import GmailSanitizingAuditSink
from aegisvault.integrations.gmail_mcp.client import DiscoveredMCPTool, GmailMCPClient
from aegisvault.integrations.gmail_mcp.config import GmailMCPConfig
from aegisvault.integrations.gmail_mcp.guardrails import GmailAegisVaultGuardrails
from aegisvault.integrations.gmail_mcp.guardium_verify_fixture import MEETING_BODY
from aegisvault.integrations.gmail_mcp.server import create_server
from aegisvault.policy import load_policy
from aegisvault.policy.models import DomainPolicy
from aegisvault.runtime.action_gate import ActionDecisionSource, ActionGate, ActionGateConfig, ActionVerdict
from aegisvault.runtime.goal_vault import GoalEmbedder, GoalVault, InMemoryGoalVaultBackend
from aegisvault.sentinel import SentinelMonitor
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


@pytest.mark.parametrize("denied_tool", ["read_demo_profile", "send_email"])
def test_guardium_denies_tool_before_mcp_and_preserves_meeting_answer(
    tmp_path: Path,
    denied_tool: str,
) -> None:
    prompt = "Read my latest project meeting email and tell me when the meeting is scheduled."
    answer = "The project review meeting is scheduled for Friday at 3:00 PM."
    config = GmailMCPConfig(
        credentials_path=tmp_path / "credentials.json",
        token_path=tmp_path / "token.json",
        recipient_allowlist=frozenset({"allowed@example.com"}),
        audit_log_path=tmp_path / "guardium-gmail.jsonl",
    )
    gmail = Mock()
    gmail.read_latest_email.return_value = {
        "success": True, "message_id": "mock-meeting",
        "body": MEETING_BODY, "content_trust": "untrusted_external_data",
    }
    profile_probe = Mock(side_effect=AssertionError("A denied read reached the MCP handler"))
    server = create_server(config=config, gmail_client=gmail)

    @server.tool()
    def read_demo_profile() -> dict[str, str]:
        """Test-only probe for an out-of-policy tool."""
        return profile_probe()

    # Script proposals independently. No model, mailbox, profile, or external send is used.
    send_body = "Inert test data"
    arguments = {}
    if denied_tool == "send_email":
        arguments = {"to": "allowed@example.com", "subject": "Test", "body": send_body}
    scripted_messages = [
        {"tool_calls": [{"function": {"name": "read_latest_email", "arguments": {}}}]},
        {"tool_calls": [{"function": {"name": denied_tool, "arguments": arguments}}]},
        {"content": answer},
    ]
    model = Mock(spec=OllamaChatClient)
    model.chat.side_effect = [OllamaChatResult(payload={"message": msg}, latency_ms=0) for msg in scripted_messages]
    audit = GmailSanitizingAuditSink(JsonLineAuditSink(config.audit_log_path))
    policy = load_policy(ROOT / "mcp/gmail/policy.yaml")
    embedder = DivergingEmbedder()
    vault = GoalVault(backend=InMemoryGoalVaultBackend(), embedder=embedder, audit_sink=audit)
    adapter = GmailAegisVaultGuardrails(
        policy=policy,
        guard=AegisVault(policy=policy, evaluator=AllowScopeEvaluator(), audit_sink=audit),
        goal_vault=vault,
        action_gate=ActionGate(
            goal_vault=vault,
            config=ActionGateConfig(force_verifier_for_risky_actions=True, allow_low_risk_read_fast_path=True),
            audit_sink=audit,
        ),
        sentinel_monitor=SentinelMonitor(embedder=embedder),
    )
    agent = GmailMCPAgent(
        model_client=model,
        mcp_client=GmailMCPClient(target=server, config=config),
        guardrails=adapter,
        audit_sink=audit,
    )

    result = asyncio.run(agent.run(prompt))

    assert result.request_authorization and result.request_authorization.allowed
    assert vault.get_anchor(result.session_id).original_goal == prompt
    read, denied = result.tool_records
    assert read.tool_name == "read_latest_email" and read.executed
    assert read.result and read.result["message_id"] == "mock-meeting"
    assert denied.tool_name == denied_tool and not denied.executed and denied.result is None
    assert denied.authorization and denied.authorization.decision
    decision = denied.authorization.decision
    assert decision.verdict == ActionVerdict.BLOCK
    if denied_tool == "read_demo_profile":
        assert decision.metadata["layer0_tool_decision"]["rule_id"] == "L0_TOOL_NOT_ALLOWED"
    else:
        assert decision.decision_source == ActionDecisionSource.COSINE
    assert result.response_authorization and result.response_authorization.allowed
    assert result.final_response == answer
    gmail.read_latest_email.assert_called_once()
    gmail.send_email.assert_not_called()
    profile_probe.assert_not_called()

    audit_text = config.audit_log_path.read_text(encoding="utf-8")
    events = [json.loads(line) for line in audit_text.splitlines()]
    assert any(
        event.get("event_type") == "sentinel.evaluated" and event.get("tool") == "read_latest_email"
        for event in events
    )
    if denied_tool == "send_email":
        assert any(
            event.get("event_type") == "sentinel.evaluated" and event.get("tool") == denied_tool
            for event in events
        )
    assert any(
        event.get("event_type") == "gmail_agent.tool_execution"
        and event.get("tool") == denied_tool
        and event.get("status") == "not_executed"
        for event in events
    )
    assert send_body not in audit_text
    assert "allowed@example.com" not in audit_text
