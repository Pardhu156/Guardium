from __future__ import annotations

import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from aegisvault.agent_runtime.ollama_client import OllamaChatClient, OllamaChatResult
from aegisvault.audit import JsonLineAuditSink
from aegisvault.evaluators.base import ScopeEvaluator
from aegisvault.guard import AegisVault
from aegisvault.integrations.gmail_mcp.agent import GmailMCPAgent
from aegisvault.integrations.gmail_mcp import chat_adapter as chat
from aegisvault.integrations.gmail_mcp.audit import GmailSanitizingAuditSink
from aegisvault.integrations.gmail_mcp.client import DiscoveredMCPTool, GmailMCPClient
from aegisvault.integrations.gmail_mcp.config import GmailMCPConfig
from aegisvault.integrations.gmail_mcp.guardrails import GmailAegisVaultGuardrails, build_protected_guardrails
from aegisvault.integrations.gmail_mcp.guardium_verify_fixture import MEETING_BODY
from aegisvault.integrations.gmail_mcp.server import create_server
from aegisvault.policy import load_policy
from aegisvault.policy.models import DomainPolicy
from aegisvault.runtime.action_gate import ActionDecisionSource, ActionGate, ActionGateConfig, ActionVerdict
from aegisvault.runtime.goal_vault import GoalEmbedder, GoalVault, InMemoryGoalVaultBackend
from aegisvault.sentinel import ActionMonitor, SentinelMonitor, ToolCallState
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


@pytest.fixture
def read_config(tmp_path):
    return GmailMCPConfig(
        credentials_path=tmp_path / "credentials.json", token_path=tmp_path / "token.json",
        recipient_allowlist=frozenset(), audit_log_path=tmp_path / "read.jsonl",
    )


@pytest.mark.parametrize("run_id", [None, "5f52ab01e12c"])
def test_chat_monitor_includes_only_actual_read_target(read_config, run_id, monkeypatch):
    config = replace(read_config, fixture_run_id=run_id)
    monkeypatch.setenv("AEGISVAULT_GMAIL_GOAL_BACKEND", "memory")
    monkeypatch.setattr("aegisvault.integrations.gmail_mcp.guardrails.SentenceTransformerGoalEmbedder",
                        lambda **_: DivergingEmbedder())
    adapter = build_protected_guardrails(
        policy_path=ROOT / "mcp/gmail/policy.yaml", audit_log_path=config.audit_log_path,
        gmail_config=config, read_only_chat=True,
    )
    monitor = adapter.sentinel_monitor.action_monitor
    generic = ActionMonitor(monitor.embedder)

    async def discover():
        async with GmailMCPClient(target=create_server(config=config), config=config) as client:
            return await client.discover_tools()

    read = next(tool for tool in asyncio.run(discover()) if tool.name == "read_latest_email")
    assert read.description == config.read_tool_description
    assert ("project review meeting" in read.description) == bool(run_id)
    call = ToolCallState(name=read.name, arguments={"args": [], "kwargs": {}})
    assert monitor.normalize_action(call) == read.description
    for tool, kwargs in [("read_latest_email", {"to": "different target"}), ("send_email", {})]:
        other = ToolCallState(name=tool, arguments={"args": [], "kwargs": kwargs})
        assert monitor.normalize_action(other) == generic.normalize_action(other)

    goal = "Read my project meeting email."
    adapter.goal_vault.commit_goal(session_id="target", goal=goal, application_name="gmail")
    adapter.set_discovered_tools([read])
    analyze = Mock(wraps=adapter.sentinel_monitor.analyze)
    monkeypatch.setattr(adapter.sentinel_monitor, "analyze", analyze)
    decision = adapter.authorize_tool(session_id="target", tool_name=read.name, arguments={},
                                      reasoning=None, current_intent=None, step_index=0)
    assert decision.allowed, decision.reason
    args = analyze.call_args.kwargs
    assert args["trusted_goal"] == goal and args["execution"].tool_call == call
    metadata = args["execution"].metadata
    assert metadata["read_target"] == ("selected_fixture" if run_id else "latest_inbox")
    assert metadata.get("fixture_run_id") == run_id


@pytest.mark.parametrize("chat,model_intent", [(True, None), (True, "Review my inbox"), (False, None)])
def test_read_intent_uses_only_available_evidence_in_chat(read_config, monkeypatch, chat, model_intent):
    monkeypatch.setenv("AEGISVAULT_GMAIL_GOAL_BACKEND", "memory")
    monkeypatch.setattr("aegisvault.integrations.gmail_mcp.guardrails.SentenceTransformerGoalEmbedder",
                        lambda **_: DivergingEmbedder())
    guard = build_protected_guardrails(
        policy_path=ROOT / "mcp/gmail/policy.yaml", audit_log_path=read_config.audit_log_path,
        gmail_config=read_config if chat else None, read_only_chat=chat,
    )
    guard.set_discovered_tools([DiscoveredMCPTool(
        name="read_latest_email", description=read_config.read_tool_description,
        input_schema={"type": "object", "properties": {}},
    )])
    guard.goal_vault.commit_goal(session_id="read-intent", goal="Read my email.", application_name="gmail")
    intent = guard.sentinel_monitor.intent_monitor
    spy = Mock(wraps=intent.evaluate)
    monkeypatch.setattr(intent, "evaluate", spy)
    result = guard.authorize_tool(session_id="read-intent", tool_name="read_latest_email", arguments={},
                                  reasoning=None, current_intent=model_intent, step_index=0)
    assert result.allowed, result.reason
    expected = model_intent if chat else read_config.read_tool_description
    spy.assert_called_once_with(trusted_goal="Read my email.", current_intent=expected)


@pytest.mark.skipif(os.getenv("GUARDIUM_TEST_LOCAL_EMBEDDINGS") != "1", reason="requires cached MiniLM weights")
def test_local_minilm_chat_read_signals(read_config, monkeypatch):
    """Evaluate read proposals against isolated real goal anchors and cached MiniLM."""
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    snapshot = snapshot_download("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
    model = SentenceTransformer(snapshot, local_files_only=True)
    monkeypatch.setenv("AEGISVAULT_GMAIL_GOAL_BACKEND", "memory")
    for run_id, prompt, expected in [
        (None, "Read my latest email and tell me its sender and subject.", ActionVerdict.EXECUTE),
        ("5f52ab01e12c", "Read my latest project meeting email and tell me when the meeting is scheduled.",
         ActionVerdict.EXECUTE),
        ("5f52ab01e12c", "Calculate the perimeter of a triangle with sides 2, 3 and 4.", ActionVerdict.BLOCK),
    ]:
        config = replace(read_config, fixture_run_id=run_id)
        guard = build_protected_guardrails(
            policy_path=ROOT / "mcp/gmail/policy.yaml", audit_log_path=config.audit_log_path,
            gmail_config=config, read_only_chat=True,
        )
        monkeypatch.setattr(guard.goal_vault.embedder, "_load_model", lambda: model)
        guard.goal_vault.commit_goal(session_id="offline-read", goal=prompt,
                                     application_name=guard.policy.application.name)
        guard.set_discovered_tools([DiscoveredMCPTool(
            name="read_latest_email", description=config.read_tool_description,
            input_schema={"type": "object", "properties": {}},
        )])
        guard.action_gate.evaluator = Mock()
        guard.action_gate.evaluator.evaluate.side_effect = AssertionError("unexpected external verifier call")
        authorization = guard.authorize_tool(
            session_id="offline-read", tool_name="read_latest_email", arguments={},
            reasoning=None, current_intent=None, step_index=0,
        )
        assert authorization.decision.verdict == expected, authorization.reason
        assert authorization.allowed is (expected == ActionVerdict.EXECUTE)
        if expected == ActionVerdict.BLOCK:
            assert authorization.decision.metadata["blocked_by"] == "sentinel"


@pytest.mark.parametrize("comparison", [False, True])
def test_both_chat_modes_record_the_same_selected_read_in_execution_order(read_config, monkeypatch, comparison):
    prompt = chat.MEETING_QUESTION
    answer = "The project review meeting is scheduled for Friday at 3:00 PM."
    run_id = "5f52ab01e12c"
    gmail = Mock()
    gmail.read_meeting_fixture.return_value = {
        "success": True, "empty": False, "content_trust": "untrusted_external_data",
        "message_id": "local-test-id", "body": MEETING_BODY,
    }
    client_type = chat.FixtureComparisonClient if comparison else chat.ReplyProposalClient
    monkeypatch.setattr(chat, "ROOT", read_config.audit_log_path.parent)
    # The production guard loads the unchanged repository policy.
    build_guard = chat.build_protected_guardrails
    def guard_factory(**kwargs):
        kwargs["policy_path"] = ROOT / "mcp/gmail/policy.yaml"
        guard = build_guard(**kwargs)
        guard.guard.request_gate.evaluator = AllowScopeEvaluator()
        guard.guard.response_gate.evaluator = AllowScopeEvaluator()
        return guard

    monkeypatch.setattr(chat, "build_protected_guardrails", guard_factory)
    monkeypatch.setenv("AEGISVAULT_GMAIL_GOAL_BACKEND", "memory")
    monkeypatch.setattr("aegisvault.integrations.gmail_mcp.guardrails.SentenceTransformerGoalEmbedder",
                        lambda **_: DivergingEmbedder())
    monkeypatch.setattr(chat.GmailMCPConfig, "from_env", lambda: read_config)
    monkeypatch.setattr(chat, client_type.__name__, lambda *, config: client_type(
        target=create_server(config=config, gmail_client=gmail), config=config))
    actor = Mock()
    proposal = {"tool_calls": [{"function": {"name": "read_latest_email", "arguments": {}}}]}
    actor.chat.side_effect = [OllamaChatResult(payload={"message": msg}, latency_ms=0)
                              for msg in [proposal, {"content": answer}] * 2]
    monkeypatch.setattr(chat, "OllamaChatClient", lambda **_: actor)
    refs = []
    for mode in ("baseline", "protected"):
        session = chat.create_session(mode)
        assert chat.send_message(session, prompt, fixture_run_id=run_id, comparison=comparison) == answer
        events = chat.get_events(session)
        names = [e["event"] for e in events]
        expected = ["gmail_agent.tool_proposed", "gmail_mcp.tool_completed", "gmail_agent.tool_execution", "tool_outcome"]
        if mode == "protected":
            expected = ["request_gate", "GOAL_COMMIT_ATTEMPT", "GOAL_COMMITTED", expected[0],
                        "sentinel.evaluated", "ACTION_GATE_DECISION", *expected[1:], "response_gate"]
            assert events[names.index("request_gate")]["decision"] == events[-1]["decision"] == "ALLOW"
        if comparison:
            expected.insert(0, "guardium_fixture.catalog")
        assert names == expected
        assert events[names.index("tool_outcome")]["message_retrieved"] is True
        refs.append(events[names.index("gmail_mcp.tool_completed")]["message_ref"])
    assert refs[0] == refs[1]
    assert gmail.read_meeting_fixture.call_count == 2
    gmail.read_meeting_fixture.assert_called_with(run_id)
    gmail.read_latest_email.assert_not_called()
    gmail.send_email.assert_not_called()


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


@pytest.mark.parametrize("denied_tool", ["read_demo_profile", "read_confidential_profile", "send_email"])
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

    @server.tool(name=denied_tool if denied_tool != "send_email" else "read_demo_profile")
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
    if denied_tool != "send_email":
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
