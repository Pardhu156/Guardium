"""Stage 3.2 Action Gate example."""

from __future__ import annotations

from aegisvault.audit import JsonLineAuditSink
from aegisvault.policy import load_policy
from aegisvault.runtime.action_gate import (
    ActionDecisionSource,
    ActionEvaluator,
    ActionGate,
    ActionGateDecision,
    ActionRuntimeContext,
    ActionVerdict,
    ProposedToolAction,
    SideEffectLevel,
    ToolMetadata,
)
from aegisvault.runtime.goal_vault import GoalAnchor, GoalEmbedder, GoalVault, InMemoryGoalVaultBackend


class DemoEmbedder(GoalEmbedder):
    """Deterministic vectors that demonstrate execute, block, and justify paths."""

    model_name = "demo-action-3d"
    dimension = 3

    def embed(self, text: str) -> list[float]:
        if "terminal.execute_python" in text:
            return [-1.0, 0.0, 0.0]
        if "gmail.send" in text:
            return [0.6, 0.8, 0.0]
        return [1.0, 0.0, 0.0]


class DemoActionEvaluator(ActionEvaluator):
    """Fake Ollama verifier for the example uncertainty band."""

    def evaluate(
        self,
        *,
        goal_anchor: GoalAnchor,
        action: ProposedToolAction,
        tool_metadata: ToolMetadata,
        policy,
        runtime_context: ActionRuntimeContext | None = None,
        goal_similarity: float | None = None,
    ) -> ActionGateDecision:
        return ActionGateDecision(
            tool_name=action.tool_name,
            tool_arguments=action.tool_arguments,
            goal_similarity=goal_similarity,
            decision_source=ActionDecisionSource.OLLAMA,
            verdict=ActionVerdict.JUSTIFY,
            confidence=0.86,
            reason="Sending email may be valid, but it has write side effects and needs explicit approval.",
            latency_ms=2.0,
            ollama_called=True,
            goal_session=goal_anchor.session_id,
        )


def read_unread_email(label: str) -> str:
    return f"Read messages with label={label}"


def execute_python(code: str) -> str:
    return f"Executed: {code}"


def send_email(to: str, body: str) -> str:
    return f"Sent email to {to}: {body}"


def main() -> None:
    policy = load_policy("evaluation/policies/email_assistant.yaml")
    embedder = DemoEmbedder()
    goal_vault = GoalVault(
        backend=InMemoryGoalVaultBackend(),
        embedder=embedder,
        default_ttl_seconds=3600,
    )
    goal_vault.commit_goal(
        session_id="action-example-session",
        application_name="email-assistant",
        goal="Summarize unread emails",
    )
    action_gate = ActionGate(
        goal_vault=goal_vault,
        embedder=embedder,
        evaluator=DemoActionEvaluator(),
        audit_sink=JsonLineAuditSink("logs/action_gate_example.jsonl"),
    )

    read_metadata = ToolMetadata(
        risk_level="low",
        allowed_domains=("email_assistant",),
        required_permissions=("gmail.read",),
        side_effect_level=SideEffectLevel.READ,
    )
    send_metadata = ToolMetadata(
        risk_level="medium",
        allowed_domains=("email_assistant",),
        required_permissions=("gmail.send",),
        side_effect_level=SideEffectLevel.WRITE,
    )
    system_metadata = ToolMetadata(
        risk_level="high",
        allowed_domains=(),
        required_permissions=("system.execute",),
        side_effect_level=SideEffectLevel.SYSTEM,
    )

    protected_read = action_gate.protect_tool(
        read_unread_email,
        tool_metadata=read_metadata,
        policy=policy,
        tool_name="gmail.read",
        tool_description="Read unread email messages",
    )
    protected_python = action_gate.protect_tool(
        execute_python,
        tool_metadata=system_metadata,
        policy=policy,
        tool_name="terminal.execute_python",
        tool_description="Execute Python code on the local system",
    )
    protected_send = action_gate.protect_tool(
        send_email,
        tool_metadata=send_metadata,
        policy=policy,
        tool_name="gmail.send",
        tool_description="Send an email message",
    )

    for label, result in [
        ("EXECUTE", protected_read("UNREAD", session_id="action-example-session")),
        ("BLOCK", protected_python("print(1)", session_id="action-example-session")),
        (
            "JUSTIFY",
            protected_send(
                "customer@example.com",
                "Here is your email summary.",
                session_id="action-example-session",
            ),
        ),
    ]:
        print(label)
        print("  executed:", result.executed)
        print("  verdict:", result.decision.verdict.value)
        print("  source:", result.decision.decision_source.value)
        print("  similarity:", result.decision.goal_similarity)
        print("  reason:", result.decision.reason)


if __name__ == "__main__":
    main()
