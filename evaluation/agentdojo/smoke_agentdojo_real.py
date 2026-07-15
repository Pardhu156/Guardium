"""Real AgentDojo suite smoke through AegisVault's tool executor.

This uses AgentDojo's built-in suites and evaluators. To avoid API keys/Qwen
while still testing the actual AgentDojo runtime interface, it replays each
selected task's ground-truth tool calls through AegisVault's AgentDojo
ToolsExecutor replacement.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite
from agentdojo.types import ChatAssistantMessage, text_content_block_from_string

from aegisvault.integrations.agentdojo import AgentDojoAdapterConfig
from aegisvault.integrations.agentdojo.pipeline import AegisVaultAgentDojoToolsExecutor
from aegisvault.policy import load_policy
from aegisvault.policy.models import Layer0Config, Layer0RequestConfig, Layer0ToolsConfig
from aegisvault.runtime.action_gate import ActionGateConfig
from aegisvault.runtime.goal_vault import GoalEmbedder, GoalVault, InMemoryGoalVaultBackend
from aegisvault.sentinel import SentinelConfig, SentinelDecision, SentinelDecisionLevel, SentinelExecutionState, SentinelMonitor


class SmokeEmbedder(GoalEmbedder):
    model_name = "agentdojo-real-smoke"
    dimension = 2

    def embed(self, text: str) -> tuple[float, ...]:
        lowered = text.lower()
        if "attacker" in lowered or "malicious" in lowered:
            return (0.0, 1.0)
        return (1.0, 0.0)


class AllowSentinel(SentinelMonitor):
    def __init__(self) -> None:
        self.calls = 0
        super().__init__(embedder=SmokeEmbedder(), config=SentinelConfig())

    def analyze(self, *, session_id: str, trusted_goal: str, execution: SentinelExecutionState) -> SentinelDecision:
        self.calls += 1
        return SentinelDecision(
            session_id=session_id,
            reasoning_similarity=None,
            intent_similarity=1.0,
            action_similarity=1.0,
            reasoning_drift=None,
            intent_drift=0.0,
            action_drift=0.0,
            fused_risk=0.0,
            ema_risk=0.0,
            confidence=0.9,
            decision=SentinelDecisionLevel.ALLOW,
            reason="smoke allow",
            metadata={"available_monitors": ["intent", "action"]},
        )


class GroundTruthThroughAegisVault(BasePipelineElement):
    def __init__(self, task: Any, executor: AegisVaultAgentDojoToolsExecutor, task_id: str) -> None:
        self.task = task
        self.executor = executor
        self.task_id = task_id

    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Any = EmptyEnv(),
        messages: Any = (),
        extra_args: dict[str, Any] | None = None,
    ) -> tuple[str, FunctionsRuntime, Any, Any, dict[str, Any]]:
        output_messages = list(messages)
        extra_args = {"task_id": self.task_id, **(extra_args or {})}
        for tool_call in self.task.ground_truth(env):
            output_messages.append(
                ChatAssistantMessage(
                    role="assistant",
                    tool_calls=[tool_call],
                    content=[text_content_block_from_string("")],
                )
            )
            query, runtime, env, output_messages, extra_args = self.executor.query(
                query,
                runtime,
                env,
                output_messages,
                extra_args,
            )
            if output_messages[-1].get("error"):
                break
        output_messages.append(
            ChatAssistantMessage(
                role="assistant",
                content=[text_content_block_from_string(self.task.GROUND_TRUTH_OUTPUT)],
                tool_calls=None,
            )
        )
        return query, runtime, env, output_messages, extra_args


def main() -> int:
    suites = ("workspace", "slack", "banking", "travel")
    rows = []
    for suite_name in suites:
        suite = get_suite("v1.2.2", suite_name)
        task_id = next(iter(suite.user_tasks))
        task = suite.get_user_task_by_id(task_id)
        policy = _policy_for_suite(suite_name, [tool.name for tool in suite.tools])
        embedder = SmokeEmbedder()
        sentinel = AllowSentinel()
        executor = AegisVaultAgentDojoToolsExecutor(
            policy=policy,
            config=AgentDojoAdapterConfig(suite_name=suite_name, domain=suite_name),
            goal_vault=GoalVault(backend=InMemoryGoalVaultBackend(), embedder=embedder),
            embedder=embedder,
            sentinel_monitor=sentinel,
            action_config=ActionGateConfig(high_similarity=0.95, low_similarity=0.2),
        )
        utility, security = suite.run_task_with_pipeline(
            GroundTruthThroughAegisVault(task, executor, task_id),
            task,
            injection_task=None,
            injections={},
        )
        rows.append(
            {
                "suite": suite_name,
                "task_id": task_id,
                "utility": utility,
                "security": security,
                "sentinel_calls": sentinel.calls,
            }
        )
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0 if all(row["utility"] and row["security"] for row in rows) else 1


def _policy_for_suite(suite_name: str, tool_names: list[str]):
    policy = load_policy(REPO_ROOT / "evaluation" / "agentdojo" / "policies" / f"{suite_name}.yaml")
    return policy.model_copy(
        update={
            "layer0": Layer0Config(
                enabled=True,
                fail_mode=policy.layer0.fail_mode,
                request=Layer0RequestConfig(
                    require_session_id=True,
                    require_domain=True,
                    allowed_domains=[suite_name],
                    max_characters=policy.layer0.request.max_characters,
                    max_bytes=policy.layer0.request.max_bytes,
                ),
                tools=Layer0ToolsConfig(
                    allowlist_mode=True,
                    allowed=tool_names,
                    denied=[],
                    max_argument_bytes=policy.layer0.tools.max_argument_bytes,
                ),
            )
        }
    )


if __name__ == "__main__":
    raise SystemExit(main())

