"""Mock AgentDojo compatibility smoke test for AegisVault.

This script does not claim to run the real AgentDojo benchmark package. It
verifies the adapter contract with one Workspace, Slack, Banking, and Travel
case using AgentDojo-like tasks, agents, and tools.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aegisvault.integrations.agentdojo import AgentDojoAegisVaultAdapter, AgentDojoAdapterConfig, AgentDojoToolSpec
from aegisvault.policy import load_policy
from aegisvault.runtime.action_gate import ActionGateConfig, SideEffectLevel
from aegisvault.runtime.goal_vault import GoalEmbedder, GoalVault, InMemoryGoalVaultBackend


class SmokeEmbedder(GoalEmbedder):
    model_name = "agentdojo-smoke"
    dimension = 2

    def embed(self, text: str) -> tuple[float, ...]:
        lowered = text.lower()
        if "attacker" in lowered or "ignore" in lowered:
            return (0.0, 1.0)
        return (1.0, 0.0)


@dataclass(slots=True)
class MockAgent:
    tool_name: str
    arguments: dict[str, Any]

    def run(self, task: dict[str, Any], tool_executor: Any) -> dict[str, Any]:
        result = tool_executor.execute_tool(
            self.tool_name,
            self.arguments,
            reasoning=task["objective"],
            intent=task["objective"],
            step_index=1,
        )
        return {"tool_executed": result.executed, "result": result.result, "error": result.error}


def main() -> int:
    cases = [
        ("workspace", "search_workspace", {"query": "quarterly notes"}),
        ("slack", "search_slack", {"query": "deployment update"}),
        ("banking", "check_balance", {"account_id": "checking"}),
        ("travel", "search_flights", {"destination": "BOS"}),
    ]
    rows = []
    for suite, tool_name, arguments in cases:
        policy = load_policy(REPO_ROOT / "evaluation" / "agentdojo" / "policies" / f"{suite}.yaml")
        embedder = SmokeEmbedder()
        adapter = AgentDojoAegisVaultAdapter(
            policy=policy,
            config=AgentDojoAdapterConfig(suite_name=suite, domain=suite),
            tools=[
                AgentDojoToolSpec(
                    name=tool_name,
                    description=f"{suite} smoke tool",
                    parameters={"type": "object", "properties": {}, "required": []},
                    function=lambda **kwargs: {"ok": True, "kwargs": kwargs},
                    side_effect_level=SideEffectLevel.READ,
                )
            ],
            goal_vault=GoalVault(backend=InMemoryGoalVaultBackend(), embedder=embedder),
            embedder=embedder,
            action_config=ActionGateConfig(high_similarity=0.95, low_similarity=0.2),
        )
        result = adapter.run_task(
            {"id": f"{suite}_smoke_001", "objective": f"Complete {suite} benchmark task", "metadata": {}},
            MockAgent(tool_name=tool_name, arguments=arguments),
        )
        rows.append(
            {
                "suite": suite,
                "request_allowed": result.request_allowed,
                "goal_initialized": result.goal_initialized,
                "agent_executed": result.agent_executed,
                "tool_count": len(result.tool_results),
                "tool_executed": result.tool_results[0].executed if result.tool_results else False,
                "stopped_by": result.stopped_by,
            }
        )
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0 if all(row["request_allowed"] and row["goal_initialized"] and row["tool_count"] == 1 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

