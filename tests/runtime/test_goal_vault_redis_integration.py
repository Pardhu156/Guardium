from __future__ import annotations

import os
from uuid import uuid4

import pytest

from aegisvault.runtime.goal_vault import (
    GoalEmbedder,
    GoalVault,
    RedisGoalVaultBackend,
)


class FakeEmbedder(GoalEmbedder):
    model_name = "fake-3d"
    dimension = 3

    def embed(self, text: str) -> list[float]:
        return [3.0, 4.0, 0.0]


pytestmark = pytest.mark.redis_integration


@pytest.mark.skipif(os.getenv("AEGISVAULT_RUN_REDIS_TESTS") != "1", reason="set AEGISVAULT_RUN_REDIS_TESTS=1")
def test_live_redis_commit_retrieve_and_delete() -> None:
    backend = RedisGoalVaultBackend.from_env()
    vault = GoalVault(backend=backend, embedder=FakeEmbedder(), default_ttl_seconds=30)
    session_id = f"redis-it-{uuid4().hex}"

    anchor = vault.commit_goal(
        session_id=session_id,
        application_name="integration-app",
        goal="Remember the original support goal",
    )

    try:
        assert vault.get_anchor(session_id) == anchor
        assert vault.get_ttl(session_id) is not None
    finally:
        vault.delete_anchor(session_id)
