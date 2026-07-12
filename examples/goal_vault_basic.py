"""Minimal Stage 3.1 Goal Vault usage.

This example uses the in-memory backend so it can run without Redis. For a
Redis-backed vault, replace `InMemoryGoalVaultBackend()` with
`RedisGoalVaultBackend.from_env()` and install `.[runtime]`.
"""

from __future__ import annotations

from aegisvault.runtime.goal_vault import GoalEmbedder, GoalVault, InMemoryGoalVaultBackend


class DemoEmbedder(GoalEmbedder):
    """Tiny deterministic embedder for the example; production uses SentenceTransformer."""

    model_name = "demo-3d"
    dimension = 3

    def embed(self, text: str) -> list[float]:
        return [3.0, 4.0, 0.0]


def main() -> None:
    vault = GoalVault(
        backend=InMemoryGoalVaultBackend(),
        embedder=DemoEmbedder(),
        default_ttl_seconds=3600,
    )

    anchor = vault.commit_goal(
        session_id="example-session-1",
        application_name="ecommerce-support",
        goal="Track my delayed order and explain the refund status.",
        metadata={"example": True},
    )
    retrieved = vault.get_anchor("example-session-1")

    print("Committed session:", anchor.session_id)
    print("Normalized goal:", anchor.normalized_goal)
    print("Embedding model:", anchor.embedding_model)
    print("Embedding dimension:", anchor.embedding_dimension)
    print("Integrity verified:", vault.verify_anchor(retrieved))


if __name__ == "__main__":
    main()
