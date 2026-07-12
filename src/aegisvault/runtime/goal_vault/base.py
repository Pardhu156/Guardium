"""Goal Vault backend abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod

from aegisvault.runtime.goal_vault.models import GoalAnchor


class GoalVaultBackend(ABC):
    """Write-once storage backend for immutable goal anchors."""

    @abstractmethod
    def commit_anchor(self, anchor: GoalAnchor, ttl_seconds: int) -> None:
        """Atomically commit an anchor once for its session."""

    @abstractmethod
    def get_anchor(self, session_id: str) -> GoalAnchor:
        """Retrieve a committed anchor."""

    @abstractmethod
    def delete_anchor(self, session_id: str) -> bool:
        """Delete an anchor for administrative cleanup."""

    def get_ttl(self, session_id: str) -> int | None:
        """Return remaining TTL seconds when supported."""

        return None
