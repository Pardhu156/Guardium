"""Thread-safe in-memory Goal Vault backend."""

from __future__ import annotations

import threading
import time

from aegisvault.runtime.goal_vault.base import GoalVaultBackend
from aegisvault.runtime.goal_vault.exceptions import GoalAlreadyCommittedError, GoalNotFoundError, GoalTTLValidationError
from aegisvault.runtime.goal_vault.models import GoalAnchor, validate_session_id


class InMemoryGoalVaultBackend(GoalVaultBackend):
    """In-memory backend for tests and single-process local development."""

    backend_type = "memory"

    def __init__(self) -> None:
        self._anchors: dict[str, tuple[GoalAnchor, float]] = {}
        self._lock = threading.Lock()

    def commit_anchor(self, anchor: GoalAnchor, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            raise GoalTTLValidationError("ttl_seconds must be positive")
        expires_at = time.time() + ttl_seconds
        with self._lock:
            self._purge_expired_locked(anchor.session_id)
            if anchor.session_id in self._anchors:
                raise GoalAlreadyCommittedError(f"goal already committed for session {anchor.session_id!r}")
            self._anchors[anchor.session_id] = (anchor, expires_at)

    def get_anchor(self, session_id: str) -> GoalAnchor:
        validate_session_id(session_id)
        with self._lock:
            self._purge_expired_locked(session_id)
            try:
                return self._anchors[session_id][0]
            except KeyError as exc:
                raise GoalNotFoundError(f"goal anchor not found for session {session_id!r}") from exc

    def delete_anchor(self, session_id: str) -> bool:
        validate_session_id(session_id)
        with self._lock:
            return self._anchors.pop(session_id, None) is not None

    def get_ttl(self, session_id: str) -> int | None:
        validate_session_id(session_id)
        with self._lock:
            self._purge_expired_locked(session_id)
            item = self._anchors.get(session_id)
            if item is None:
                return None
            return max(0, int(item[1] - time.time()))

    def _purge_expired_locked(self, session_id: str) -> None:
        item = self._anchors.get(session_id)
        if item is not None and item[1] <= time.time():
            self._anchors.pop(session_id, None)
