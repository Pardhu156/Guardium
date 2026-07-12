"""Redis-backed immutable Goal Vault backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from aegisvault.runtime.goal_vault.base import GoalVaultBackend
from aegisvault.runtime.goal_vault.exceptions import (
    GoalAlreadyCommittedError,
    GoalBackendError,
    GoalBackendUnavailableError,
    GoalNotFoundError,
    GoalSerializationError,
    GoalTTLValidationError,
)
from aegisvault.runtime.goal_vault.models import GoalAnchor, validate_session_id
from aegisvault.runtime.goal_vault.serialization import dumps_anchor, loads_anchor


@dataclass(frozen=True, slots=True)
class RedisGoalVaultConfig:
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    username: str | None = None
    password: str | None = None
    ssl: bool = False
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    default_ttl_seconds: int = 3600
    key_prefix: str = "aegisvault:goal_anchor:"


class RedisGoalVaultBackend(GoalVaultBackend):
    """Redis backend using atomic SET NX EX write-once commits."""

    backend_type = "redis"

    def __init__(self, config: RedisGoalVaultConfig | None = None, client: Any | None = None) -> None:
        self.config = config or RedisGoalVaultConfig()
        self._client = client or self._build_client(self.config)

    @classmethod
    def from_env(cls) -> "RedisGoalVaultBackend":
        config = RedisGoalVaultConfig(
            host=os.getenv("AEGISVAULT_REDIS_HOST", "localhost"),
            port=int(os.getenv("AEGISVAULT_REDIS_PORT", "6379")),
            db=int(os.getenv("AEGISVAULT_REDIS_DB", "0")),
            username=os.getenv("AEGISVAULT_REDIS_USERNAME") or None,
            password=os.getenv("AEGISVAULT_REDIS_PASSWORD") or None,
            ssl=_env_bool(os.getenv("AEGISVAULT_REDIS_SSL")),
            default_ttl_seconds=int(os.getenv("AEGISVAULT_REDIS_TTL_SECONDS", "3600")),
            key_prefix=os.getenv("AEGISVAULT_REDIS_KEY_PREFIX", "aegisvault:goal_anchor:"),
        )
        return cls(config=config)

    @property
    def default_ttl_seconds(self) -> int:
        return self.config.default_ttl_seconds

    def commit_anchor(self, anchor: GoalAnchor, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            raise GoalTTLValidationError("ttl_seconds must be positive")
        key = self._key(anchor.session_id)
        value = dumps_anchor(anchor)
        try:
            committed = self._client.set(key, value, nx=True, ex=ttl_seconds)
        except Exception as exc:
            raise GoalBackendUnavailableError("Redis goal anchor commit failed") from exc
        if not committed:
            raise GoalAlreadyCommittedError(f"goal already committed for session {anchor.session_id!r}")

    def get_anchor(self, session_id: str) -> GoalAnchor:
        validate_session_id(session_id)
        try:
            raw = self._client.get(self._key(session_id))
        except Exception as exc:
            raise GoalBackendUnavailableError("Redis goal anchor retrieval failed") from exc
        if raw is None:
            raise GoalNotFoundError(f"goal anchor not found for session {session_id!r}")
        try:
            return loads_anchor(raw)
        except GoalSerializationError:
            raise
        except Exception as exc:
            raise GoalSerializationError("failed to deserialize Redis goal anchor") from exc

    def delete_anchor(self, session_id: str) -> bool:
        validate_session_id(session_id)
        try:
            return bool(self._client.delete(self._key(session_id)))
        except Exception as exc:
            raise GoalBackendError("Redis goal anchor deletion failed") from exc

    def get_ttl(self, session_id: str) -> int | None:
        validate_session_id(session_id)
        try:
            ttl = int(self._client.ttl(self._key(session_id)))
        except Exception as exc:
            raise GoalBackendError("Redis TTL lookup failed") from exc
        return ttl if ttl >= 0 else None

    def _key(self, session_id: str) -> str:
        validate_session_id(session_id)
        return f"{self.config.key_prefix}{session_id}"

    def _build_client(self, config: RedisGoalVaultConfig) -> Any:
        try:
            import redis
        except Exception as exc:
            raise GoalBackendUnavailableError("redis package is required; install with `pip install -e \".[runtime]\"`") from exc
        return redis.Redis(
            host=config.host,
            port=config.port,
            db=config.db,
            username=config.username,
            password=config.password,
            ssl=config.ssl,
            socket_timeout=config.socket_timeout,
            socket_connect_timeout=config.socket_connect_timeout,
            decode_responses=False,
        )


def _env_bool(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}
