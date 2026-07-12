"""Immutable Goal Vault models."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, Mapping

from aegisvault.runtime.goal_vault.exceptions import GoalValidationError

SCHEMA_VERSION = "1.0"
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
SECRET_METADATA_KEY_PARTS = ("api_key", "apikey", "password", "secret", "token", "credential")


@dataclass(frozen=True, slots=True)
class GoalCommitRequest:
    """Input used to commit a user's original goal."""

    session_id: str
    application_name: str
    goal: str
    ttl_seconds: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_session_id(self.session_id)
        if not self.application_name.strip():
            raise GoalValidationError("application_name must not be empty")
        if not self.goal.strip():
            raise GoalValidationError("goal must not be empty")
        if self.ttl_seconds is not None and self.ttl_seconds <= 0:
            from aegisvault.runtime.goal_vault.exceptions import GoalTTLValidationError

            raise GoalTTLValidationError("ttl_seconds must be positive")
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class GoalAnchor:
    """Immutable semantic anchor for a committed original user goal."""

    session_id: str
    application_name: str
    original_goal: str
    normalized_goal: str
    goal_embedding: tuple[float, ...]
    embedding_model: str
    embedding_dimension: int
    integrity_hash: str
    created_at: datetime
    expires_at: datetime | None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise GoalValidationError(f"unsupported goal anchor schema_version {self.schema_version!r}")
        validate_session_id(self.session_id)
        if not self.application_name.strip():
            raise GoalValidationError("application_name must not be empty")
        if not self.original_goal.strip():
            raise GoalValidationError("original_goal must not be empty")
        if not self.normalized_goal.strip():
            raise GoalValidationError("normalized_goal must not be empty")
        if self.embedding_dimension <= 0:
            raise GoalValidationError("embedding_dimension must be positive")
        if len(self.goal_embedding) != self.embedding_dimension:
            raise GoalValidationError("goal_embedding length must equal embedding_dimension")
        embedding_norm = 0.0
        for value in self.goal_embedding:
            if not isinstance(value, int | float) or not math.isfinite(float(value)):
                raise GoalValidationError("goal_embedding values must be finite numbers")
            embedding_norm += float(value) * float(value)
        if not math.isclose(math.sqrt(embedding_norm), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise GoalValidationError("goal_embedding must be L2 normalized")
        if self.created_at.tzinfo is None:
            raise GoalValidationError("created_at must be timezone-aware")
        if self.expires_at is not None and self.expires_at.tzinfo is None:
            raise GoalValidationError("expires_at must be timezone-aware")
        object.__setattr__(self, "created_at", self.created_at.astimezone(UTC))
        object.__setattr__(self, "expires_at", self.expires_at.astimezone(UTC) if self.expires_at else None)
        object.__setattr__(self, "goal_embedding", tuple(float(v) for v in self.goal_embedding))
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))


def validate_session_id(session_id: str) -> None:
    if not SESSION_ID_RE.fullmatch(session_id):
        raise GoalValidationError(
            "session_id must be 1-128 chars and contain only letters, numbers, underscore, dot, colon, or hyphen"
        )


def freeze_metadata(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Recursively freeze metadata mappings to reduce accidental mutation."""

    frozen: dict[str, Any] = {}
    for key, raw_value in dict(value).items():
        key_text = str(key)
        _validate_metadata_key(key_text)
        frozen[key_text] = _freeze_value(raw_value)
    return MappingProxyType(frozen)


def thaw_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    return {str(k): _thaw_value(v) for k, v in dict(value).items()}


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return freeze_metadata(value)
    if isinstance(value, list | tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze_value(item) for item in value))
    return value


def _thaw_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return thaw_metadata(value)
    if isinstance(value, tuple):
        return [_thaw_value(item) for item in value]
    return value


def _validate_metadata_key(key: str) -> None:
    lowered = key.lower().replace("-", "_")
    if any(secret_part in lowered for secret_part in SECRET_METADATA_KEY_PARTS):
        raise GoalValidationError(f"metadata key {key!r} looks secret-bearing and must not be stored")
