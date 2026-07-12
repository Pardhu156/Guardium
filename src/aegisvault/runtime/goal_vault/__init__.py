"""Immutable Goal Vault runtime API."""

from aegisvault.runtime.goal_vault.base import GoalVaultBackend
from aegisvault.runtime.goal_vault.embedding import GoalEmbedder, SentenceTransformerGoalEmbedder, l2_normalize
from aegisvault.runtime.goal_vault.exceptions import (
    GoalAlreadyCommittedError,
    GoalBackendError,
    GoalBackendUnavailableError,
    GoalEmbeddingError,
    GoalIntegrityError,
    GoalNotFoundError,
    GoalSerializationError,
    GoalTTLValidationError,
    GoalValidationError,
    GoalVaultError,
)
from aegisvault.runtime.goal_vault.hashing import compute_integrity_hash, verify_integrity_hash
from aegisvault.runtime.goal_vault.memory_backend import InMemoryGoalVaultBackend
from aegisvault.runtime.goal_vault.models import GoalAnchor, GoalCommitRequest
from aegisvault.runtime.goal_vault.redis_backend import RedisGoalVaultBackend, RedisGoalVaultConfig
from aegisvault.runtime.goal_vault.serialization import dumps_anchor, loads_anchor
from aegisvault.runtime.goal_vault.service import GoalVault, normalize_goal

__all__ = [
    "GoalAlreadyCommittedError",
    "GoalAnchor",
    "GoalBackendError",
    "GoalBackendUnavailableError",
    "GoalCommitRequest",
    "GoalEmbedder",
    "GoalEmbeddingError",
    "GoalIntegrityError",
    "GoalNotFoundError",
    "GoalSerializationError",
    "GoalTTLValidationError",
    "GoalValidationError",
    "GoalVault",
    "GoalVaultBackend",
    "GoalVaultError",
    "InMemoryGoalVaultBackend",
    "RedisGoalVaultBackend",
    "RedisGoalVaultConfig",
    "SentenceTransformerGoalEmbedder",
    "compute_integrity_hash",
    "dumps_anchor",
    "l2_normalize",
    "loads_anchor",
    "normalize_goal",
    "verify_integrity_hash",
]
