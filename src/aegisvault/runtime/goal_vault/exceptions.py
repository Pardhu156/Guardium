"""Goal Vault exceptions."""


class GoalVaultError(Exception):
    """Base class for Goal Vault errors."""


class GoalValidationError(GoalVaultError):
    """Raised when goal anchor input is invalid."""


class GoalAlreadyCommittedError(GoalVaultError):
    """Raised when a session already has an immutable goal anchor."""


class GoalNotFoundError(GoalVaultError):
    """Raised when a goal anchor is missing or expired."""


class GoalIntegrityError(GoalVaultError):
    """Raised when a stored anchor fails integrity verification."""


class GoalSerializationError(GoalVaultError):
    """Raised when anchor serialization or deserialization fails."""


class GoalEmbeddingError(GoalVaultError):
    """Raised when embedding generation or validation fails."""


class GoalBackendError(GoalVaultError):
    """Raised when the storage backend fails."""


class GoalBackendUnavailableError(GoalBackendError):
    """Raised when the configured backend is unavailable."""


class GoalTTLValidationError(GoalValidationError):
    """Raised when a TTL is invalid."""
