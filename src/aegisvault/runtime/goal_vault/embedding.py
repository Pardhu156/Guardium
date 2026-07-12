"""Goal embedding generation and normalization."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from aegisvault.runtime.goal_vault.exceptions import GoalEmbeddingError


class GoalEmbedder(ABC):
    """Interface for goal embedding providers."""

    @abstractmethod
    def embed(self, text: str) -> Sequence[float]:
        """Return an embedding vector for text."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Embedding model identifier."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Expected embedding dimension."""


class SentenceTransformerGoalEmbedder(GoalEmbedder):
    """Sentence Transformers embedder for Goal Vault anchors."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", expected_dimension: int = 384) -> None:
        self._model_name = model_name
        self._dimension = expected_dimension
        self._model: Any | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> Sequence[float]:
        model = self._load_model()
        vector = model.encode(text, normalize_embeddings=False)
        values = [float(item) for item in vector]
        if len(values) != self.dimension:
            raise GoalEmbeddingError(
                f"embedding dimension mismatch for {self.model_name}: expected {self.dimension}, got {len(values)}"
            )
        return values

    def _load_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except Exception as exc:
                raise GoalEmbeddingError(
                    "sentence-transformers is required; install with `pip install -e \".[runtime]\"`"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model


def l2_normalize(values: Sequence[float], expected_dimension: int) -> tuple[float, ...]:
    """Validate and L2-normalize an embedding vector."""

    if len(values) != expected_dimension:
        raise GoalEmbeddingError(f"embedding dimension mismatch: expected {expected_dimension}, got {len(values)}")
    floats = tuple(float(value) for value in values)
    for value in floats:
        if not math.isfinite(value):
            raise GoalEmbeddingError("embedding contains NaN or infinity")
    norm = math.sqrt(sum(value * value for value in floats))
    if norm <= 0 or not math.isfinite(norm):
        raise GoalEmbeddingError("embedding norm must be positive and finite")
    normalized = tuple(value / norm for value in floats)
    normalized_norm = math.sqrt(sum(value * value for value in normalized))
    if not math.isclose(normalized_norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise GoalEmbeddingError("normalized embedding norm is not approximately 1.0")
    return normalized
