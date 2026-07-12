"""Cosine utilities for Action Gate."""

from __future__ import annotations

import math
from collections.abc import Sequence

from aegisvault.runtime.action_gate.exceptions import ActionGateValidationError


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Compute cosine similarity for two vectors."""

    if len(left) != len(right):
        raise ActionGateValidationError("cosine vectors must have the same dimension")
    if not left:
        raise ActionGateValidationError("cosine vectors must not be empty")

    left_values = tuple(float(value) for value in left)
    right_values = tuple(float(value) for value in right)
    if any(not math.isfinite(value) for value in left_values + right_values):
        raise ActionGateValidationError("cosine vectors must contain only finite values")

    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm <= 0 or right_norm <= 0:
        raise ActionGateValidationError("cosine vectors must have positive norms")

    similarity = sum(a * b for a, b in zip(left_values, right_values, strict=True)) / (left_norm * right_norm)
    return max(-1.0, min(1.0, similarity))
