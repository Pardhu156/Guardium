"""Integrity hashing for Goal Vault anchors."""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import replace

from aegisvault.runtime.goal_vault.models import GoalAnchor
from aegisvault.runtime.goal_vault.serialization import canonical_hash_payload


def compute_integrity_hash(anchor: GoalAnchor) -> str:
    """Compute deterministic SHA-256 hash over security-relevant immutable fields."""

    placeholder = replace(anchor, integrity_hash="")
    canonical = canonical_hash_payload(placeholder)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def verify_integrity_hash(anchor: GoalAnchor) -> bool:
    """Verify an anchor hash with constant-time comparison."""

    expected = compute_integrity_hash(anchor)
    return hmac.compare_digest(expected, anchor.integrity_hash)
