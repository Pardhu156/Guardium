"""Goal Vault public service."""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import Any, Mapping

from aegisvault.audit import AuditSink, NullAuditSink
from aegisvault.runtime.goal_vault.base import GoalVaultBackend
from aegisvault.runtime.goal_vault.embedding import GoalEmbedder, SentenceTransformerGoalEmbedder, l2_normalize
from aegisvault.runtime.goal_vault.exceptions import (
    GoalAlreadyCommittedError,
    GoalIntegrityError,
    GoalTTLValidationError,
)
from aegisvault.runtime.goal_vault.hashing import compute_integrity_hash, verify_integrity_hash
from aegisvault.runtime.goal_vault.models import GoalAnchor, GoalCommitRequest
from aegisvault.runtime.goal_vault.serialization import utc_iso


class GoalVault:
    """Service for committing and retrieving immutable original-goal anchors."""

    def __init__(
        self,
        *,
        backend: GoalVaultBackend,
        embedder: GoalEmbedder | None = None,
        default_ttl_seconds: int = 3600,
        audit_sink: AuditSink | None = None,
        include_goal_text_in_audit: bool = False,
    ) -> None:
        if default_ttl_seconds <= 0:
            raise GoalTTLValidationError("default_ttl_seconds must be positive")
        self.backend = backend
        self.embedder = embedder or SentenceTransformerGoalEmbedder()
        self.default_ttl_seconds = default_ttl_seconds
        self.audit_sink = audit_sink or NullAuditSink()
        self.include_goal_text_in_audit = include_goal_text_in_audit

    def commit_goal(
        self,
        *,
        session_id: str,
        application_name: str,
        goal: str,
        ttl_seconds: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> GoalAnchor:
        """Commit a goal exactly once for a session."""

        started = time.perf_counter()
        request = GoalCommitRequest(
            session_id=session_id,
            application_name=application_name,
            goal=goal,
            ttl_seconds=ttl_seconds,
            metadata=metadata or {},
        )
        ttl = request.ttl_seconds or self.default_ttl_seconds
        if ttl <= 0:
            raise GoalTTLValidationError("ttl_seconds must be positive")
        self._audit("GOAL_COMMIT_ATTEMPT", request.session_id, request.application_name, ttl_seconds=ttl)
        created_at = datetime.now(UTC)
        expires_at = created_at + timedelta(seconds=ttl)
        normalized_goal = normalize_goal(request.goal)
        embedding = l2_normalize(self.embedder.embed(normalized_goal), self.embedder.dimension)
        anchor = GoalAnchor(
            session_id=request.session_id,
            application_name=request.application_name,
            original_goal=request.goal,
            normalized_goal=normalized_goal,
            goal_embedding=embedding,
            embedding_model=self.embedder.model_name,
            embedding_dimension=self.embedder.dimension,
            integrity_hash="",
            created_at=created_at,
            expires_at=expires_at,
            metadata=request.metadata,
        )
        anchor = GoalAnchor(**{**anchor_to_init_dict(anchor), "integrity_hash": compute_integrity_hash(anchor)})
        try:
            self.backend.commit_anchor(anchor, ttl)
        except GoalAlreadyCommittedError:
            self._audit(
                "GOAL_DUPLICATE_REJECTED",
                request.session_id,
                request.application_name,
                ttl_seconds=ttl,
                success=False,
                reason="duplicate commit rejected",
                latency_ms=(time.perf_counter() - started) * 1000,
            )
            raise
        self._audit(
            "GOAL_COMMITTED",
            anchor.session_id,
            anchor.application_name,
            ttl_seconds=ttl,
            embedding_model=anchor.embedding_model,
            embedding_dimension=anchor.embedding_dimension,
            latency_ms=(time.perf_counter() - started) * 1000,
            goal=anchor.original_goal,
        )
        return anchor

    def get_anchor(self, session_id: str) -> GoalAnchor:
        """Retrieve and verify a committed goal anchor."""

        started = time.perf_counter()
        anchor = self.backend.get_anchor(session_id)
        self._audit("GOAL_RETRIEVED", anchor.session_id, anchor.application_name, latency_ms=(time.perf_counter() - started) * 1000)
        if not self.verify_anchor(anchor):
            self._audit("GOAL_INTEGRITY_FAILED", anchor.session_id, anchor.application_name, success=False)
            raise GoalIntegrityError(f"goal anchor integrity verification failed for session {session_id!r}")
        self._audit("GOAL_INTEGRITY_VERIFIED", anchor.session_id, anchor.application_name)
        return anchor

    def verify_anchor(self, anchor: GoalAnchor) -> bool:
        """Return True when anchor integrity hash is valid."""

        return verify_integrity_hash(anchor)

    def delete_anchor(self, session_id: str) -> bool:
        """Delete an anchor for administrative cleanup or tests."""

        deleted = self.backend.delete_anchor(session_id)
        self._audit("GOAL_DELETED", session_id, None, success=deleted)
        return deleted

    def get_ttl(self, session_id: str) -> int | None:
        """Return remaining TTL seconds when the backend supports it."""

        return self.backend.get_ttl(session_id)

    def _audit(
        self,
        event_type: str,
        session_id: str,
        application_name: str | None,
        *,
        success: bool = True,
        reason: str | None = None,
        ttl_seconds: int | None = None,
        embedding_model: str | None = None,
        embedding_dimension: int | None = None,
        latency_ms: float | None = None,
        goal: str | None = None,
    ) -> None:
        from uuid import uuid4

        event: dict[str, Any] = {
            "event_id": str(uuid4()),
            "timestamp": utc_iso(datetime.now(UTC)),
            "event_type": event_type,
            "session_id": session_id,
            "application_name": application_name,
            "backend_type": getattr(self.backend, "backend_type", self.backend.__class__.__name__),
            "embedding_model": embedding_model,
            "embedding_dimension": embedding_dimension,
            "ttl_seconds": ttl_seconds,
            "latency_ms": latency_ms,
            "success": success,
            "reason": reason,
        }
        if self.include_goal_text_in_audit and goal is not None:
            event["goal"] = goal
        try:
            self.audit_sink.record(event)
        except Exception:
            return None


def normalize_goal(goal: str) -> str:
    """Deterministically normalize goal text without semantic rewriting."""

    return " ".join(goal.replace("\r\n", "\n").replace("\r", "\n").split())


def anchor_to_init_dict(anchor: GoalAnchor) -> dict[str, Any]:
    return {
        "session_id": anchor.session_id,
        "application_name": anchor.application_name,
        "original_goal": anchor.original_goal,
        "normalized_goal": anchor.normalized_goal,
        "goal_embedding": anchor.goal_embedding,
        "embedding_model": anchor.embedding_model,
        "embedding_dimension": anchor.embedding_dimension,
        "integrity_hash": anchor.integrity_hash,
        "created_at": anchor.created_at,
        "expires_at": anchor.expires_at,
        "metadata": anchor.metadata,
        "schema_version": anchor.schema_version,
    }
