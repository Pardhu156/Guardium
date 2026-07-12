from __future__ import annotations

import json
import math
import threading
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from aegisvault import AegisVault
from aegisvault.audit import AuditSink
from aegisvault.evaluators import ScopeEvaluator
from aegisvault.runtime.goal_vault import (
    GoalAlreadyCommittedError,
    GoalAnchor,
    GoalBackendUnavailableError,
    GoalCommitRequest,
    GoalEmbeddingError,
    GoalEmbedder,
    GoalIntegrityError,
    GoalNotFoundError,
    GoalSerializationError,
    GoalTTLValidationError,
    GoalValidationError,
    GoalVault,
    InMemoryGoalVaultBackend,
    RedisGoalVaultBackend,
    RedisGoalVaultConfig,
    compute_integrity_hash,
    dumps_anchor,
    l2_normalize,
    loads_anchor,
    normalize_goal,
    verify_integrity_hash,
)
from aegisvault.policy.models import DomainPolicy
from aegisvault.types import EvaluationContext, GateDecision, GateType, Verdict


class FakeEmbedder(GoalEmbedder):
    model_name = "fake-3d"
    dimension = 3

    def __init__(self, values: list[float] | None = None) -> None:
        self.values = values or [3.0, 4.0, 0.0]
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return list(self.values)


class MemoryAuditSink(AuditSink):
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def record(self, event: dict[str, Any]) -> None:
        self.events.append(event)


class FakeRedisClient:
    def __init__(self, *, fail: bool = False) -> None:
        self.values: dict[str, bytes | str] = {}
        self.expiry: dict[str, float] = {}
        self.fail = fail
        self.set_calls: list[dict[str, Any]] = []

    def set(self, key: str, value: str, *, nx: bool = False, ex: int | None = None) -> bool:
        if self.fail:
            raise ConnectionError("redis unavailable")
        self.set_calls.append({"key": key, "nx": nx, "ex": ex})
        if nx and key in self.values:
            return False
        self.values[key] = value
        if ex is not None:
            self.expiry[key] = time.time() + ex
        return True

    def get(self, key: str) -> bytes | str | None:
        if self.fail:
            raise ConnectionError("redis unavailable")
        expires_at = self.expiry.get(key)
        if expires_at is not None and expires_at <= time.time():
            self.values.pop(key, None)
            return None
        return self.values.get(key)

    def delete(self, key: str) -> int:
        if self.fail:
            raise ConnectionError("redis unavailable")
        existed = key in self.values
        self.values.pop(key, None)
        self.expiry.pop(key, None)
        return int(existed)

    def ttl(self, key: str) -> int:
        if self.fail:
            raise ConnectionError("redis unavailable")
        if key not in self.values:
            return -2
        expires_at = self.expiry.get(key)
        if expires_at is None:
            return -1
        return max(0, int(expires_at - time.time()))


class FakeEvaluator(ScopeEvaluator):
    def __init__(self, verdict: Verdict) -> None:
        self.verdict = verdict

    def evaluate(
        self,
        text: str,
        policy: DomainPolicy,
        gate_type: GateType,
        context: EvaluationContext | None = None,
    ) -> GateDecision:
        return GateDecision(
            verdict=self.verdict,
            confidence=0.95,
            reason="fake",
            gate=gate_type,
            evaluator="fake",
            latency_ms=1.0,
        )


@pytest.fixture
def vault() -> GoalVault:
    return GoalVault(backend=InMemoryGoalVaultBackend(), embedder=FakeEmbedder(), default_ttl_seconds=30)


def commit_sample(vault: GoalVault, session_id: str = "session-1") -> GoalAnchor:
    return vault.commit_goal(
        session_id=session_id,
        application_name="ecommerce-support",
        goal="  Track   my\n order please. ",
        metadata={"case_id": "unit"},
    )


def build_anchor(session_id: str = "session-1", integrity_hash: str = "") -> GoalAnchor:
    anchor = GoalAnchor(
        session_id=session_id,
        application_name="app",
        original_goal="Track my order",
        normalized_goal="Track my order",
        goal_embedding=(0.6, 0.8, 0.0),
        embedding_model="fake-3d",
        embedding_dimension=3,
        integrity_hash=integrity_hash,
        created_at=datetime.now(UTC),
        expires_at=datetime.now(UTC) + timedelta(seconds=60),
        metadata={"case_id": "abc"},
    )
    return GoalAnchor(
        session_id=anchor.session_id,
        application_name=anchor.application_name,
        original_goal=anchor.original_goal,
        normalized_goal=anchor.normalized_goal,
        goal_embedding=anchor.goal_embedding,
        embedding_model=anchor.embedding_model,
        embedding_dimension=anchor.embedding_dimension,
        integrity_hash=compute_integrity_hash(anchor),
        created_at=anchor.created_at,
        expires_at=anchor.expires_at,
        metadata=anchor.metadata,
    )


def test_commit_goal_creates_l2_normalized_anchor(vault: GoalVault) -> None:
    anchor = commit_sample(vault)

    assert anchor.normalized_goal == "Track my order please."
    assert anchor.goal_embedding == pytest.approx((0.6, 0.8, 0.0))
    assert math.sqrt(sum(value * value for value in anchor.goal_embedding)) == pytest.approx(1.0)
    assert anchor.embedding_model == "fake-3d"
    assert anchor.embedding_dimension == 3
    assert verify_integrity_hash(anchor)


def test_retrieve_anchor_verifies_integrity(vault: GoalVault) -> None:
    committed = commit_sample(vault)

    retrieved = vault.get_anchor(committed.session_id)

    assert retrieved == committed


def test_duplicate_commit_is_rejected(vault: GoalVault) -> None:
    commit_sample(vault)

    with pytest.raises(GoalAlreadyCommittedError):
        commit_sample(vault)


def test_memory_backend_rejects_concurrent_duplicate_commits() -> None:
    backend = InMemoryGoalVaultBackend()
    anchor = build_anchor()
    errors: list[Exception] = []

    def commit() -> None:
        try:
            backend.commit_anchor(anchor, 30)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=commit) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(errors) == 7
    assert all(isinstance(error, GoalAlreadyCommittedError) for error in errors)


def test_missing_anchor_raises(vault: GoalVault) -> None:
    with pytest.raises(GoalNotFoundError):
        vault.get_anchor("missing-session")


def test_ttl_expiration_removes_memory_anchor() -> None:
    vault = GoalVault(backend=InMemoryGoalVaultBackend(), embedder=FakeEmbedder(), default_ttl_seconds=1)
    commit_sample(vault)

    time.sleep(1.05)

    with pytest.raises(GoalNotFoundError):
        vault.get_anchor("session-1")


def test_invalid_ttl_rejected() -> None:
    with pytest.raises(GoalTTLValidationError):
        GoalVault(backend=InMemoryGoalVaultBackend(), embedder=FakeEmbedder(), default_ttl_seconds=0)

    with pytest.raises(GoalTTLValidationError):
        GoalCommitRequest(session_id="s1", application_name="app", goal="goal", ttl_seconds=0)


@pytest.mark.parametrize("session_id", ["", "has space", "/bad", "a" * 129])
def test_invalid_session_ids_rejected(session_id: str) -> None:
    with pytest.raises(GoalValidationError):
        GoalCommitRequest(session_id=session_id, application_name="app", goal="goal")


def test_empty_goal_rejected() -> None:
    with pytest.raises(GoalValidationError):
        GoalCommitRequest(session_id="s1", application_name="app", goal="   ")


def test_secret_metadata_keys_are_rejected() -> None:
    with pytest.raises(GoalValidationError):
        GoalCommitRequest(session_id="s1", application_name="app", goal="goal", metadata={"api_key": "secret"})


def test_l2_normalize_validates_dimension_and_values() -> None:
    assert l2_normalize([3, 4], 2) == pytest.approx((0.6, 0.8))

    with pytest.raises(GoalEmbeddingError):
        l2_normalize([1.0], 2)
    with pytest.raises(GoalEmbeddingError):
        l2_normalize([0.0, 0.0], 2)
    with pytest.raises(GoalEmbeddingError):
        l2_normalize([float("nan"), 1.0], 2)
    with pytest.raises(GoalEmbeddingError):
        l2_normalize([float("inf"), 1.0], 2)


def test_goal_anchor_requires_normalized_embedding() -> None:
    with pytest.raises(GoalValidationError):
        GoalAnchor(
            session_id="s1",
            application_name="app",
            original_goal="goal",
            normalized_goal="goal",
            goal_embedding=(3.0, 4.0, 0.0),
            embedding_model="fake",
            embedding_dimension=3,
            integrity_hash="",
            created_at=datetime.now(UTC),
            expires_at=None,
        )


def test_integrity_hash_detects_tampering() -> None:
    anchor = build_anchor()
    tampered = GoalAnchor(
        session_id=anchor.session_id,
        application_name=anchor.application_name,
        original_goal="Different goal",
        normalized_goal=anchor.normalized_goal,
        goal_embedding=anchor.goal_embedding,
        embedding_model=anchor.embedding_model,
        embedding_dimension=anchor.embedding_dimension,
        integrity_hash=anchor.integrity_hash,
        created_at=anchor.created_at,
        expires_at=anchor.expires_at,
        metadata=anchor.metadata,
    )

    assert not verify_integrity_hash(tampered)


def test_vault_rejects_tampered_backend_anchor() -> None:
    backend = InMemoryGoalVaultBackend()
    vault = GoalVault(backend=backend, embedder=FakeEmbedder())
    anchor = build_anchor()
    tampered = GoalAnchor(
        session_id=anchor.session_id,
        application_name=anchor.application_name,
        original_goal="Tampered",
        normalized_goal=anchor.normalized_goal,
        goal_embedding=anchor.goal_embedding,
        embedding_model=anchor.embedding_model,
        embedding_dimension=anchor.embedding_dimension,
        integrity_hash=anchor.integrity_hash,
        created_at=anchor.created_at,
        expires_at=anchor.expires_at,
        metadata=anchor.metadata,
    )
    backend.commit_anchor(tampered, 30)

    with pytest.raises(GoalIntegrityError):
        vault.get_anchor(anchor.session_id)


def test_serialization_round_trip() -> None:
    anchor = build_anchor()

    loaded = loads_anchor(dumps_anchor(anchor))

    assert loaded == anchor
    assert verify_integrity_hash(loaded)


def test_serialization_rejects_malformed_json_and_missing_fields() -> None:
    with pytest.raises(GoalSerializationError):
        loads_anchor("{bad json")
    with pytest.raises(GoalSerializationError):
        loads_anchor(json.dumps({"session_id": "s1"}))


def test_unsupported_schema_version_rejected() -> None:
    anchor = build_anchor()
    payload = json.loads(dumps_anchor(anchor))
    payload["schema_version"] = "2.0"

    with pytest.raises((GoalSerializationError, GoalValidationError)):
        loads_anchor(json.dumps(payload))


def test_audit_events_do_not_include_goal_text_by_default() -> None:
    audit = MemoryAuditSink()
    vault = GoalVault(backend=InMemoryGoalVaultBackend(), embedder=FakeEmbedder(), audit_sink=audit)

    commit_sample(vault)
    vault.get_anchor("session-1")

    event_types = [event["event_type"] for event in audit.events]
    assert "GOAL_COMMIT_ATTEMPT" in event_types
    assert "GOAL_COMMITTED" in event_types
    assert "GOAL_RETRIEVED" in event_types
    assert "GOAL_INTEGRITY_VERIFIED" in event_types
    assert all("goal" not in event for event in audit.events)
    assert all("event_id" in event and "timestamp" in event for event in audit.events)


def test_audit_can_include_goal_text_when_explicitly_enabled() -> None:
    audit = MemoryAuditSink()
    vault = GoalVault(
        backend=InMemoryGoalVaultBackend(),
        embedder=FakeEmbedder(),
        audit_sink=audit,
        include_goal_text_in_audit=True,
    )

    commit_sample(vault)

    assert any(event.get("goal") == "  Track   my\n order please. " for event in audit.events)


def test_redis_backend_uses_set_nx_ex_and_key_schema() -> None:
    client = FakeRedisClient()
    backend = RedisGoalVaultBackend(
        RedisGoalVaultConfig(key_prefix="aegisvault:goal_anchor:"),
        client=client,
    )
    anchor = build_anchor("redis-session")

    backend.commit_anchor(anchor, 45)

    assert client.set_calls == [{"key": "aegisvault:goal_anchor:redis-session", "nx": True, "ex": 45}]
    assert backend.get_anchor("redis-session") == anchor
    with pytest.raises(GoalAlreadyCommittedError):
        backend.commit_anchor(anchor, 45)


def test_redis_backend_wraps_connection_failures() -> None:
    backend = RedisGoalVaultBackend(client=FakeRedisClient(fail=True))

    with pytest.raises(GoalBackendUnavailableError):
        backend.commit_anchor(build_anchor("redis-fail"), 30)
    with pytest.raises(GoalBackendUnavailableError):
        backend.get_anchor("redis-fail")


def test_redis_backend_surfaces_malformed_stored_value() -> None:
    client = FakeRedisClient()
    backend = RedisGoalVaultBackend(client=client)
    client.values["aegisvault:goal_anchor:bad-json"] = "{bad json"

    with pytest.raises(GoalSerializationError):
        backend.get_anchor("bad-json")


def test_delete_anchor_cleanup(vault: GoalVault) -> None:
    commit_sample(vault)

    assert vault.delete_anchor("session-1") is True
    assert vault.delete_anchor("session-1") is False


def test_no_overwrite_or_update_api_exists(vault: GoalVault) -> None:
    assert not hasattr(vault, "update_goal")
    assert not hasattr(vault.backend, "update_anchor")
    assert not hasattr(vault.backend, "overwrite_anchor")


def test_stage1_blocked_request_does_not_commit_goal(policy: DomainPolicy) -> None:
    goal_vault = GoalVault(backend=InMemoryGoalVaultBackend(), embedder=FakeEmbedder())
    guard = AegisVault(policy=policy, evaluator=FakeEvaluator(Verdict.BLOCK))
    app_called = False

    def app(prompt: str) -> str:
        nonlocal app_called
        app_called = True
        goal_vault.commit_goal(session_id="blocked-session", application_name="test-app", goal=prompt)
        return "ok"

    result = guard.wrap(app)("outside domain", session_id="blocked-session")

    assert result.application_called is False
    assert app_called is False
    with pytest.raises(GoalNotFoundError):
        goal_vault.get_anchor("blocked-session")


def test_stage1_allowed_request_can_commit_goal(policy: DomainPolicy) -> None:
    goal_vault = GoalVault(backend=InMemoryGoalVaultBackend(), embedder=FakeEmbedder())
    guard = AegisVault(policy=policy, evaluator=FakeEvaluator(Verdict.ALLOW))

    def app(prompt: str) -> str:
        goal_vault.commit_goal(session_id="allowed-session", application_name="test-app", goal=prompt)
        return "support response"

    result = guard.wrap(app)("help with my order", session_id="allowed-session")

    assert result.application_called is True
    assert goal_vault.get_anchor("allowed-session").original_goal == "help with my order"


def test_normalize_goal_is_deterministic() -> None:
    assert normalize_goal("  Help\r\nwith\t my   order  ") == "Help with my order"
