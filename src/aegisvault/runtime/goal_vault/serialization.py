"""Canonical Goal Vault serialization."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from aegisvault.runtime.goal_vault.exceptions import GoalSerializationError
from aegisvault.runtime.goal_vault.models import GoalAnchor, thaw_metadata


def utc_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def parse_utc_iso(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value).astimezone(UTC)
    except ValueError as exc:
        raise GoalSerializationError(f"invalid UTC timestamp {value!r}") from exc


def anchor_to_dict(anchor: GoalAnchor, *, include_hash: bool = True, hash_float_strings: bool = False) -> dict[str, Any]:
    embedding: list[float | str]
    if hash_float_strings:
        embedding = [format_float(value) for value in anchor.goal_embedding]
    else:
        embedding = list(anchor.goal_embedding)
    payload: dict[str, Any] = {
        "application_name": anchor.application_name,
        "created_at": utc_iso(anchor.created_at),
        "embedding_dimension": anchor.embedding_dimension,
        "embedding_model": anchor.embedding_model,
        "expires_at": utc_iso(anchor.expires_at),
        "goal_embedding": embedding,
        "metadata": thaw_metadata(anchor.metadata),
        "normalized_goal": anchor.normalized_goal,
        "original_goal": anchor.original_goal,
        "schema_version": anchor.schema_version,
        "session_id": anchor.session_id,
    }
    if include_hash:
        payload["integrity_hash"] = anchor.integrity_hash
    return payload


def anchor_from_dict(payload: dict[str, Any]) -> GoalAnchor:
    try:
        return GoalAnchor(
            session_id=payload["session_id"],
            application_name=payload["application_name"],
            original_goal=payload["original_goal"],
            normalized_goal=payload["normalized_goal"],
            goal_embedding=tuple(float(value) for value in payload["goal_embedding"]),
            embedding_model=payload["embedding_model"],
            embedding_dimension=int(payload["embedding_dimension"]),
            integrity_hash=payload["integrity_hash"],
            created_at=parse_utc_iso(payload["created_at"]) or datetime.now(UTC),
            expires_at=parse_utc_iso(payload.get("expires_at")),
            metadata=payload.get("metadata", {}),
            schema_version=payload["schema_version"],
        )
    except KeyError as exc:
        raise GoalSerializationError(f"missing anchor field {exc.args[0]!r}") from exc
    except TypeError as exc:
        raise GoalSerializationError(f"invalid anchor payload: {exc}") from exc


def dumps_anchor(anchor: GoalAnchor) -> str:
    return canonical_json(anchor_to_dict(anchor, include_hash=True))


def loads_anchor(raw: str | bytes) -> GoalAnchor:
    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        payload = json.loads(text)
    except Exception as exc:
        raise GoalSerializationError(f"stored anchor is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise GoalSerializationError("stored anchor JSON must be an object")
    return anchor_from_dict(payload)


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_hash_payload(anchor: GoalAnchor) -> str:
    return canonical_json(anchor_to_dict(anchor, include_hash=False, hash_float_strings=True))


def format_float(value: float) -> str:
    return format(float(value), ".17g")
