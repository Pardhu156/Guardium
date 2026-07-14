"""Deterministic Layer 0 rules."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any, Mapping

from aegisvault.layer0.models import (
    Layer0Action,
    Layer0RequestInput,
    Layer0RiskLevel,
    Layer0RuleResult,
    Layer0ToolCallInput,
    thaw_mapping,
)
from aegisvault.policy.models import Layer0Config


REQUEST_RULE_IDS = (
    "L0_REQUEST_TYPE_INVALID",
    "L0_REQUEST_EMPTY",
    "L0_REQUEST_TOO_LARGE",
    "L0_SESSION_MISSING",
    "L0_POLICY_MISSING",
    "L0_DOMAIN_MISSING",
    "L0_DOMAIN_NOT_ALLOWED",
    "L0_RESERVED_METADATA_KEY",
    "L0_GOAL_OVERWRITE_ATTEMPT",
    "L0_FORBIDDEN_PATTERN",
)

TOOL_RULE_IDS = (
    "L0_TOOL_NAME_MISSING",
    "L0_TOOL_UNDECLARED",
    "L0_TOOL_DENIED",
    "L0_TOOL_NOT_ALLOWED",
    "L0_TOOL_ARGUMENTS_INVALID",
    "L0_TOOL_SCHEMA_INVALID",
    "L0_TOOL_ARGUMENT_TOO_LARGE",
    "L0_TOOL_RESERVED_ARGUMENT",
    "L0_TOOL_SECRET_EXPOSURE",
    "L0_TOOL_EXTERNAL_DESTINATION",
)


def request_rules(context: Layer0RequestInput, config: Layer0Config) -> list[Layer0RuleResult]:
    """Run all deterministic request rules."""

    rules = [
        _request_type_invalid(context),
        _request_empty(context),
        _request_too_large(context, config),
        _session_missing(context, config),
        _policy_missing(context),
        _domain_missing(context, config),
        _domain_not_allowed(context, config),
        _reserved_metadata_key(context.metadata, config.request.reserved_metadata_keys, "L0_RESERVED_METADATA_KEY"),
        _goal_overwrite_attempt(context, config),
        _forbidden_pattern(context, config),
    ]
    return [result for result in rules if result is not None]


def tool_rules(context: Layer0ToolCallInput, config: Layer0Config) -> list[Layer0RuleResult]:
    """Run all deterministic tool-call rules."""

    results: list[Layer0RuleResult | None] = [
        _tool_name_missing(context),
        _tool_undeclared(context),
        _tool_denied(context, config),
        _tool_not_allowed(context, config),
        _tool_arguments_invalid(context),
        _tool_schema_invalid(context),
        _tool_argument_too_large(context, config),
    ]
    arguments = context.arguments if isinstance(context.arguments, Mapping) else {}
    results.append(_reserved_metadata_key(arguments, config.tools.reserved_argument_keys, "L0_TOOL_RESERVED_ARGUMENT"))
    results.append(_tool_secret_exposure(arguments, config))
    results.append(_tool_external_destination(context, config))
    return [result for result in results if result is not None]


def find_key_paths(value: Any, keys: Iterable[str]) -> list[str]:
    """Find nested key paths case-insensitively."""

    wanted = {item.lower() for item in keys}
    matches: list[str] = []

    def visit(raw: Any, path: str) -> None:
        if isinstance(raw, Mapping):
            for key, child in raw.items():
                key_str = str(key)
                child_path = f"{path}.{key_str}" if path else key_str
                if key_str.lower() in wanted:
                    matches.append(child_path)
                visit(child, child_path)
        elif isinstance(raw, list | tuple):
            for index, child in enumerate(raw):
                visit(child, f"{path}[{index}]")

    visit(value, "")
    return matches


def safe_json_size(value: Any) -> int:
    """Return UTF-8 byte size of a deterministic JSON representation."""

    return len(json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True).encode("utf-8"))


def validate_schema(schema: Mapping[str, Any], arguments: Mapping[str, Any]) -> str | None:
    """Validate a small JSON-schema subset used by local tool definitions."""

    required = schema.get("required", [])
    missing = [str(name) for name in required if name not in arguments]
    if missing:
        return f"missing required tool parameters: {', '.join(missing)}"
    properties = schema.get("properties", {})
    if isinstance(properties, Mapping):
        for key, definition in properties.items():
            if key not in arguments or not isinstance(definition, Mapping):
                continue
            expected = definition.get("type")
            if expected and not _matches_json_type(arguments[key], str(expected)):
                return f"argument {key!r} must be {expected}"
    return None


def _request_type_invalid(context: Layer0RequestInput) -> Layer0RuleResult | None:
    if isinstance(context.request_text, str):
        return None
    return _block("L0_REQUEST_TYPE_INVALID", Layer0RiskLevel.HIGH, "Request must be a string.")


def _request_empty(context: Layer0RequestInput) -> Layer0RuleResult | None:
    if not isinstance(context.request_text, str) or context.request_text.strip():
        return None
    return _block("L0_REQUEST_EMPTY", Layer0RiskLevel.LOW, "Request must not be empty.")


def _request_too_large(context: Layer0RequestInput, config: Layer0Config) -> Layer0RuleResult | None:
    if not isinstance(context.request_text, str):
        return None
    byte_count = len(context.request_text.encode("utf-8"))
    if len(context.request_text) > config.request.max_characters or byte_count > config.request.max_bytes:
        return _block(
            "L0_REQUEST_TOO_LARGE",
            Layer0RiskLevel.MEDIUM,
            "Request exceeds Layer 0 size limits.",
            {"characters": len(context.request_text), "bytes": byte_count},
        )
    return None


def _session_missing(context: Layer0RequestInput, config: Layer0Config) -> Layer0RuleResult | None:
    if config.request.require_session_id and not (context.session_id or "").strip():
        return _block("L0_SESSION_MISSING", Layer0RiskLevel.MEDIUM, "Session ID is required.")
    return None


def _policy_missing(context: Layer0RequestInput) -> Layer0RuleResult | None:
    if not (context.policy_name or "").strip():
        return _block("L0_POLICY_MISSING", Layer0RiskLevel.HIGH, "Active policy could not be resolved.")
    return None


def _domain_missing(context: Layer0RequestInput, config: Layer0Config) -> Layer0RuleResult | None:
    if config.request.require_domain and not (context.domain or "").strip():
        return _block("L0_DOMAIN_MISSING", Layer0RiskLevel.MEDIUM, "Declared domain is required.")
    return None


def _domain_not_allowed(context: Layer0RequestInput, config: Layer0Config) -> Layer0RuleResult | None:
    if not context.domain or not config.request.allowed_domains:
        return None
    allowed = {_normalize(item) for item in config.request.allowed_domains}
    if _normalize(context.domain) not in allowed:
        return _block(
            "L0_DOMAIN_NOT_ALLOWED",
            Layer0RiskLevel.HIGH,
            "Declared domain is not allowed by Layer 0 policy.",
            {"domain": context.domain},
        )
    return None


def _reserved_metadata_key(value: Any, keys: Iterable[str], rule_id: str) -> Layer0RuleResult | None:
    matches = find_key_paths(value, keys)
    if not matches:
        return None
    return _block(rule_id, Layer0RiskLevel.HIGH, "Input attempted to set reserved middleware fields.", {"paths": matches})


def _goal_overwrite_attempt(context: Layer0RequestInput, config: Layer0Config) -> Layer0RuleResult | None:
    if context.trusted_goal_exists and context.requested_goal_update is not None:
        return _block("L0_GOAL_OVERWRITE_ATTEMPT", Layer0RiskLevel.CRITICAL, "Trusted goal overwrite attempts are not allowed.")
    paths = find_key_paths(context.metadata, ("trusted_goal", "goal_embedding"))
    if context.trusted_goal_exists and paths:
        return _block(
            "L0_GOAL_OVERWRITE_ATTEMPT",
            Layer0RiskLevel.CRITICAL,
            "Metadata attempted to overwrite trusted goal state.",
            {"paths": paths},
        )
    return None


def _forbidden_pattern(context: Layer0RequestInput, config: Layer0Config) -> Layer0RuleResult | None:
    if not isinstance(context.request_text, str):
        return None
    text = context.request_text
    for literal in config.request.forbidden_patterns.literals:
        if literal and literal in text:
            return _block("L0_FORBIDDEN_PATTERN", Layer0RiskLevel.HIGH, "Request matched a configured forbidden literal.")
    for pattern in config.request.forbidden_patterns.regex:
        if re.search(pattern, text):
            return _block("L0_FORBIDDEN_PATTERN", Layer0RiskLevel.HIGH, "Request matched a configured forbidden regex.")
    return None


def _tool_name_missing(context: Layer0ToolCallInput) -> Layer0RuleResult | None:
    if not (context.tool_name or "").strip():
        return _block("L0_TOOL_NAME_MISSING", Layer0RiskLevel.HIGH, "Tool name is required.")
    return None


def _tool_undeclared(context: Layer0ToolCallInput) -> Layer0RuleResult | None:
    if not context.tool_name or not context.tool_catalog:
        return None
    if context.tool_name not in context.tool_catalog:
        return _block("L0_TOOL_UNDECLARED", Layer0RiskLevel.HIGH, "Tool is not declared in the active tool catalogue.")
    return None


def _tool_denied(context: Layer0ToolCallInput, config: Layer0Config) -> Layer0RuleResult | None:
    if context.tool_name and _normalize(context.tool_name) in {_normalize(item) for item in config.tools.denied}:
        return _block("L0_TOOL_DENIED", Layer0RiskLevel.HIGH, "Tool is denied by Layer 0 policy.")
    return None


def _tool_not_allowed(context: Layer0ToolCallInput, config: Layer0Config) -> Layer0RuleResult | None:
    if not config.tools.allowlist_mode or not context.tool_name:
        return None
    if _normalize(context.tool_name) not in {_normalize(item) for item in config.tools.allowed}:
        return _block("L0_TOOL_NOT_ALLOWED", Layer0RiskLevel.HIGH, "Tool is not in the Layer 0 allowlist.")
    return None


def _tool_arguments_invalid(context: Layer0ToolCallInput) -> Layer0RuleResult | None:
    if isinstance(context.arguments, Mapping):
        return None
    return _block("L0_TOOL_ARGUMENTS_INVALID", Layer0RiskLevel.HIGH, "Tool arguments must be an object.")


def _tool_schema_invalid(context: Layer0ToolCallInput) -> Layer0RuleResult | None:
    if not context.tool_name or context.tool_name not in context.tool_catalog or not isinstance(context.arguments, Mapping):
        return None
    definition = context.tool_catalog[context.tool_name]
    schema = _schema_from_definition(definition)
    if not schema:
        return None
    arguments = _schema_arguments(context.arguments)
    error = validate_schema(schema, arguments)
    if error:
        return _block("L0_TOOL_SCHEMA_INVALID", Layer0RiskLevel.MEDIUM, error)
    return None


def _tool_argument_too_large(context: Layer0ToolCallInput, config: Layer0Config) -> Layer0RuleResult | None:
    if safe_json_size(context.arguments) > config.tools.max_argument_bytes:
        return _block("L0_TOOL_ARGUMENT_TOO_LARGE", Layer0RiskLevel.MEDIUM, "Tool arguments exceed Layer 0 size limits.")
    return None


def _tool_secret_exposure(arguments: Mapping[str, Any], config: Layer0Config) -> Layer0RuleResult | None:
    matches = find_key_paths(arguments, config.tools.sensitive_argument_keys)
    if not matches:
        return None
    action = Layer0Action(config.tools.sensitive_argument_action.value)
    risk = Layer0RiskLevel.HIGH if action == Layer0Action.BLOCK else Layer0RiskLevel.MEDIUM
    return Layer0RuleResult(
        rule_id="L0_TOOL_SECRET_EXPOSURE",
        matched=True,
        action=action,
        risk_level=risk,
        reason="Tool arguments contain secret-bearing field names.",
        metadata={"redacted_fields": matches},
    )


def _tool_external_destination(context: Layer0ToolCallInput, config: Layer0Config) -> Layer0RuleResult | None:
    if not context.tool_name or context.tool_name not in config.tools.destination_rules or not isinstance(context.arguments, Mapping):
        return None
    rule = config.tools.destination_rules[context.tool_name]
    allowed = {_normalize(value) for value in rule.allowed_values}
    denied: list[str] = []
    for field in rule.fields:
        values = _destination_values(context.arguments, field)
        denied.extend(value for value in values if _normalize(value) not in allowed)
    if denied:
        return _block(
            "L0_TOOL_EXTERNAL_DESTINATION",
            Layer0RiskLevel.HIGH,
            "Tool destination is not allowed by Layer 0 policy.",
            {"fields": list(rule.fields), "denied_count": len(denied)},
        )
    return None


def _destination_values(arguments: Mapping[str, Any], field: str) -> list[str]:
    values: list[str] = []
    if field in arguments:
        values.extend(_as_values(arguments[field]))
    kwargs = arguments.get("kwargs")
    if isinstance(kwargs, Mapping) and field in kwargs:
        values.extend(_as_values(kwargs[field]))
    return values


def _schema_arguments(arguments: Mapping[str, Any]) -> Mapping[str, Any]:
    kwargs = arguments.get("kwargs")
    if set(arguments).issubset({"args", "kwargs"}) and isinstance(kwargs, Mapping):
        return kwargs
    return arguments


def _schema_from_definition(definition: Any) -> Mapping[str, Any] | None:
    if isinstance(definition, Mapping):
        if isinstance(definition.get("parameters"), Mapping):
            return definition["parameters"]
        return definition
    parameters = getattr(definition, "parameters", None)
    return parameters if isinstance(parameters, Mapping) else None


def _block(rule_id: str, risk_level: Layer0RiskLevel, reason: str, metadata: Mapping[str, Any] | None = None) -> Layer0RuleResult:
    return Layer0RuleResult(
        rule_id=rule_id,
        matched=True,
        action=Layer0Action.BLOCK,
        risk_level=risk_level,
        reason=reason,
        metadata=metadata or {},
    )


def _normalize(value: str) -> str:
    return value.strip().casefold()


def _as_values(value: Any) -> list[str]:
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value]
    return [str(value)]


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return thaw_mapping(value)
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    return value


def _matches_json_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list | tuple)
    return True
