"""Pydantic policy models."""

from __future__ import annotations

from enum import Enum
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aegisvault.exceptions import PolicyValidationError


SUPPORTED_POLICY_VERSIONS = {"1.0"}


class LowConfidenceAction(str, Enum):
    """Actions available when evaluator confidence is below threshold."""

    ALLOW = "allow"
    BLOCK = "block"
    CLARIFY = "clarify"
    REPLACE = "replace"


class FallbackAction(str, Enum):
    """Runtime fallback actions for evaluator failures."""

    ALLOW = "allow"
    BLOCK = "block"
    CLARIFY = "clarify"
    REPLACE = "replace"


class ApplicationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)


class GateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    allow_threshold: float = Field(ge=0.0, le=1.0)
    block_threshold: float = Field(ge=0.0, le=1.0)
    low_confidence_action: LowConfidenceAction


class GatesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: GateConfig
    response: GateConfig

    @model_validator(mode="after")
    def validate_gate_actions(self) -> "GatesConfig":
        if self.request.low_confidence_action == LowConfidenceAction.REPLACE:
            raise ValueError("gates.request.low_confidence_action cannot be 'replace'")
        if self.response.low_confidence_action == LowConfidenceAction.CLARIFY:
            raise ValueError("gates.response.low_confidence_action cannot be 'clarify'")
        return self


class EvaluatorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["ollama"]
    model: str = Field(min_length=1)
    base_url: str = Field(default="http://localhost:11434", min_length=1)
    timeout_seconds: float = Field(default=30, gt=0)
    temperature: float = Field(default=0, ge=0.0)


class FallbackConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluator_failure_action: FallbackAction = FallbackAction.BLOCK
    malformed_output_action: FallbackAction = FallbackAction.BLOCK


class AuditConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    output_path: str = "logs/aegisvault.jsonl"
    include_request_text: bool = True
    include_response_text: bool = True


class DeterministicChecksConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_request_chars: int = Field(default=8000, gt=0)
    max_response_chars: int = Field(default=12000, gt=0)
    blocked_phrases: list[str] = Field(default_factory=list)
    blocked_keywords: list[str] = Field(default_factory=list)
    keyword_case_insensitive: bool = True


class MessagesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_blocked: str = "I can only help with requests that fit this application's purpose."
    request_clarify: str = "Please restate your request within this application's purpose."
    response_blocked: str = "I cannot provide that response because it falls outside this application's purpose."
    response_replaced: str = "I cannot provide that response because it falls outside this application's purpose."


class Layer0FailMode(str, Enum):
    """Layer 0 unexpected-error behavior."""

    CLOSED = "closed"
    OPEN = "open"


class Layer0RuleAction(str, Enum):
    """Actions supported by deterministic Layer 0 rules."""

    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"


class Layer0ForbiddenPatternsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    literals: list[str] = Field(default_factory=list)
    regex: list[str] = Field(default_factory=list)

    @field_validator("regex")
    @classmethod
    def validate_regex(cls, value: list[str]) -> list[str]:
        for pattern in value:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid layer0 forbidden regex {pattern!r}: {exc}") from exc
        return value


class Layer0RequestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    require_session_id: bool = False
    require_domain: bool = False
    allowed_domains: list[str] = Field(default_factory=list)
    max_characters: int = Field(default=20000, gt=0)
    max_bytes: int = Field(default=50000, gt=0)
    reserved_metadata_keys: list[str] = Field(
        default_factory=lambda: [
            "trusted_goal",
            "goal_embedding",
            "policy_internal",
            "middleware_decision",
            "sentinel_state",
            "ema_drift",
            "authorization_result",
        ]
    )
    forbidden_patterns: Layer0ForbiddenPatternsConfig = Field(default_factory=Layer0ForbiddenPatternsConfig)


class Layer0DestinationRuleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fields: list[str] = Field(default_factory=list)
    allowed_values: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_destination_rule(self) -> "Layer0DestinationRuleConfig":
        if not self.fields:
            raise ValueError("layer0.tools.destination_rules fields must not be empty")
        if not self.allowed_values:
            raise ValueError("layer0.tools.destination_rules allowed_values must not be empty")
        return self


class Layer0ToolsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    allowlist_mode: bool = False
    allowed: list[str] = Field(default_factory=list)
    denied: list[str] = Field(default_factory=list)
    max_argument_bytes: int = Field(default=50000, gt=0)
    reserved_argument_keys: list[str] = Field(
        default_factory=lambda: [
            "trusted_goal",
            "goal_embedding",
            "sentinel_state",
            "authorization",
            "policy_override",
            "middleware_context",
        ]
    )
    sensitive_argument_keys: list[str] = Field(
        default_factory=lambda: [
            "password",
            "passwd",
            "secret",
            "api_key",
            "access_token",
            "refresh_token",
            "private_key",
        ]
    )
    sensitive_argument_action: Layer0RuleAction = Layer0RuleAction.WARN
    destination_rules: dict[str, Layer0DestinationRuleConfig] = Field(default_factory=dict)


class Layer0Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    fail_mode: Layer0FailMode = Layer0FailMode.CLOSED
    stop_on_first_block: bool = False
    request: Layer0RequestConfig = Field(default_factory=Layer0RequestConfig)
    tools: Layer0ToolsConfig = Field(default_factory=Layer0ToolsConfig)


class SentinelFailMode(str, Enum):
    """Sentinel unexpected-error behavior."""

    CLOSED = "closed"
    OPEN = "open"


class SentinelSignalsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: bool = True
    intent: bool = True
    action: bool = True


class SentinelRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evaluate_before_every_tool: bool = True
    require_trusted_goal: bool = True
    audit_missing_signals: bool = True


class SentinelEnforcementConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    block_on_sentinel_block: bool = True
    review_requires_action_gate_verification: bool = True


class SentinelPolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    fail_mode: SentinelFailMode = SentinelFailMode.CLOSED
    signals: SentinelSignalsConfig = Field(default_factory=SentinelSignalsConfig)
    runtime: SentinelRuntimeConfig = Field(default_factory=SentinelRuntimeConfig)
    enforcement: SentinelEnforcementConfig = Field(default_factory=SentinelEnforcementConfig)
    reasoning_weight: float = Field(default=0.20, ge=0.0, le=1.0)
    intent_weight: float = Field(default=0.35, ge=0.0, le=1.0)
    action_weight: float = Field(default=0.45, ge=0.0, le=1.0)
    ema_alpha: float = Field(default=0.40, ge=0.0, le=1.0)
    allow_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    observe_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    review_threshold: float = Field(default=0.65, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_thresholds(self) -> "SentinelPolicyConfig":
        if not (self.allow_threshold <= self.observe_threshold <= self.review_threshold):
            raise ValueError("sentinel thresholds must be ordered: allow <= observe <= review")
        return self


class DomainPolicy(BaseModel):
    """Complete domain policy loaded from YAML."""

    model_config = ConfigDict(extra="forbid")

    version: str
    application: ApplicationConfig
    purpose: str = Field(min_length=1)
    allowed_topics: list[str] = Field(min_length=1)
    blocked_topics: list[str] = Field(default_factory=list)
    gates: GatesConfig
    evaluator: EvaluatorConfig
    fallback: FallbackConfig = Field(default_factory=FallbackConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)
    checks: DeterministicChecksConfig = Field(default_factory=DeterministicChecksConfig)
    messages: MessagesConfig = Field(default_factory=MessagesConfig)
    layer0: Layer0Config = Field(default_factory=Layer0Config)
    sentinel: SentinelPolicyConfig = Field(default_factory=SentinelPolicyConfig)

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if value not in SUPPORTED_POLICY_VERSIONS:
            supported = ", ".join(sorted(SUPPORTED_POLICY_VERSIONS))
            raise ValueError(f"unsupported policy version {value!r}; supported versions: {supported}")
        return value

    @field_validator("allowed_topics", "blocked_topics")
    @classmethod
    def validate_topic_strings(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("topics must be non-empty strings")
        return value


def validation_error_from_exception(exc: Exception) -> PolicyValidationError:
    """Convert a Pydantic error into AegisVault's public validation exception."""

    return PolicyValidationError(f"Invalid AegisVault policy: {exc}")
