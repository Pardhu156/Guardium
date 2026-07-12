"""Stage 3.2 Action Gate runtime API."""

from aegisvault.runtime.action_gate.cosine import cosine_similarity
from aegisvault.runtime.action_gate.evaluators import ActionEvaluator, OllamaActionEvaluator
from aegisvault.runtime.action_gate.exceptions import (
    ActionEvaluatorError,
    ActionEvaluatorTimeoutError,
    ActionGateError,
    ActionGateValidationError,
    MalformedActionEvaluatorResponseError,
)
from aegisvault.runtime.action_gate.models import (
    ActionDecisionSource,
    ActionGateConfig,
    ActionGateDecision,
    ActionRuntimeContext,
    ActionVerdict,
    ProposedToolAction,
    SideEffectLevel,
    ToolExecutionResult,
    ToolMetadata,
)
from aegisvault.runtime.action_gate.service import ActionGate, build_action_embedding_text

__all__ = [
    "ActionDecisionSource",
    "ActionEvaluator",
    "ActionEvaluatorError",
    "ActionEvaluatorTimeoutError",
    "ActionGate",
    "ActionGateConfig",
    "ActionGateDecision",
    "ActionGateError",
    "ActionGateValidationError",
    "ActionRuntimeContext",
    "ActionVerdict",
    "MalformedActionEvaluatorResponseError",
    "OllamaActionEvaluator",
    "ProposedToolAction",
    "SideEffectLevel",
    "ToolExecutionResult",
    "ToolMetadata",
    "build_action_embedding_text",
    "cosine_similarity",
]
