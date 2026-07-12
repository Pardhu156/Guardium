"""Action Gate exceptions."""


class ActionGateError(Exception):
    """Base class for Action Gate errors."""


class ActionGateValidationError(ActionGateError):
    """Raised when Action Gate input or configuration is invalid."""


class ActionEvaluatorError(ActionGateError):
    """Raised when the Action Gate LLM evaluator fails."""


class ActionEvaluatorTimeoutError(ActionEvaluatorError):
    """Raised when the Action Gate LLM evaluator times out."""


class MalformedActionEvaluatorResponseError(ActionEvaluatorError):
    """Raised when the Action Gate evaluator returns malformed output."""
