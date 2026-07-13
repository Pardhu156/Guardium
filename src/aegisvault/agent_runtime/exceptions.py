"""Agent runtime exceptions."""


class AgentRuntimeError(Exception):
    """Base class for Stage 4.0 agent runtime errors."""


class ToolRegistryError(AgentRuntimeError):
    """Raised when a tool registry operation is invalid."""


class ToolExecutionError(AgentRuntimeError):
    """Raised when a tool execution fails."""


class OllamaRuntimeError(AgentRuntimeError):
    """Raised when Ollama cannot complete a runtime request."""


class ToolCallParseError(AgentRuntimeError):
    """Raised when a tool call cannot be parsed."""
