"""Stage 4.0 local agent runtime."""

from aegisvault.agent_runtime.mock_tools import clear_notes, default_tool_registry
from aegisvault.agent_runtime.ollama_client import OllamaChatClient
from aegisvault.agent_runtime.runtime import AgentRunResult, AgentRuntime
from aegisvault.agent_runtime.tools import ToolDefinition, ToolExecutionRecord, ToolRegistry
from aegisvault.agent_runtime.tracing import AgentTrace, JsonlTraceLogger, TraceEvent

__all__ = [
    "AgentRunResult",
    "AgentRuntime",
    "AgentTrace",
    "JsonlTraceLogger",
    "OllamaChatClient",
    "ToolDefinition",
    "ToolExecutionRecord",
    "ToolRegistry",
    "TraceEvent",
    "clear_notes",
    "default_tool_registry",
]
