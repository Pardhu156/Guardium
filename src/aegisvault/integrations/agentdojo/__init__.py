"""AgentDojo compatibility adapter."""

from aegisvault.integrations.agentdojo.adapter import AgentDojoAegisVaultAdapter, AgentDojoAdapterConfig
from aegisvault.integrations.agentdojo.models import (
    AgentDojoAdapterResult,
    AgentDojoTaskView,
    AgentDojoToolResult,
    AgentDojoToolSpec,
)
try:
    from aegisvault.integrations.agentdojo.pipeline import AegisVaultAgentDojoToolsExecutor, build_aegisvault_agentdojo_pipeline
except ModuleNotFoundError:
    AegisVaultAgentDojoToolsExecutor = None  # type: ignore[assignment]
    build_aegisvault_agentdojo_pipeline = None  # type: ignore[assignment]

__all__ = [
    "AgentDojoAdapterConfig",
    "AgentDojoAdapterResult",
    "AgentDojoAegisVaultAdapter",
    "AgentDojoTaskView",
    "AgentDojoToolResult",
    "AgentDojoToolSpec",
    "AegisVaultAgentDojoToolsExecutor",
    "build_aegisvault_agentdojo_pipeline",
]
