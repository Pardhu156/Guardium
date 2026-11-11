"""Gmail MCP demo integration for AegisVault."""

from aegisvault.integrations.gmail_mcp.config import GmailMCPConfig
from aegisvault.integrations.gmail_mcp.gmail_api import GmailAPIClient

__all__ = ["GmailAPIClient", "GmailMCPConfig"]
