from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from aegisvault.integrations.gmail_mcp.client import GmailMCPClient
from aegisvault.integrations.gmail_mcp.config import GmailMCPConfig
from aegisvault.integrations.gmail_mcp.server import create_server


class FakeGmailClient:
    def read_latest_email(self) -> dict[str, Any]:
        return {
            "success": True,
            "empty": False,
            "message_id": "mock-1",
            "subject": "Mock",
            "sender": "sender@example.com",
            "body": "Mock body",
            "content_trust": "untrusted_external_data",
        }

    def send_email(self, *, to: str, subject: str, body: str) -> dict[str, Any]:
        return {"success": True, "dry_run": True, "message_id": None, "status": "validated_not_sent"}


def config(tmp_path: Path) -> GmailMCPConfig:
    return GmailMCPConfig(
        credentials_path=tmp_path / "credentials.json",
        token_path=tmp_path / "token.json",
        recipient_allowlist=frozenset({"allowed@example.com"}),
        dry_run=True,
        audit_log_path=tmp_path / "audit.jsonl",
    )


def test_mcp_tool_discovery_uses_server_descriptors(tmp_path: Path) -> None:
    async def scenario() -> None:
        server = create_server(config=config(tmp_path), gmail_client=FakeGmailClient())
        async with GmailMCPClient(target=server, config=config(tmp_path)) as client:
            tools = await client.discover_tools()
        assert [tool.name for tool in tools] == ["read_latest_email", "send_email"]
        send = next(tool for tool in tools if tool.name == "send_email")
        assert set(send.input_schema["required"]) == {"to", "subject", "body"}

    asyncio.run(scenario())


def test_mcp_read_tool_executes_and_preserves_trust_marker(tmp_path: Path) -> None:
    async def scenario() -> None:
        server = create_server(config=config(tmp_path), gmail_client=FakeGmailClient())
        async with GmailMCPClient(target=server, config=config(tmp_path)) as client:
            result = await client.call_tool("read_latest_email", {})
        assert result["message_id"] == "mock-1"
        assert result["content_trust"] == "untrusted_external_data"

    asyncio.run(scenario())
