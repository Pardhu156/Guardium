"""Dynamic MCP client for the Gmail server."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import Client
from mcp.client.stdio import StdioServerParameters

from aegisvault.integrations.gmail_mcp.config import GmailMCPConfig


class GmailMCPClientError(RuntimeError):
    """Raised for MCP discovery or tool execution failures."""


@dataclass(frozen=True, slots=True)
class DiscoveredMCPTool:
    name: str
    description: str
    input_schema: dict[str, Any]

    def to_ollama_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


class GmailMCPClient:
    """Connect to the Gmail MCP server and discover/call tools dynamically."""

    def __init__(self, target: Any | None = None, *, config: GmailMCPConfig | None = None) -> None:
        self.config = config or GmailMCPConfig.from_env()
        self.target = target or _stdio_target(self.config)
        self._client = Client(self.target, raise_exceptions=False)
        self._connected: Client | None = None

    async def __aenter__(self) -> "GmailMCPClient":
        self._connected = await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self._client.__aexit__(exc_type, exc, traceback)
        self._connected = None

    async def discover_tools(self) -> list[DiscoveredMCPTool]:
        client = self._require_connected()
        try:
            result = await client.list_tools()
        except Exception as exc:
            raise GmailMCPClientError(f"MCP tool discovery failed: {exc.__class__.__name__}") from exc
        return [
            DiscoveredMCPTool(
                name=tool.name,
                description=tool.description or tool.name,
                input_schema=dict(tool.input_schema),
            )
            for tool in result.tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        client = self._require_connected()
        try:
            result = await client.call_tool(name, arguments or {})
        except Exception as exc:
            raise GmailMCPClientError(f"MCP tool {name!r} failed: {exc.__class__.__name__}") from exc
        if result.is_error:
            raise GmailMCPClientError(f"MCP tool {name!r} returned an error")
        if isinstance(result.structured_content, dict):
            return dict(result.structured_content)
        for item in result.content:
            text = getattr(item, "text", None)
            if isinstance(text, str):
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    return payload
        return {"success": True, "result": None}

    def _require_connected(self) -> Client:
        if self._connected is None:
            raise GmailMCPClientError("MCP client is not connected; use 'async with GmailMCPClient(...)'")
        return self._connected


def _stdio_target(config: GmailMCPConfig) -> StdioServerParameters:
    root = Path(__file__).resolve().parents[4]
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "aegisvault.integrations.gmail_mcp.server"],
        cwd=root,
        env=config.server_environment(),
    )


async def _run_command(args: argparse.Namespace) -> dict[str, Any]:
    config = GmailMCPConfig.from_env()
    if args.command == "send-dry-run" and not config.dry_run:
        raise GmailMCPClientError("send-dry-run requires AEGISVAULT_GMAIL_DRY_RUN=true")
    async with GmailMCPClient(config=config) as client:
        if args.command == "list-tools":
            tools = await client.discover_tools()
            return {"tools": [tool.name for tool in tools]}
        if args.command == "read-latest":
            return await client.call_tool("read_latest_email", {})
        return await client.call_tool(
            "send_email",
            {"to": args.to, "subject": args.subject, "body": args.body},
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Call the AegisVault Gmail MCP server.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list-tools")
    subparsers.add_parser("read-latest")
    send = subparsers.add_parser("send-dry-run")
    send.add_argument("--to", required=True)
    send.add_argument("--subject", required=True)
    send.add_argument("--body", required=True)
    args = parser.parse_args()
    result = asyncio.run(_run_command(args))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("success", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
