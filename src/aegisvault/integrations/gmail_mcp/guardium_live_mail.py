"""Run one explicitly authorized Guardium Gmail delivery check."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from aegisvault.agent_runtime import OllamaChatClient
from aegisvault.audit import JsonLineAuditSink
from aegisvault.integrations.gmail_mcp.agent import GmailMCPAgent
from aegisvault.integrations.gmail_mcp.auth import GmailAuthenticator
from aegisvault.integrations.gmail_mcp.client import DiscoveredMCPTool, GmailMCPClient, GmailMCPClientError
from aegisvault.integrations.gmail_mcp.config import GmailMCPConfig
from aegisvault.integrations.gmail_mcp.gmail_api import GmailAPIClient
from aegisvault.integrations.gmail_mcp.guardrails import build_protected_guardrails
from aegisvault.integrations.gmail_mcp.server import create_server
from aegisvault.policy import load_policy


SUBJECT = "Guardium Authorized Test"
BODY = "GUARDIUM_LIVE_TEST_123"
RUNS = Path("logs/guardium-live-mail")
POLICY = Path(__file__).resolve().parents[4] / "mcp/gmail/policy.yaml"


class LiveMailError(RuntimeError):
    """An operator-facing delivery check error."""


def _mailbox(address: str) -> str:
    value = address.strip().lower()
    if not re.fullmatch(r"[a-z0-9._+\-]+@[a-z0-9\-]+(?:\.[a-z0-9\-]+)+", value):
        raise LiveMailError("Use a single plain mailbox address.")
    return value


def _check_addresses(config: GmailMCPConfig, sender: str, recipient: str) -> None:
    if sender == recipient:
        raise LiveMailError("Choose a second test mailbox as the recipient.")
    if recipient not in config.recipient_allowlist:
        raise LiveMailError("Add the recipient to the existing Gmail MCP recipient allowlist.")


def _save(path: Path, report: dict[str, Any]) -> None:
    # A reader sees either the previous report or the entire updated report.
    temporary = path.with_suffix(f".{uuid4().hex}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class ConfirmedMessageClient(GmailMCPClient):
    """Expose only the existing send tool and dispatch at most one exact message."""

    def __init__(
        self,
        *,
        config: GmailMCPConfig,
        service: Any,
        recipient: str,
        report: dict[str, Any],
        report_path: Path,
    ) -> None:
        gmail = GmailAPIClient(service=service, recipient_allowlist=config.recipient_allowlist, dry_run=config.dry_run)
        super().__init__(target=create_server(config=config, gmail_client=gmail), config=config)
        self.message = {"to": recipient, "subject": SUBJECT, "body": BODY}
        self.report = report
        self.report_path = report_path

    async def discover_tools(self) -> list[DiscoveredMCPTool]:
        tools = [tool for tool in await super().discover_tools() if tool.name == "send_email"]
        if len(tools) != 1:
            raise GmailMCPClientError("Expected exactly one Gmail MCP send_email tool.")
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        if name != "send_email" or arguments != self.message:
            raise GmailMCPClientError("Proposed tool call differs from the confirmed message.")
        if self.report["dispatch_status"] != "not_dispatched":
            raise GmailMCPClientError("This run already dispatched its one permitted MCP call.")

        # Persist before dispatch so a crash cannot be mistaken for a clean failure.
        self.report["dispatch_status"] = "outcome_unknown"
        _save(self.report_path, self.report)
        result = await super().call_tool(name, arguments)
        if result.get("success") is True:
            if self.config.dry_run and result.get("dry_run") is True and result.get("status") == "validated_not_sent":
                self.report["dispatch_status"] = "validated_not_sent"
            elif not self.config.dry_run and result.get("dry_run") is False and result.get("status") == "sent":
                message_id = result.get("message_id")
                if isinstance(message_id, str) and re.fullmatch(r"[A-Za-z0-9_-]+", message_id):
                    self.report["dispatch_status"] = "gmail_api_accepted"
                    self.report["gmail_api_message_id"] = message_id
        _save(self.report_path, self.report)
        return result


async def _preflight(args: argparse.Namespace) -> tuple[GmailMCPConfig, OllamaChatClient, Any, str]:
    config = GmailMCPConfig.from_env()
    if not config.dry_run:
        raise LiveMailError("Keep the environment in dry-run mode. Enable delivery with --live-send on this CLI.")
    sender, recipient = _mailbox(args.sender), _mailbox(args.recipient)
    _check_addresses(config, sender, recipient)
    ollama_url = os.getenv("AEGISVAULT_GMAIL_OLLAMA_URL", "http://localhost:11434")
    if urlsplit(ollama_url).hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise LiveMailError("Use a local Ollama URL for this Gmail check.")
    actor = OllamaChatClient(
        model=os.getenv("AEGISVAULT_GMAIL_AGENT_MODEL", "qwen3:4b-instruct"),
        base_url=ollama_url,
        timeout_seconds=float(os.getenv("AEGISVAULT_GMAIL_OLLAMA_TIMEOUT_SECONDS", "600")),
        temperature=0,
    )
    actor_models = actor.list_models()
    if actor.model not in actor_models:
        raise LiveMailError("The configured Qwen model is missing from Ollama.")
    evaluator = load_policy(POLICY).evaluator
    verifier = OllamaChatClient(model=evaluator.model, base_url=evaluator.base_url, timeout_seconds=5)
    verifier_models = actor_models if verifier.base_url == actor.base_url else verifier.list_models()
    if not {verifier.model, f"{verifier.model}:latest"}.intersection(verifier_models):
        raise LiveMailError("The Guardium policy verifier model is missing from Ollama.")
    authenticated = GmailAuthenticator(
        credentials_path=config.credentials_path, token_path=config.token_path,
    ).authenticate(interactive=False)
    if authenticated.email_address.casefold() != sender:
        raise LiveMailError("OAuth account differs from the expected sender.")
    # Stdio discovery requires neither a mailbox read nor a send.
    async with GmailMCPClient(config=config) as client:
        if not any(tool.name == "send_email" for tool in await client.discover_tools()):
            raise LiveMailError("Gmail MCP did not advertise send_email.")
    print("Guardium preflight passed. Actor, verifier, Gmail account, MCP send tool, and recipient allowlist verified.")
    return config, actor, authenticated.service, recipient


async def _send(args: argparse.Namespace) -> int:
    config, actor, service, recipient = await _preflight(args)
    if args.live_send:
        print(f"From: {_mailbox(args.sender)}\nTo: {recipient}\nSubject: {SUBJECT}\nBody: {BODY}")
        if input("Type SEND to authorize exactly this message: ") != "SEND":
            raise LiveMailError("Live send was not confirmed.")

    run_id = uuid4().hex
    folder = RUNS / run_id
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / "result.json"
    config = replace(
        config, recipient_allowlist=frozenset({recipient}), dry_run=not args.live_send,
        live_send_approved=args.live_send, audit_log_path=folder / "audit.jsonl",
    )
    report = {
        "run_id": run_id,
        "model": actor.model,
        "mode": "live" if args.live_send else "dry_run",
        "dispatch_status": "not_dispatched",
        "gmail_api_message_id": None,
        "operator_confirmed_inbox_delivery": False,
        "request_gate": None,
        "response_gate": None,
        "policy": "mcp/gmail/policy.yaml",
        "send_proposals": [],
        "error": None,
    }
    _save(path, report)
    print(f"Guardium report: {path}")

    try:
        guardrails = build_protected_guardrails(policy_path=POLICY, audit_log_path=config.audit_log_path)
        agent = GmailMCPAgent(
            model_client=actor,
            mcp_client=ConfirmedMessageClient(
                config=config, service=service, recipient=recipient, report=report, report_path=path,
            ),
            guardrails=guardrails,
            audit_sink=JsonLineAuditSink(config.audit_log_path),
            max_tool_rounds=1,
        )
        request = f'Send a test email to my approved test mailbox {recipient} with subject "{SUBJECT}" and body "{BODY}".'
        result = await agent.run(request, session_id=run_id)
        if result.request_authorization and result.request_authorization.decision:
            report["request_gate"] = result.request_authorization.decision.verdict.value
        if result.response_authorization:
            report["response_gate"] = result.response_authorization.decision.verdict.value
        for record in result.tool_records:
            if record.tool_name == "send_email":
                decision = record.authorization.decision if record.authorization else None
                report["send_proposals"].append({
                    "action_gate": decision.verdict.value if decision else None,
                    "decision_source": decision.decision_source.value if decision else None,
                    "policy_rule": (
                        decision.metadata.get("layer0_tool_decision", {}).get("rule_id")
                        or decision.metadata.get("blocked_by")
                        or decision.metadata.get("override")
                        or decision.metadata.get("threshold_type")
                    ) if decision else None,
                    "mcp_returned_result": record.executed,
                })
    except Exception as exc:
        report["error"] = type(exc).__name__
    finally:
        _save(path, report)
    print(json.dumps(report, indent=2))
    expected = "gmail_api_accepted" if args.live_send else "validated_not_sent"
    return 0 if report["dispatch_status"] == expected and report["error"] is None else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight")
    send = commands.add_parser("send")
    send.add_argument("--live-send", action="store_true")
    for command in (preflight, send):
        command.add_argument("--sender", required=True)
        command.add_argument("--recipient", required=True)
    report = commands.add_parser("report")
    report.add_argument("run_id")
    report.add_argument("--confirm-inbox-delivery", action="store_true", help="Record your observed inbox delivery.")
    args = parser.parse_args()

    try:
        if args.command == "preflight":
            asyncio.run(_preflight(args))
            return 0
        if args.command == "send":
            return asyncio.run(_send(args))
        if not re.fullmatch(r"[0-9a-f]{32}", args.run_id):
            raise LiveMailError("Use the run ID printed by Guardium.")
        path = RUNS / args.run_id / "result.json"
        if not path.is_file():
            raise LiveMailError("No saved result for that run ID.")
        result = json.loads(path.read_text(encoding="utf-8"))
        if args.confirm_inbox_delivery:
            if result["dispatch_status"] != "gmail_api_accepted":
                raise LiveMailError("Inbox confirmation requires a Gmail API acceptance receipt.")
            result["operator_confirmed_inbox_delivery"] = True
            _save(path, result)
        print(json.dumps(result, indent=2))
        return 0
    except (EOFError, KeyboardInterrupt):
        print("Guardium check interrupted. Inspect any printed report before retrying.", file=sys.stderr)
    except LiveMailError as exc:
        print(f"Guardium: {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"Guardium check failed ({type(exc).__name__}). Check OAuth and Ollama configuration.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
