"""Tool-call parsing helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from aegisvault.agent_runtime.exceptions import ToolCallParseError


@dataclass(frozen=True, slots=True)
class ParsedToolCall:
    name: str
    arguments: dict[str, Any]


def parse_tool_calls(message: dict[str, Any]) -> list[ParsedToolCall]:
    """Parse native Ollama tool_calls or structured JSON fallback."""

    native = message.get("tool_calls")
    if isinstance(native, list) and native:
        calls = []
        for item in native:
            function = item.get("function", item) if isinstance(item, dict) else {}
            name = function.get("name")
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError as exc:
                    raise ToolCallParseError(f"tool arguments for {name!r} are not valid JSON") from exc
            if not isinstance(name, str) or not isinstance(arguments, dict):
                raise ToolCallParseError("native tool call must contain function name and object arguments")
            calls.append(ParsedToolCall(name=name, arguments=arguments))
        return calls

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return []
    payload = _extract_json(content)
    if payload is None:
        return []
    if "tool_calls" in payload:
        raw_calls = payload["tool_calls"]
        if not isinstance(raw_calls, list):
            raise ToolCallParseError("tool_calls must be a list")
        return [_parse_json_call(item) for item in raw_calls]
    if "tool" in payload:
        return [_parse_json_call(payload)]
    return []


def _parse_json_call(payload: dict[str, Any]) -> ParsedToolCall:
    name = payload.get("name", payload.get("tool"))
    arguments = payload.get("arguments", {})
    if not isinstance(name, str) or not name:
        raise ToolCallParseError("tool call JSON must include a tool/name string")
    if not isinstance(arguments, dict):
        raise ToolCallParseError("tool call arguments must be an object")
    return ParsedToolCall(name=name, arguments=arguments)


def _extract_json(content: str) -> dict[str, Any] | None:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload
