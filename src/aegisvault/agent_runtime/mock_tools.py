"""Local mock tools for Stage 4.0 validation."""

from __future__ import annotations

import ast
import operator
from datetime import UTC, datetime
from typing import Any

from aegisvault.agent_runtime.exceptions import ToolExecutionError
from aegisvault.agent_runtime.tools import ToolDefinition, ToolRegistry

_NOTES: list[str] = []


def get_time() -> str:
    """Return current UTC time."""

    return datetime.now(UTC).isoformat()


def calculator(expression: str) -> float:
    """Evaluate a simple arithmetic expression."""

    try:
        return float(_eval_arithmetic(ast.parse(expression, mode="eval").body))
    except Exception as exc:
        raise ToolExecutionError(f"invalid arithmetic expression: {expression!r}") from exc


def weather(location: str) -> str:
    """Return deterministic mock weather."""

    return f"Mock weather for {location}: 22C, clear, light wind."


def read_text(text: str) -> str:
    """Return simple statistics about supplied text."""

    words = [word for word in text.split() if word]
    return f"Text has {len(words)} words and {len(text)} characters."


def echo(text: str) -> str:
    """Echo text."""

    return text


def save_note(note: str) -> str:
    """Save a note in local in-memory storage."""

    _NOTES.append(note)
    return f"Saved note #{len(_NOTES)}"


def list_notes() -> list[str]:
    """List saved notes."""

    return list(_NOTES)


def clear_notes() -> None:
    _NOTES.clear()


def default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="get_time",
            description="Return the current UTC time.",
            parameters={"type": "object", "properties": {}, "required": []},
            function=get_time,
        )
    )
    registry.register(
        ToolDefinition(
            name="calculator",
            description="Evaluate a simple arithmetic expression.",
            parameters={
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "Arithmetic expression"}},
                "required": ["expression"],
            },
            function=calculator,
        )
    )
    registry.register(
        ToolDefinition(
            name="weather",
            description="Return deterministic mock weather for a location.",
            parameters={
                "type": "object",
                "properties": {"location": {"type": "string", "description": "Location name"}},
                "required": ["location"],
            },
            function=weather,
        )
    )
    registry.register(
        ToolDefinition(
            name="read_text",
            description="Read supplied text and return simple statistics.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Text to inspect"}},
                "required": ["text"],
            },
            function=read_text,
        )
    )
    registry.register(
        ToolDefinition(
            name="echo",
            description="Echo back the supplied text.",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string", "description": "Text to echo"}},
                "required": ["text"],
            },
            function=echo,
        )
    )
    registry.register(
        ToolDefinition(
            name="save_note",
            description="Save a note to local in-memory note storage.",
            parameters={
                "type": "object",
                "properties": {"note": {"type": "string", "description": "Note text"}},
                "required": ["note"],
            },
            function=save_note,
        )
    )
    registry.register(
        ToolDefinition(
            name="list_notes",
            description="List notes saved during this process.",
            parameters={"type": "object", "properties": {}, "required": []},
            function=list_notes,
        )
    )
    return registry


def _eval_arithmetic(node: ast.AST) -> int | float:
    operators: dict[type[ast.AST], Any] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in operators:
        return operators[type(node.op)](_eval_arithmetic(node.left), _eval_arithmetic(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
        return operators[type(node.op)](_eval_arithmetic(node.operand))
    raise ToolExecutionError("unsupported arithmetic expression")
