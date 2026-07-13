"""Reusable tool registry for the Stage 4.0 runtime."""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Mapping

from aegisvault.agent_runtime.exceptions import ToolExecutionError, ToolRegistryError


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A callable tool exposed to the agent runtime."""

    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable[..., Any] = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ToolRegistryError("tool name must not be empty")
        if not self.description.strip():
            raise ToolRegistryError("tool description must not be empty")
        if not callable(self.function):
            raise ToolRegistryError(f"tool {self.name!r} function must be callable")

    def to_ollama_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass(slots=True)
class ToolExecutionRecord:
    """Structured record for a single tool execution."""

    tool_name: str
    arguments: dict[str, Any]
    result: Any
    latency_ms: float
    error: str | None = None


class ToolRegistry:
    """Registry for named callable tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            raise ToolRegistryError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise ToolRegistryError(f"unknown tool {name!r}") from exc

    def list_tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def to_ollama_tools(self) -> list[dict[str, Any]]:
        return [tool.to_ollama_tool() for tool in self.list_tools()]

    def execute(self, name: str, arguments: Mapping[str, Any] | None = None) -> ToolExecutionRecord:
        tool = self.get(name)
        args = dict(arguments or {})
        started = time.perf_counter()
        try:
            _validate_required_parameters(tool.parameters, args)
            result = tool.function(**args)
            return ToolExecutionRecord(
                tool_name=name,
                arguments=args,
                result=result,
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        except Exception as exc:
            if isinstance(exc, ToolExecutionError):
                error = str(exc)
            else:
                error = f"{exc.__class__.__name__}: {exc}"
            return ToolExecutionRecord(
                tool_name=name,
                arguments=args,
                result=None,
                latency_ms=(time.perf_counter() - started) * 1000,
                error=error,
            )


def _validate_required_parameters(schema: dict[str, Any], args: dict[str, Any]) -> None:
    required = schema.get("required", [])
    missing = [name for name in required if name not in args]
    if missing:
        raise ToolExecutionError(f"missing required tool parameters: {', '.join(missing)}")


def schema_from_callable(function: Callable[..., Any]) -> dict[str, Any]:
    """Build a small JSON schema from a callable signature."""

    signature = inspect.signature(function)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, parameter in signature.parameters.items():
        if parameter.default is inspect.Signature.empty:
            required.append(name)
        properties[name] = {"type": "string"}
    return {"type": "object", "properties": properties, "required": required}
