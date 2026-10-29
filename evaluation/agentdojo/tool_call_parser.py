"""Conservative syntax adaptation for AgentDojo's text tool-call protocol."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError


PARSER_REVISION = "schema-envelope-v1"
_OPEN = re.compile(r"<function\s*=\s*(?P<name>[A-Za-z_][A-Za-z0-9_.:-]*)\s*(?P<self_closing>/?)>")
_CLOSE = re.compile(r"</function\s*>")
_HINT = re.compile(r"<function\s*=")


class ToolCallFormatError(ValueError):
    """A proposed call is malformed or ambiguous; no action may be inferred."""


@dataclass(frozen=True, slots=True)
class ParsedModelCall:
    """Syntax-only interpretation; Layer 0 and runtime gates still authorize it."""

    name: str
    arguments: dict[str, Any]
    normalized_content: str
    repairs: tuple[str, ...]


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ToolCallFormatError(f"Duplicate JSON argument key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ToolCallFormatError(f"Non-finite JSON value: {value}")


def _empty_arguments_allowed(name: str, functions: Mapping[str, Any]) -> None:
    function = functions.get(name)
    parameters = getattr(function, "parameters", None)
    if parameters is None or not hasattr(parameters, "model_validate"):
        raise ToolCallFormatError(f"Cannot infer empty arguments for undeclared or untyped tool {name!r}")
    try:
        parameters.model_validate({})
    except ValidationError as exc:
        raise ToolCallFormatError(f"Tool {name!r} requires arguments; refusing to invent values") from exc


def parse_model_call(content: str, functions: Mapping[str, Any]) -> ParsedModelCall | None:
    """Accept strict JSON or unambiguous envelope repairs, without guessing values.

    An empty closed/self-closing call becomes ``{}`` only if the declared tool's
    real schema accepts it. JSON fencing and a trailing XML angle delimiter can
    be removed. Broken JSON, duplicate keys, arrays, non-finite numbers, and
    multiple call envelopes remain errors. Valid unknown tools are left for
    Layer 0 to reject rather than being implicitly registered here.
    """
    match = _OPEN.search(content)
    hint = _HINT.search(content)
    if match is None:
        if hint:
            raise ToolCallFormatError("Malformed function-call opening tag")
        return None
    if hint is not None and hint.start() < match.start():
        raise ToolCallFormatError("Malformed function-call opening tag before proposed call")

    name = match.group("name")
    repairs: list[str] = []
    position = match.end()
    position += len(content[position:]) - len(content[position:].lstrip())
    self_closing = bool(match.group("self_closing"))
    closing = _CLOSE.match(content, position)
    if self_closing:
        if closing is not None:
            raise ToolCallFormatError("Conflicting self-closing and closing function tags")
        end = match.end()
        repairs.append("self_closing_function_tag")
    if self_closing or closing is not None:
        _empty_arguments_allowed(name, functions)
        arguments: dict[str, Any] = {}
        repairs.append("empty_arguments_schema_verified")
        if not self_closing:
            end = closing.end()
    else:
        if not content[position:].strip():
            raise ToolCallFormatError("Incomplete call has neither arguments nor a closing tag")
        fenced = re.match(r"```(?:json)?[ \t]*\n?", content[position:])
        if fenced:
            position += fenced.end()
            position += len(content[position:]) - len(content[position:].lstrip())
            repairs.append("json_markdown_fence")
        decoder = json.JSONDecoder(object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        try:
            arguments, consumed = decoder.raw_decode(content[position:])
        except json.JSONDecodeError as exc:
            raise ToolCallFormatError(f"Invalid argument JSON: {exc.msg}") from exc
        if not isinstance(arguments, dict):
            raise ToolCallFormatError("Tool arguments must be a JSON object")
        # Let the JSON decoder find the boundary; tag-like strings inside arguments
        # are data and must never be interpreted as additional call envelopes.
        position += consumed
        position += len(content[position:]) - len(content[position:].lstrip())
        if fenced:
            if not content.startswith("```", position):
                raise ToolCallFormatError("Unclosed JSON markdown fence")
            position += 3
            position += len(content[position:]) - len(content[position:].lstrip())
        if content.startswith(">", position):
            repairs.append("trailing_xml_angle")
            position += 1
            position += len(content[position:]) - len(content[position:].lstrip())
        closing = _CLOSE.match(content, position)
        if closing is not None:
            end = closing.end()
        elif content[position:].strip() == "```" and content[:match.start()].lstrip().startswith("```"):
            end = position  # Preserve the surrounding fence after the repaired tag.
            repairs.append("outer_markdown_fence")
        elif not content[position:].strip():
            end = len(content)
        else:
            raise ToolCallFormatError("Unexpected content after argument JSON; refusing to discard it")
        # Also reject numerical overflow such as 1e999, which JSONDecoder maps to infinity.
        try:
            json.dumps(arguments, allow_nan=False)
        except ValueError as exc:
            raise ToolCallFormatError("Tool arguments contain a non-finite number") from exc
        if closing is None:
            repairs.append("missing_function_closing_tag")

    if _HINT.search(content[end:]):
        raise ToolCallFormatError("Multiple proposed function calls are ambiguous; one call per step is required")

    if repairs:
        normalized = (
            content[:match.start()]
            + f"<function={name}>"
            + json.dumps(arguments, ensure_ascii=True, allow_nan=False)
            + "</function>"
            + content[end:]
        )
    else:
        normalized = content
    return ParsedModelCall(name, arguments, normalized, tuple(repairs))
