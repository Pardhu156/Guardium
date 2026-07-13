"""Minimal tool-calling agent runtime for Stage 4.0."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from aegisvault.agent_runtime.ollama_client import OllamaChatClient
from aegisvault.agent_runtime.parser import ParsedToolCall, parse_tool_calls
from aegisvault.agent_runtime.tools import ToolExecutionRecord, ToolRegistry
from aegisvault.agent_runtime.tracing import AgentTrace, JsonlTraceLogger, utc_now

SYSTEM_PROMPT = """You are a minimal local tool-calling assistant.
Use tools when they are useful. If calling tools as JSON fallback, return only:
{"tool_calls":[{"name":"tool_name","arguments":{}}]}
After receiving tool results, answer the user concisely."""


@dataclass(slots=True)
class AgentRunResult:
    final_response: str
    trace: AgentTrace
    tool_records: list[ToolExecutionRecord]


class AgentRuntime:
    """Generic local agent runtime independent of AegisVault guards."""

    def __init__(
        self,
        *,
        client: OllamaChatClient,
        tools: ToolRegistry,
        trace_logger: JsonlTraceLogger | None = None,
        max_tool_rounds: int = 4,
    ) -> None:
        self.client = client
        self.tools = tools
        self.trace_logger = trace_logger
        self.max_tool_rounds = max_tool_rounds

    def run(self, prompt: str) -> AgentRunResult:
        started = time.perf_counter()
        trace = AgentTrace.start(model=self.client.model, user_prompt=prompt)
        trace.add_event("user_prompt", payload={"prompt": prompt})
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        tool_records: list[ToolExecutionRecord] = []
        final_response = ""

        for round_index in range(self.max_tool_rounds + 1):
            trace.add_event("model_request", payload={"round": round_index, "messages": _redact_messages(messages)})
            chat_result = self.client.chat(messages=messages, tools=self.tools.to_ollama_tools())
            message = chat_result.payload.get("message", {})
            trace.add_event(
                "model_response",
                payload={"round": round_index, "message": message},
                latency_ms=chat_result.latency_ms,
            )
            trace.token_usage = _extract_usage(chat_result.payload)

            tool_calls = parse_tool_calls(message)
            if not tool_calls:
                final_response = str(message.get("content") or "")
                break

            for order, tool_call in enumerate(tool_calls, start=1):
                record = self._execute_tool(tool_call)
                tool_records.append(record)
                trace.add_event(
                    "tool_execution",
                    payload={
                        "round": round_index,
                        "order": order,
                        "tool": record.tool_name,
                        "arguments": record.arguments,
                        "result": record.result,
                    },
                    latency_ms=record.latency_ms,
                    error=record.error,
                )
                messages.append(
                    {
                        "role": "tool",
                        "name": record.tool_name,
                        "content": str({"result": record.result, "error": record.error}),
                    }
                )
        else:
            final_response = "Tool loop stopped after reaching the maximum tool rounds."

        trace.final_response = final_response
        trace.completed_at = utc_now()
        trace.total_latency_ms = (time.perf_counter() - started) * 1000
        trace.add_event("final_response", payload={"content": final_response}, latency_ms=trace.total_latency_ms)
        if self.trace_logger is not None:
            self.trace_logger.record(trace)
        return AgentRunResult(final_response=final_response, trace=trace, tool_records=tool_records)

    def _execute_tool(self, tool_call: ParsedToolCall) -> ToolExecutionRecord:
        return self.tools.execute(tool_call.name, tool_call.arguments)


def _extract_usage(payload: dict[str, Any]) -> dict[str, Any]:
    keys = ("prompt_eval_count", "eval_count", "total_duration", "load_duration", "prompt_eval_duration", "eval_duration")
    return {key: payload[key] for key in keys if key in payload}


def _redact_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(message) for message in messages]
