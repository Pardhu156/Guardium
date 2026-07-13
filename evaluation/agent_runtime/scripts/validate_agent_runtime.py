from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from aegisvault.agent_runtime import AgentRuntime, JsonlTraceLogger, OllamaChatClient, clear_notes, default_tool_registry
from aegisvault.agent_runtime.ollama_client import OllamaChatResult


class MockToolCallingClient:
    model = "mock-qwen-tool-runtime"

    def chat(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> OllamaChatResult:
        last = messages[-1]
        if last["role"] == "tool":
            return OllamaChatResult(payload={"message": {"role": "assistant", "content": f"Done: {last['content']}"}}, latency_ms=1.0)
        prompt = messages[-1]["content"].lower()
        calls = []
        if "time" in prompt:
            calls.append({"function": {"name": "get_time", "arguments": {}}})
        if "calculate" in prompt or "*" in prompt or "/" in prompt or "**" in prompt:
            expression = prompt.replace("calculate", "").split(",")[0].strip().strip(".") or "1+1"
            calls.append({"function": {"name": "calculator", "arguments": {"expression": expression}}})
        if "weather" in prompt:
            calls.append({"function": {"name": "weather", "arguments": {"location": "local"}}})
        if "echo" in prompt:
            calls.append({"function": {"name": "echo", "arguments": {"text": "Hello"}}})
        if "read this text" in prompt:
            calls.append({"function": {"name": "read_text", "arguments": {"text": prompt}}})
        if ("save" in prompt or "saving" in prompt) and "note" in prompt:
            calls.append({"function": {"name": "save_note", "arguments": {"note": "validation note"}}})
        if "list" in prompt and "notes" in prompt:
            calls.append({"function": {"name": "list_notes", "arguments": {}}})
        if calls:
            return OllamaChatResult(payload={"message": {"role": "assistant", "content": "", "tool_calls": calls}}, latency_ms=1.0)
        return OllamaChatResult(payload={"message": {"role": "assistant", "content": "No tool needed."}}, latency_ms=1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Stage 4.0 agent runtime.")
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--base-url", default="http://localhost:11434")
    parser.add_argument("--prompts", default="evaluation/agent_runtime/prompts/validation_prompts.jsonl")
    parser.add_argument("--reports-dir", default="evaluation/agent_runtime/reports")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    run_id = time.strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path(args.reports_dir) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_logger = JsonlTraceLogger(output_dir / "tool_traces.jsonl")
    client = MockToolCallingClient() if args.mock else OllamaChatClient(model=args.model, base_url=args.base_url)
    runtime = AgentRuntime(client=client, tools=default_tool_registry(), trace_logger=trace_logger)
    rows = [json.loads(line) for line in Path(args.prompts).read_text(encoding="utf-8").splitlines() if line.strip()]
    results = []
    clear_notes()
    for row in rows:
        result = runtime.run(row["prompt"])
        actual_tools = [record.tool_name for record in result.tool_records]
        expected = row["expected_tools"]
        match = all(tool in actual_tools for tool in expected)
        results.append(
            {
                "id": row["id"],
                "prompt": row["prompt"],
                "expected_tools": expected,
                "actual_tools": actual_tools,
                "match": match,
                "final_response": result.final_response,
                "latency_ms": result.trace.total_latency_ms,
                "trace_id": result.trace.trace_id,
                "errors": [record.error for record in result.tool_records if record.error],
            }
        )
    latencies = [row["latency_ms"] for row in results if row["latency_ms"] is not None]
    metrics = {
        "run_id": run_id,
        "model": client.model,
        "total_prompts": len(results),
        "tool_call_success_rate": sum(1 for row in results if row["match"]) / len(results),
        "tool_parsing_success_rate": sum(1 for row in results if not row["errors"]) / len(results),
        "invalid_tool_call_count": sum(1 for row in results for err in row["errors"] if err),
        "tool_usage_frequency": _tool_frequency(results),
        "latency": {
            "mean": statistics.fmean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
            "min": min(latencies) if latencies else None,
        },
    }
    (output_dir / "validation_results.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in results),
        encoding="utf-8",
    )
    (output_dir / "tool_statistics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / "runtime_summary.md").write_text(_summary(metrics, output_dir), encoding="utf-8")
    print(f"Report folder: {output_dir}")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["tool_call_success_rate"] >= 0.8 else 1


def _tool_frequency(results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in results:
        for tool in row["actual_tools"]:
            counts[tool] = counts.get(tool, 0) + 1
    return counts


def _summary(metrics: dict[str, Any], output_dir: Path) -> str:
    return f"""# Stage 4.0 Agent Runtime Validation

Output folder: `{output_dir}`

- Model: `{metrics['model']}`
- Prompts: {metrics['total_prompts']}
- Tool call success rate: {metrics['tool_call_success_rate']:.3f}
- Tool parsing success rate: {metrics['tool_parsing_success_rate']:.3f}
- Invalid tool call count: {metrics['invalid_tool_call_count']}

```json
{json.dumps(metrics['latency'], indent=2, sort_keys=True)}
```
"""


if __name__ == "__main__":
    raise SystemExit(main())
