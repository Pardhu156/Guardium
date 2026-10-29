"""Check one real Qwen proposal using suite schemas, without executing any tools."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline
from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
from agentdojo.functions_runtime import FunctionsRuntime
from agentdojo.task_suite.load_suites import get_suite

from evaluation.agentdojo import run_pilot_benchmark as harness
from evaluation.agentdojo.reliability import CaseObserver, CaseTimeout, atomic_json, case_deadline, run_lock
from evaluation.agentdojo.tool_call_parser import PARSER_REVISION


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=harness.DEFAULT_SUITES, default="workspace")
    parser.add_argument("--user-task", default="user_task_35")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--timeout-seconds", type=float, default=600)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "evaluation/agentdojo/results/format_check")
    options = parser.parse_args(argv)
    forwarded = ["--llm-timeout-seconds", str(options.timeout_seconds)]
    if options.model_id:
        forwarded.extend(["--model-id", options.model_id])
    args = harness._parse_args(forwarded)
    args.phase = "baseline"  # This diagnostic proposes one call and does not use the gates.
    harness._preflight(args)
    suite = get_suite(args.benchmark_version, options.suite)
    task = suite.get_user_task_by_id(options.user_task)
    runtime = FunctionsRuntime(suite.tools)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"suite": options.suite, "user_task": options.user_task, "model": args.model_id,
              "parser_revision": PARSER_REVISION, "tools_executed": 0}
    with run_lock(options.output_dir):
        observer = CaseObserver(options.output_dir, "format_check", f"{options.suite}/{options.user_task}", 30)
        args._observer = observer
        try:
            with observer, case_deadline(options.timeout_seconds + 30):
                # No executor or tool loop is constructed: only the initial proposal is checked.
                pipeline = AgentPipeline([SystemMessage(harness._system_message(args)), InitQuery(), harness._llm(args)])
                result = pipeline.query(task.PROMPT, runtime)
                calls = result[3][-1].get("tool_calls") or []
                report.update(status="parsed" if calls else "no_tool_call", proposed_calls=[call.model_dump() for call in calls])
        except (Exception, CaseTimeout) as exc:
            report.update(status="runtime_failure", error=f"{type(exc).__name__}: {exc}")
        finally:
            report["parser_repairs"] = observer.parser_repairs
            atomic_json(options.output_dir / "format_check.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Format-check output: {options.output_dir.resolve()}")
    return 0 if report["status"] == "parsed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
