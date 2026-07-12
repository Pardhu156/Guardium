from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from aegisvault.runtime.action_gate import ActionGateConfig
from action_eval_lib import (
    DOMAIN_FILES,
    JsonlWriter,
    calculate_metrics,
    completed_keys,
    generate_run_id,
    latency_summary,
    load_cases,
    load_policies,
    progress_totals,
    render_summary,
    run_case,
    run_metadata,
    utc_now,
    verdict_consistency,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 3.3 Goal Vault + Action Gate evaluation.")
    parser.add_argument("--domains", nargs="+", default=list(DOMAIN_FILES))
    parser.add_argument("--backend", choices=["memory", "redis"], default="memory")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dataset-dir", default="evaluation/action_gate/datasets")
    parser.add_argument("--policy-dir", default="evaluation/action_gate/policies")
    parser.add_argument("--output-dir", default="evaluation/action_gate/results")
    parser.add_argument("--low-threshold", type=float, default=0.35)
    parser.add_argument("--high-threshold", type=float, default=0.82)
    parser.add_argument("--ollama-confidence", type=float, default=0.75)
    parser.add_argument("--ollama-model", default="llama3.2")
    parser.add_argument("--no-ollama", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--resume", help="Existing run folder to resume.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started_at = utc_now()
    started = time.perf_counter()
    dataset_dir = Path(args.dataset_dir)
    policy_dir = Path(args.policy_dir)
    output_root = Path(args.output_dir)
    cases, dataset_files = load_cases(dataset_dir, args.domains)
    if args.limit is not None:
        limited = []
        for domain in args.domains:
            limited.extend([case for case in cases if case.domain == domain][: args.limit])
        cases = limited
    policies, policy_files = load_policies(policy_dir, args.domains)
    config = ActionGateConfig(
        high_similarity=args.high_threshold,
        low_similarity=args.low_threshold,
        minimum_llm_confidence=args.ollama_confidence,
    )
    run_id = Path(args.resume).name if args.resume else generate_run_id()
    run_dir = Path(args.resume) if args.resume else output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = run_metadata(
        run_id=run_id,
        started_at=started_at,
        domains=args.domains,
        cases=cases,
        backend=args.backend,
        runs=args.runs,
        warmup_runs=args.warmup_runs,
        dataset_files=dataset_files,
        policy_files=policy_files,
        config=config,
        ollama_model=args.ollama_model,
    )
    write_json(run_dir / "run_metadata.json", metadata)

    totals = progress_totals(cases, args.runs)
    print("Stage 3.3 evaluation plan")
    print(f"Datasets discovered: {len(dataset_files)}")
    print(f"Domains: {', '.join(args.domains)}")
    print(f"Total cases: {totals['case_count']}")
    print(f"Expected verdicts: {json.dumps(totals['expected_verdicts'], sort_keys=True)}")
    print(f"Backend: {args.backend}")
    print(f"Ollama model: {args.ollama_model}")
    print(f"Embedding model: {metadata['embedding_model']}")
    print(f"Thresholds: low={args.low_threshold}, high={args.high_threshold}, confidence={args.ollama_confidence}")

    completed = completed_keys(run_dir / "case_results.jsonl") if args.resume else set()
    writers = {
        "case": JsonlWriter(run_dir / "case_results.jsonl"),
        "goal": JsonlWriter(run_dir / "goal_vault_results.jsonl"),
        "action": JsonlWriter(run_dir / "action_gate_results.jsonl"),
        "tool": JsonlWriter(run_dir / "tool_execution_results.jsonl"),
        "failures": JsonlWriter(run_dir / "failures.jsonl"),
    }
    results = []
    failures = []
    counters = {"matched": 0, "mismatched": 0, "false_execute": 0, "false_block": 0, "skipped": 0}
    pbar = tqdm(total=len(cases) * args.runs, desc="action-gate", ncols=110)
    try:
        for case in cases:
            for run_index in range(args.runs):
                key = (case.id, run_index)
                if key in completed:
                    counters["skipped"] += 1
                    pbar.update(1)
                    continue
                try:
                    row = run_case(
                        case=case,
                        policy=policies[case.domain],
                        backend=args.backend,
                        config=config,
                        no_ollama=args.no_ollama,
                        run_index=run_index,
                    )
                    results.append(row)
                    writers["case"].write(row)
                    writers["goal"].write({"case_id": row["case_id"], "run_index": run_index, **row["goal_vault"], "latency_ms": row["latency_ms"]["goal_commit"] + row["latency_ms"]["goal_retrieval"]})
                    writers["action"].write({k: row[k] for k in ("case_id", "run_index", "domain", "category", "expected_verdict", "actual_verdict", "goal_similarity", "decision_source", "ollama_called", "confidence", "reason")})
                    writers["tool"].write({k: row[k] for k in ("case_id", "run_index", "tool_expected_to_execute", "tool_actually_executed", "execution_count", "execution_match", "blocked_tool_executed", "justify_auto_executed")})
                    if row["matched"]:
                        counters["matched"] += 1
                    else:
                        counters["mismatched"] += 1
                        failures.append(row)
                        writers["failures"].write(row)
                    if row["false_execute"]:
                        counters["false_execute"] += 1
                    if row["false_block"]:
                        counters["false_block"] += 1
                    if args.verbose:
                        print(
                            f"{case.id}: expected={row['expected_verdict']} actual={row['actual_verdict']} "
                            f"source={row['decision_source']} sim={row['goal_similarity']} executed={row['tool_actually_executed']} matched={row['matched']}"
                        )
                except Exception as exc:
                    failure = {"case_id": case.id, "run_index": run_index, "error": str(exc), "domain": case.domain}
                    failures.append(failure)
                    writers["failures"].write(failure)
                    counters["mismatched"] += 1
                    if args.fail_fast:
                        raise
                finally:
                    avg_latency = 0.0
                    if results:
                        avg_latency = sum(row["latency_ms"]["total_case"] for row in results) / len(results)
                    pbar.set_description(f"{case.domain} actions")
                    pbar.set_postfix(
                        matched=counters["matched"],
                        mismatched=counters["mismatched"],
                        false_execute=counters["false_execute"],
                        false_block=counters["false_block"],
                        skipped=counters["skipped"],
                        avg=f"{avg_latency:.1f}ms",
                    )
                    pbar.update(1)
    finally:
        pbar.close()
        for writer in writers.values():
            writer.close()

    if not results and (run_dir / "case_results.jsonl").exists():
        from action_eval_lib import read_jsonl

        results = read_jsonl(run_dir / "case_results.jsonl")
    metrics = calculate_metrics(results)
    latency = latency_summary(results)
    consistency = verdict_consistency(results)
    metadata["completed_at"] = utc_now()
    write_json(run_dir / "run_metadata.json", metadata)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "latency_summary.json", {**latency, "verdict_consistency": consistency})
    (run_dir / "evaluation_summary.md").write_text(
        render_summary(output_dir=run_dir, metadata=metadata, metrics=metrics, latency=latency, consistency=consistency, failures=failures),
        encoding="utf-8",
    )

    print("\nEvaluation complete")
    print(f"Total discovered: {len(cases)}")
    print(f"Total executed: {len(results)}")
    print(f"Matched: {metrics['matched']}")
    print(f"Mismatched: {metrics['mismatched']}")
    print(f"False executes: {metrics['false_execute_count']}")
    print(f"False blocks: {metrics['false_block_count']}")
    print(f"Skipped: {counters['skipped']}")
    print(f"Ollama calls: {metrics['ollama_call_count']}")
    print(f"Cosine shortcuts: {metrics['cosine_execute_shortcut_count'] + metrics['cosine_block_shortcut_count']}")
    print(f"Blocked tools executed: {metrics['blocked_tool_execution_count']}")
    print(f"JUSTIFY tools auto-executed: {metrics['justify_auto_execution_count']}")
    print(f"Total runtime seconds: {time.perf_counter() - started:.2f}")
    print(f"Result folder: {run_dir}")
    return 0 if metrics["blocked_tool_execution_count"] == 0 and metrics["justify_auto_execution_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
