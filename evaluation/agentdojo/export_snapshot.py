"""Export aggregate coverage and safe reproduction IDs without raw benchmark text."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def failure_category(row: dict[str, Any]) -> str:
    """Classify recorded errors, not the underlying cause or security outcome."""
    error = str(row.get("error", ""))
    if "APITimeoutError" in error:
        return "model_request_timeout"
    if "APIConnectionError" in error:
        return "model_connection_error"
    if "Invalid tool-call format" in error:
        return "malformed_tool_call"
    if row.get("failure_category") == "infrastructure_context_limit":
        return "context_limit"
    return "other_runtime_failure_or_skip"


def export_snapshot(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Publish only counts and identifiers, refusing to overwrite a past snapshot."""
    source = run_dir / "protected_results.jsonl"
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    raw = source.read_bytes()
    rows: dict[str, dict[str, Any]] = {}
    for line in raw.decode("utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["case_id"]] = row
    evaluated = [row for row in rows.values() if not row.get("error") and row.get("status", "completed") == "completed"]
    evaluated_ids = {row["case_id"] for row in evaluated}
    unresolved = [row for row in rows.values() if row["case_id"] not in evaluated_ids]
    clean = [row for row in evaluated if row.get("case_type") == "benign"]
    attacks = [row for row in evaluated if row.get("case_type") == "attack"]
    attempted_attacks = sum(row.get("case_type") == "attack" for row in rows.values())
    successful_attacks = sum(row.get("injection_success") is True for row in attacks)
    unknown_attacks = attempted_attacks - len(attacks)
    metrics_raw = (run_dir / "protected_metrics_live.json").read_bytes()
    metrics = json.loads(metrics_raw)
    if metrics["cases"] != len(rows) or metrics["evaluated_cases"] != len(evaluated):
        raise ValueError("Metrics and results changed or disagree; export again after the current case is saved")
    summary = {
        "snapshot_created_at": datetime.now(UTC).isoformat(),
        "source_results_updated_at": datetime.fromtimestamp(source.stat().st_mtime, UTC).isoformat(),
        "source_results_sha256": hashlib.sha256(raw).hexdigest(),
        "source_metrics_sha256": hashlib.sha256(metrics_raw).hexdigest(),
        "saved_cases": len(rows),
        "evaluated_cases": len(evaluated),
        "unresolved_cases": len(unresolved),
        "evaluated_clean_cases": len(clean),
        "clean_task_successes": sum(row.get("utility") is True for row in clean),
        "evaluated_attack_cases": len(attacks),
        "attempted_attack_cases": attempted_attacks,
        "successful_attacks": successful_attacks,
        "conditional_attack_success_rate": _rate(successful_attacks, len(attacks)),
        "attack_evaluation_coverage": _rate(len(attacks), attempted_attacks),
        "asr_lower_bound_over_attempted": _rate(successful_attacks, attempted_attacks),
        "asr_upper_bound_over_attempted": _rate(successful_attacks + unknown_attacks, attempted_attacks),
        "candidate_middleware_false_positives": metrics["middleware_false_positives"],
        "candidate_model_utility_failures": metrics["model_utility_failures"],
        "unresolved_by_category": dict(Counter(failure_category(row) for row in unresolved)),
        "per_suite": {
            suite: {
                "saved": sum(row.get("suite") == suite for row in rows.values()),
                "evaluated": sum(row.get("suite") == suite for row in evaluated),
                "unresolved": sum(row.get("suite") == suite for row in unresolved),
            }
            for suite in sorted({row["suite"] for row in rows.values()})
        },
        "limitations": [
            "Protected-only snapshot; no matched full baseline comparison.",
            "Unknown runtime outcomes are excluded from conditional ASR, not successful defenses.",
            "FP and model-failure attribution are candidates requiring manual trace review.",
            "Latest saved outcome per case after retries, not independent trials.",
            "Serving context and parser revisions changed during recovery; not a single frozen protocol.",
        ],
    }
    manifest = {
        "benchmark_version": metadata["benchmark_version"],
        "attack": metadata["attack"],
        "case_ids": [row["case_id"] for row in unresolved],
        "cases": [
            {key: row.get(key) for key in ("case_id", "suite", "user_task_id", "injection_task_id", "case_type")}
            | {"failure_category": failure_category(row)}
            for row in unresolved
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, payload in (("summary.json", summary), ("unresolved_cases.json", manifest)):
        (output_dir / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = export_snapshot(args.run_dir, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Snapshot export failed ({type(exc).__name__}); check inputs and use a new output directory.\n")
    print(f"Exported: {summary['evaluated_cases']} evaluated, {summary['unresolved_cases']} unresolved")
    print(f"Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
