"""Inspect persisted benchmark progress without loading models or contacting servers."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def inspect_run(run_dir: Path, phase: str = "protected") -> dict[str, Any]:
    """Report unique saved cases and keep runtime failures out of the ASR denominator."""
    rows: dict[str, dict[str, Any]] = {}
    path = run_dir / f"{phase}_results.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["case_id"]] = row
    valid = [r for r in rows.values() if not r.get("error") and r.get("status", "completed") == "completed"]
    attacks = [r for r in valid if r.get("case_type") == "attack"]
    successful_attacks = sum(bool(r.get("injection_success", r.get("security", False))) for r in attacks)
    selected_path = run_dir / "selected_cases.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))["cases"] if selected_path.exists() else []
    completed_ids = {r["case_id"] for r in valid}
    pending = [r["case_id"] for r in selected if r["case_id"] not in completed_ids]
    status_path = run_dir / f"{phase}_current_case.json"
    return {
        "result_folder": str(run_dir.resolve()),
        "saved_unique_cases": len(rows),
        "evaluated_cases": len(valid),
        "saved_clean_cases": sum(r.get("case_type") == "benign" for r in rows.values()),
        "saved_attack_cases": sum(r.get("case_type") == "attack" for r in rows.values()),
        "evaluated_attack_cases": len(attacks),
        "successful_attacks": successful_attacks,
        "attack_success_rate": successful_attacks / len(attacks) if attacks else None,
        "runtime_failures_or_skips": [r["case_id"] for r in rows.values() if r["case_id"] not in completed_ids],
        "remaining_including_retries": len(pending),
        "next_pending_cases": pending[:5],
        "per_suite_saved": dict(Counter(r["suite"] for r in rows.values())),
        "last_checkpoint": json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else None,
        "note": "A checkpoint is persisted history, not proof that a process is currently running.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--phase", choices=["baseline", "protected"], default="protected")
    args = parser.parse_args()
    if not args.run_dir.is_dir():
        parser.error(f"Result directory does not exist: {args.run_dir}")
    print(json.dumps(inspect_run(args.run_dir, args.phase), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
