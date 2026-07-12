from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from action_eval_lib import read_jsonl, render_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate Stage 3.3 Markdown report.")
    parser.add_argument("run_dir")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    latency = json.loads((run_dir / "latency_summary.json").read_text(encoding="utf-8"))
    failures = read_jsonl(run_dir / "failures.jsonl") if (run_dir / "failures.jsonl").exists() else []
    consistency = latency.get("verdict_consistency", {"unstable_case_count": 0, "unstable_cases": {}})
    (run_dir / "evaluation_summary.md").write_text(
        render_summary(output_dir=run_dir, metadata=metadata, metrics=metrics, latency=latency, consistency=consistency, failures=failures),
        encoding="utf-8",
    )
    print(run_dir / "evaluation_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
