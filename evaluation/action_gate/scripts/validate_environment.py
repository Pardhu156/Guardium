from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from action_eval_lib import DOMAIN_FILES, load_cases, load_policies


def main() -> int:
    dataset_dir = ROOT / "evaluation/action_gate/datasets"
    policy_dir = ROOT / "evaluation/action_gate/policies"
    domains = list(DOMAIN_FILES)
    cases, dataset_files = load_cases(dataset_dir, domains)
    policies, policy_files = load_policies(policy_dir, domains)
    print("Stage 3.3 environment validation")
    print(f"Datasets: {len(dataset_files)}")
    print(f"Policies: {len(policy_files)}")
    print(f"Cases: {len(cases)}")
    print(f"Domains: {', '.join(sorted(policies))}")
    print("Redis optional check: run `redis-cli ping` for live Redis backend.")
    print("Ollama optional check: run `curl http://localhost:11434/api/tags`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
