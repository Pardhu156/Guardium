# AegisBench v1

AegisBench v1 is the Stage 4.2 email security benchmark for AegisVault. It evaluates middleware behavior, not Email Agent quality.

## Flow

```text
WITHOUT AegisVault:
User goal -> Email Agent/tool action -> Email tools -> response

WITH AegisVault:
User goal -> Request Gate -> Goal Vault -> Email Agent/tool action -> Action Gate -> Email tools -> Response Gate -> response
```

The Stage 4.1 email corpus is reused. AegisBench does not regenerate the mailbox.

## Dataset

Benchmark cases live at:

```text
datasets/benchmarks/aegisbench_v1/cases.jsonl
```

Each case includes ground truth for request verdict, action verdict, response verdict, expected execution, goal match, referenced email, and proposed tool action.

Categories:

- legitimate tasks
- cross-domain requests
- prompt injection
- goal drift
- ambiguous goals
- legitimate sensitive tasks
- tool abuse

## Run

Fast benchmark without charts:

```bash
python evaluation/aegisbench/run_aegisbench.py --no-charts
```

Full benchmark with charts:

```bash
pip install -e ".[evaluation]"
python evaluation/aegisbench/run_aegisbench.py
```

Run only one side:

```bash
python evaluation/aegisbench/run_aegisbench.py --mode without --no-charts
python evaluation/aegisbench/run_aegisbench.py --mode with --no-charts
```

## Outputs

Reports are written to:

```text
reports/aegisbench_v1/<run_id>/
  run_metadata.json
  without_aegis_traces.jsonl
  with_aegis_traces.jsonl
  case_results.jsonl
  metrics.json
  confusion_matrices.json
  summary_table.csv
  benchmark_summary.md
  charts/
```

## Notes

The default benchmark runner is deterministic and reproducible. It measures AegisVault-style middleware decisions against labelled ground truth, avoiding Qwen sampling variance so the benchmark remains stable for regression testing.
