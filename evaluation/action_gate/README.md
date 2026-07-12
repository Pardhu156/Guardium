# Stage 3.3: Goal Vault + Action Gate Evaluation

This folder evaluates Stage 3.1 Goal Vault and Stage 3.2 Action Gate together, without real agents.

It verifies:

- immutable goal commitment,
- duplicate commit rejection,
- goal retrieval and integrity verification,
- cosine shortcut routing,
- Ollama/fake-verifier uncertainty routing,
- `EXECUTE`, `JUSTIFY`, and `BLOCK` behavior,
- fake tool execution safety,
- latency and decision-source metrics.

It does not evaluate Request Gate, Response Gate, Gemini, real agents, LangChain, AutoGen, Sentinel Monitor, or EMA.

## Datasets

JSONL datasets live in:

```text
evaluation/action_gate/datasets/
```

Domains:

- `email_assistant`: 30 cases
- `ecommerce`: 30 cases
- `document_qa`: 30 cases
- `cross_domain`: 20 cases

Each row contains a manually labelled expected verdict.

## Smoke Test

```bash
python evaluation/action_gate/scripts/run_action_gate_evaluation.py \
  --domains email_assistant ecommerce document_qa cross_domain \
  --backend memory \
  --limit 3 \
  --runs 1
```

## Full Evaluation

```bash
python evaluation/action_gate/scripts/run_action_gate_evaluation.py \
  --domains email_assistant ecommerce document_qa cross_domain \
  --backend memory \
  --runs 1
```

## Redis Evaluation

```bash
redis-cli ping
python evaluation/action_gate/scripts/run_action_gate_evaluation.py \
  --domains email_assistant ecommerce document_qa cross_domain \
  --backend redis \
  --runs 1
```

## Outputs

Each run writes:

```text
evaluation/action_gate/results/<run_id>/
├── run_metadata.json
├── case_results.jsonl
├── goal_vault_results.jsonl
├── action_gate_results.jsonl
├── tool_execution_results.jsonl
├── failures.jsonl
├── metrics.json
├── latency_summary.json
└── evaluation_summary.md
```

## Threshold Tuning

```bash
python evaluation/action_gate/scripts/run_action_gate_evaluation.py \
  --low-threshold 0.30 \
  --high-threshold 0.88 \
  --ollama-confidence 0.80
```

- Lower `low-threshold`: fewer direct blocks, more verifier calls, possibly fewer false blocks.
- Higher `high-threshold`: fewer direct executes, more verifier calls, potentially fewer false executes.
- Wider uncertainty band: more verifier calls and higher latency, but more semantic judgment.

The default thresholds are starter values, not production proof.

## Manual Checks

Validate local files:

```bash
python evaluation/action_gate/scripts/validate_environment.py
```

Inspect metrics:

```bash
cat evaluation/action_gate/results/<run_id>/metrics.json | python -m json.tool
```

Inspect latency:

```bash
cat evaluation/action_gate/results/<run_id>/latency_summary.json | python -m json.tool
```

Confirm blocked/JUSTIFY tools did not execute:

```bash
cat evaluation/action_gate/results/<run_id>/metrics.json | python -m json.tool | rg "blocked_tool|justify_auto"
```
