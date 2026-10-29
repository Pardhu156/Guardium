# AgentDojo Coverage and Open Runtime Issues

## Verified snapshot

Snapshot date: **29 October 2026**. Source results last updated at approximately
00:35 IST. This is a protected-only local measurement with Qwen, not a completed
matched full baseline comparison or a production-effectiveness claim.

| Outcome | Count |
|---|---:|
| Unique saved cases | 1,046 |
| Evaluated outcomes | 968 |
| Runtime failures / unknown outcomes | 78 |
| Clean tasks evaluated | 97 / 97 |
| Attacked combinations evaluated | 871 / 949 |
| Successful attacks among evaluated cases | 1 / 871 (0.115%) |
| Clean user tasks completed | 38 / 97 (39.18%) |
| Candidate middleware false positives | 5 / 97 (5.15%) |
| Candidate model/evaluator utility failures | 54 / 97 (55.67%) |

"Evaluated" means AgentDojo returned an outcome, not that the user's task
succeeded. The FP attribution is a block-trace heuristic, not verified causality.
Correct security blocks must not be counted as FPs just because a clean task
failed. Model and evaluator failures are not automatically middleware defects.

All unresolved cases are in Workspace. The other suites have evaluated outcomes:

| Suite | Saved | Evaluated | Unresolved |
|---|---:|---:|---:|
| Workspace | 600 | 522 | 78 |
| Slack | 126 | 126 | 0 |
| Banking | 160 | 160 | 0 |
| Travel | 160 | 160 | 0 |

The [public summary](snapshots/2026-10-29/summary.json) is machine-readable.
The [reproduction manifest](snapshots/2026-10-29/unresolved_cases.json) lists
every unresolved case ID, suite, task/injection IDs, and error category. Raw
model messages, proposed arguments, error bodies, logs, and local paths are
not included. Original local result files remain ignored by Git.

## Open issues

| Recorded category | Cases | What contributors should investigate |
|---|---:|---|
| Qwen request timeout | 58 | Model server responsiveness, memory pressure, prompt/context size, and generation time |
| Model connection error | 10 | Whether Ollama exited, restarted, disconnected, or was unavailable during the request |
| Malformed tool-call output | 10 | Extra content after JSON; reproduce and inspect raw model output before considering a parser change |

These are observed error categories, not proven root causes. A timeout does not
prove an attack was blocked. A connection error does not prove a benchmark case
is incompatible. Earlier retries recovered other cases, but recovery is not
guaranteed and later attempts can produce different outputs.

The 78 unknown attack outcomes are excluded from conditional ASR. Attack outcome
coverage is **871/949 = 91.78%**. Over all 949 attempted attacks, observed success
is a lower bound of **1/949 = 0.105%**; if every unknown succeeded, the upper
bound would be **79/949 = 8.325%**. Neither bound assigns labels to unknown cases.
Do not call the other 870 evaluated attacks "detected by AegisVault" without
attribution: some may have failed because of the actor's own behavior.

Current `IMP` and `Tool ASR` fields alias the same AgentDojo success flag. They
are not independently measured attack classes. Latest-per-case outcomes after
retries are also not independent statistical trials. Serving context and parser
handling changed during recovery; retain resume history and describe that drift.

## Local configuration

- Python 3.12, Intel macOS, CPU-only Ollama in the source run.
- AgentDojo package 0.1.35; suite version `v1.2.2`; attack `direct`; seed 7.
- Actor: `qwen3:4b-instruct`, temperature 0, no explicit output token cap.
- Serving context: 16,384 tokens during recent recovery, not all historical attempts.
- Verifier: `llama3.2`; production `all-MiniLM-L6-v2`, 384 dimensions, L2-normalized goals, cosine similarity.
- Layer 0 -> Goal Vault -> agent -> Layer 0 tool validation -> Sentinel -> Action Gate -> AgentDojo tool execution/evaluation.
- Goal Vault backend in this harness: **in-memory**, not Redis. Semantic Request
  and Response Gates are bypassed in this adapter only. No Gmail, bank, or travel
  accounts are connected; tools operate on AgentDojo's benchmark environment.
- The existing Action Gate verification budget is 120 seconds; Qwen request
  budget 600 seconds; whole-case deadline 3,600 seconds. Failures stay unknown.
- Suite policies are in `evaluation/agentdojo/policies/`. Security thresholds,
  risky-tool verification, and policy behavior were not changed for publication.

Exact scores need not reproduce across hardware, library/model revisions, or
retries, even at temperature 0. This is not the entirety of all AgentDojo attack
strategies: counts refer to this installed benchmark version and direct-attack selection.

## Install and verify

Run from the repository root. Python 3.11 or 3.12 is needed for this embedding
stack; the original evaluation environment used Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[runtime,dev]" -r evaluation/agentdojo/requirements.txt
```

On **Intel macOS / Python 3.12**, reproduce the verified older dependency set
using this install command instead of the last line above:

```bash
python -m pip install -c evaluation/agentdojo/constraints-intel-macos.txt -e ".[runtime,dev]" -r evaluation/agentdojo/requirements.txt
```

The constraints are a historical compatibility aid, not a full dependency lock
or recommendation to deploy old ML dependencies in production. Do not force
them onto unsupported platforms or newer Python versions. No global/Homebrew
packages are installed by these commands. Install Ollama separately if needed.

Terminal 1 (do not start a duplicate existing server):

```bash
OLLAMA_CONTEXT_LENGTH=16384 ollama serve
```

Terminal 2:

```bash
ollama pull qwen3:4b-instruct
ollama pull llama3.2
curl -fsS http://localhost:11434/api/tags
python -c 'from aegisvault.runtime.goal_vault import SentenceTransformerGoalEmbedder; e=SentenceTransformerGoalEmbedder(model_name="all-MiniLM-L6-v2", expected_dimension=384); print(len(e.embed("Verify local embedding setup")))'
python -m pytest -q
```

The embedding check should print `384`; it can download the embedding model on
the first run. Do not set offline flags until the complete model is cached.
The runner performs actor/verifier model preflight and validates the production
embedder. After loading, `ollama ps` shows the allocated context. A larger future
conversation can still exceed it; do not silently truncate benchmark input.

## Reproduce one unresolved case

Start with one case rather than spending hours on a full rerun. This executes
the genuine protected pipeline with no weakened checks or hardcoded answers:

```bash
TOKENIZERS_PARALLELISM=false LOCAL_LLM_PORT=11434 .venv/bin/python evaluation/agentdojo/run_pilot_benchmark.py \
  --clean-limit 97 --attack-limit 949 \
  --balanced-by-suite --selection-strategy first --case-layout interleave-types \
  --case-ids workspace_direct_user_task_26_injection_task_0 \
  --phase protected --llm-timeout-seconds 600 --case-timeout-seconds 3600 \
  --output-dir evaluation/agentdojo/results/my_workspace_repro
```

Substitute another exact manifest ID to investigate another case. Use a new
output directory for a fresh configuration. Existing evaluated results in the
same directory are skipped by default. Do not use `--skip-preflight` to hide
setup problems. To test the initial tool-call envelope without executing tools:

```bash
TOKENIZERS_PARALLELISM=false LOCAL_LLM_PORT=11434 .venv/bin/python evaluation/agentdojo/check_tool_call_format.py --suite workspace --user-task user_task_26
```

## Reproduce the 78-case issue set

Contributors do not need the author's private/raw result folder. The published
manifest selects exactly these cases from the installed AgentDojo suites:

```bash
TOKENIZERS_PARALLELISM=false LOCAL_LLM_PORT=11434 .venv/bin/python evaluation/agentdojo/run_pilot_benchmark.py \
  --clean-limit 97 --attack-limit 949 \
  --balanced-by-suite --selection-strategy first --case-layout interleave-types \
  --case-ids-file evaluation/agentdojo/snapshots/2026-10-29/unresolved_cases.json \
  --phase protected --llm-timeout-seconds 600 --case-timeout-seconds 3600 \
  --output-dir evaluation/agentdojo/results/my_unresolved_repro
```

Press Ctrl+C once to interrupt. Rerun the same command with `--resume --retry-errors`
to preserve evaluated outcomes and retry unresolved records once per launch.
The in-progress case restarts; successful cases do not. The new/retry bars show
attempt progress, not a guarantee of successful evaluation. These 78 cases are
an issue-reproduction subset, **not an unbiased replacement benchmark**.

For a new matched comparison use `--phase both --order baseline-first` with the
same chosen IDs/configuration and a new output directory. Run the full selected
pool by omitting the case-ID filter. Do not mix changed settings into historical
results and then claim a controlled before/after comparison.

## Inspect and contribute a fix

```bash
.venv/bin/python evaluation/agentdojo/inspect_run.py evaluation/agentdojo/results/my_unresolved_repro
```

Inspect `protected_metrics_live.json`, `protected_summary_live.md`,
`protected_events.jsonl`, `protected_attempts.jsonl`, and the current checkpoint.
Final reports are `metrics.json` and `comparison_report.md`; they may be stale
until the phase completes. A fresh heartbeat proves liveness, not task success.

A useful issue or pull request includes the case ID, model/library versions,
context budget, command, failure category, and a reviewed minimal raw-response
excerpt if safe to share. Never post credentials, environment dumps, real tool
payloads, or uncontrolled logs. Keep raw artifacts local unless explicitly reviewed.
Separate infrastructure fixes, parser-envelope fixes, model utility failures,
and middleware decision bugs. Include a regression test with a fake model.

Do not fix malformed arguments by guessing values, dropping extra JSON blindly,
skipping Sentinel/Action Gate, relaxing thresholds, changing labels, or removing
difficult cases. Risky tools still require the full existing verification path;
BLOCK/JUSTIFY must never execute automatically. Validate any proposed change on
both clean and attacked cases before claiming improved utility or security.

Export a new reviewed aggregate snapshot without loading models:

```bash
.venv/bin/python evaluation/agentdojo/export_snapshot.py evaluation/agentdojo/results/my_unresolved_repro --output-dir evaluation/agentdojo/snapshots/my_new_snapshot
```

The exporter excludes raw content, retains unknown-outcome coverage, and refuses
to overwrite an existing snapshot. Publication does not imply that all issues
are fixed or that future benchmarks will achieve the same scores.
