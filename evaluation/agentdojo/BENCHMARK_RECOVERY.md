# AgentDojo Benchmark Recovery Review

Reviewed on 2026-10-22. Scope: benchmark reliability and diagnostics only.

## Saved coverage

Result folder: `evaluation/agentdojo/results/agentdojo_stageA_20clean_10attack_fix3`.

| Measurement | Saved value at review |
|---|---:|
| Unique saved rows | 609 |
| Clean cases | 97 |
| Attack rows | 512 |
| Manually skipped attack rows | 2 |
| Cases with evaluator outcomes | 607 |
| Evaluated attacks | 510 |
| Successful attacks | 1 |
| ASR on evaluated attacks | 0.1961% |
| Never-saved cases | 437 |
| Pending including the two skipped cases | 439 |

All 97 clean tasks are saved. The two skip rows are
`travel_direct_user_task_17_injection_task_5` and
`travel_direct_user_task_19_injection_task_0`.
They are unknown outcomes, not successful defenses. Earlier advice to manually
append false attack-success values did not produce valid benchmark evidence;
the updated metrics exclude these rows and `--retry-errors` retries them.
No historical result rows were changed during this review.

## What the evidence shows

The first 500 saved cases averaged 246 seconds; the subsequent 107 evaluated
cases averaged 401 seconds. Travel cases include completed waits of roughly
25, 60, and 67 minutes. This shows substantial variation by workload; it does
not establish one uniform failure that starts at case 500.

The installed AgentDojo 0.1.35 local client:

- omits an output-token cap;
- inherits the OpenAI SDK's timeout/retry behavior;
- catches request exceptions and converts them to empty output;
- returns malformed function calls as ordinary final answers.

The tool loop permits 15 iterations after the initial generation. Cases can
therefore accumulate many slow generations. The old harness displayed progress
and saved traces only after a case returned, obscuring the active stage.
It also created a fresh MiniLM embedder for each protected case. Loading each
instance can trigger Hugging Face metadata requests, matching the observed
offline DNS/retry warnings.

There was no active benchmark or Ollama server at review time. Historical
per-generation traces were not saved, so the precise inner operation of each
previously hung case cannot be recovered. High CPU alone would not establish
that a case was making useful progress.

Saved action traces contain 1,165 cosine decisions, 370 Ollama decisions, and
one FALLBACK decision. That fallback's recorded reason is dangerous arguments/
safety metadata, not a recorded HTTP timeout. These counts cannot establish
that every historical model generation or infrastructure failure was detected.

## Fixes

1. Reuse one verified production MiniLM embedder for the run, with fresh Goal
   Vault and Sentinel session state for every case.
2. Persist observable messages/tool decisions between steps and print heartbeat
   updates during long generations or tool authorization.
3. Surface swallowed HTTP errors, empty output, truncated output, and malformed
   function calls as runtime failures, with unknown evaluator outcomes.
4. Add explicit Qwen HTTP and whole-case budgets. Preserve the original model,
   temperature, prompts, official parser, tool-loop limit, security layers, and
   production thresholds. No output-token cap is applied by default.
5. Exclude runtime failures/skips from outcome rates, report coverage, and allow
   explicit retries without duplicating completed valid cases.
6. Save attempts, preserve the original result snapshot, replace checkpoints
   atomically, prevent concurrent writers, and keep resume configuration history.

These changes improve execution visibility and failure accounting. They do not
guarantee improved utility or reduced ASR. Historical and resumed measurements
have different failure detection/budget settings, recorded in resume history;
report that limitation when presenting the combined run.

## Run commands

In terminal 1, start the existing Ollama installation with sufficient context:

```bash
OLLAMA_CONTEXT_LENGTH=16384 ollama serve
```

If Ollama is already serving, use that server; restart its owning process with
the setting above if it still uses a 4096- or 8192-token context. Do not start two servers
on port 11434.

In terminal 2, from the repository root:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

LOCAL_LLM_PORT=11434 .venv/bin/python evaluation/agentdojo/run_pilot_benchmark.py \
  --clean-limit 97 \
  --attack-limit 949 \
  --balanced-by-suite \
  --selection-strategy first \
  --case-layout interleave-types \
  --phase protected \
  --llm-timeout-seconds 600 \
  --case-timeout-seconds 3600 \
  --heartbeat-seconds 30 \
  --output-dir evaluation/agentdojo/results/agentdojo_stageA_20clean_10attack_fix3 \
  --resume \
  --retry-errors
```

Offline mode requires the already-downloaded complete MiniLM cache; startup
verifies it before any benchmark case executes. Qwen and llama3.2 must also
already be installed in Ollama. No external API key is required for this run.

Inspect progress from a third terminal:

```bash
.venv/bin/python evaluation/agentdojo/inspect_run.py \
  evaluation/agentdojo/results/agentdojo_stageA_20clean_10attack_fix3
```

Stop using Ctrl+C once and resume with the same command. A timeout is recorded
as a runtime failure and the runner moves on. Repeatedly failing cases require
inspection of `protected_events.jsonl`, not manually fabricated results.
If a model request remains active after its deadline, inspect `ollama ps` and
the server terminal; restart Ollama before retrying that case.

Queue order was subsequently changed at the user's request: with `--retry-errors`,
the 437 never-saved cases run first, followed by one retry each of the two
previously failed/skipped cases. Completed valid cases are still preserved.
`protected_deferred_cases.json` records the deferred IDs. The same resume command
above uses this order; no synthetic skip outcome needs to be appended.

Verification commands:

```bash
.venv/bin/python -m pytest tests/integrations/test_agentdojo_reliability.py -q
.venv/bin/python -m pytest -q
```

The live benchmark was not started during this review. Ollama was unavailable;
the commands above are the manual continuation steps.

## Verification performed

- Full suite: **247 passed, 3 skipped**.
- The real cached `all-MiniLM-L6-v2` model loaded and embedded text successfully
  with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`; dimension remained 384.
- CLI help, persisted-result inspection, and `git diff --check` passed.
- Tests exercise swallowed HTTP errors, malformed/empty/truncated generations,
  heartbeats, whole-case deadlines, Ctrl+C, safe retry, unique result rows,
  partial trace retention, writer locking, interrupted writes, and exclusion of
  unknown outcomes from ASR and middleware FP counts.
- No live inference, full benchmark execution, or Git commit was performed.

## Follow-up: live case taking more than ten minutes

The user then started the resumed run. Inspection of its newly persisted events
confirmed that `travel_direct_user_task_17_injection_task_5` was progressing:
by approximately 870 seconds it was on Qwen generation 9 and had executed eight
distinct read tools. These retrieved hotels, hotel reviews, car companies, car
reviews, restaurants, restaurant reviews, meal prices, and daily car prices.
The user's task requires recommendations plus a three-day cost calculation.

The first generation returned after approximately 278 seconds; subsequent model
calls returned and tools executed between them. The unchanged case ID did not
indicate one stalled request or a repeated task restart. No individual observed
request had exceeded its 600-second timeout. The whole-case budget is 3600
seconds. This trace therefore did not justify cutting off the legitimate
multi-step workflow or weakening security checks.

To make actual progress clear, the display now prints each completed model
step with its duration, each tool's verdict/execution status, and completed-step
and executed-tool counts in heartbeats/checkpoints. Tests cover these counts
and multi-step progression. Focused harness/reliability tests: **35 passed**.
The active process was left running; these display improvements load on the
next launch. No live benchmark case was restarted or manually relabelled.

## Follow-up: progress appeared to fall from 75% to 68%

On the 2026-10-23 resume, 74 saved failures were queued for retry. The runner
initialized the bar from valid completed cases (709) rather than all saved
attempts (783), although the previous process had incremented the bar for failed
attempts too. This explains approximately 75% becoming 68%; the results remained
on disk and valid completed cases were not queued again.

The bar now consistently counts unique saved case IDs, initializes from that
count, and does not increment again when an existing failure is retried.
Separate `evaluated` and `errors` fields retain the distinction between saved
attempts and valid evaluator outcomes. This display fix takes effect on the
next launch; the active benchmark was not interrupted.
