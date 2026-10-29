# AgentDojo Integration

This folder contains the Stage 6.1 compatibility layer for routing AgentDojo
benchmark tasks through AegisVault runtime security.

**Current public snapshot (29 October 2026):** 1,046 saved cases, 968 evaluated,
78 unresolved runtime failures. See [BENCHMARK_STATUS.md](BENCHMARK_STATUS.md)
for exact counts, limitations, installation, and contributor reproduction steps.
The sanitized [snapshot](snapshots/2026-10-29/) includes all 78 unresolved IDs;
it deliberately excludes raw prompts, model responses, arguments, and credentials.

The AgentDojo adapter preserves the production runtime security path:

1. Layer 0 request sanity validation
2. Goal Vault initialization
3. Qwen or AgentDojo agent step
4. Layer 0 tool-call validation
5. Sentinel evaluation
6. Action Gate authorization
7. Tool execution
8. AgentDojo evaluator

The semantic Request Gate and Response Gate are not used for AgentDojo runtime
execution. AgentDojo focuses on tool execution security, so the protected path is
Layer 0, Goal Vault, Sentinel, Action Gate, and then tool execution.

Run the mock compatibility smoke test:

```bash
.venv/bin/python evaluation/agentdojo/smoke_agentdojo_adapter.py
```

Run the protected AgentDojo benchmark sample:

```bash
TOKENIZERS_PARALLELISM=false LOCAL_LLM_PORT=11434 .venv/bin/python evaluation/agentdojo/run_pilot_benchmark.py \
  --clean-limit 97 \
  --attack-limit 60 \
  --balanced-by-suite \
  --selection-strategy first \
  --case-layout interleave-types \
  --phase protected \
  --action-timeout-seconds 120 \
  --output-dir evaluation/agentdojo/results/agentdojo_stageA_20clean_10attack_fix3
```

The runner supports resume by default. Reusing the same output directory skips
completed case IDs and continues the remaining cases.

## Recovering a Long Run

See [BENCHMARK_RECOVERY.md](BENCHMARK_RECOVERY.md) for the diagnosed stalls,
saved coverage, and the full resume command. The model, tool loop, security
policies, embeddings, prompts, and security thresholds remain unchanged.
See [RUN_HEALTH_REVIEW.md](RUN_HEALTH_REVIEW.md) for the follow-up review of
remaining malformed outputs, memory pressure, model availability, and reporting.

For repeated `broken JSON` failures, see [PARSER_ANALYSIS.md](PARSER_ANALYSIS.md).
The isolated `schema-envelope-v1` adapter normalizes empty no-argument calls
only when their real tool schema accepts `{}`, plus unambiguous JSON fences and
XML envelope delimiters. It never invents required values or executes tools.
Raw responses, parser repairs, failure reasons, and parser revision are saved
before failed generations can be discarded. There are no extra model repair
calls. All proposed tool actions still pass the existing security pipeline.

Check one real initial proposal with zero tool executions before a long resume:

```bash
TOKENIZERS_PARALLELISM=false LOCAL_LLM_PORT=11434 \
  .venv/bin/python evaluation/agentdojo/check_tool_call_format.py \
  --suite workspace --user-task user_task_35
```

This requires Ollama and saves `format_check.json` and raw model response events
under `evaluation/agentdojo/results/format_check`. It is not a benchmark score.

Large tool results can exceed an 8192-token serving context. A repaired
`list_files` call returned approximately 28,000 characters and expanded the
next observed request to 13,651 tokens. Restart the Ollama server with
`OLLAMA_CONTEXT_LENGTH=16384 ollama serve` for that input, then resume. Verify
the allocated context with `ollama ps` after the model loads. This setting
changes the serving budget and increases memory use; it does not change model
weights or middleware thresholds. It does not guarantee every larger future
conversation will fit.

Context-limit errors now save the affected case as an unknown runtime outcome
and stop the benchmark with exit code 2, preserving later cases for resume.
No tool output, conversation, or injected content is silently truncated.
The currently loaded Ollama context is recorded in metadata when discoverable;
it may be unknown at startup if the model has not loaded yet. Failed requests
record their exact prompt/context counts when provided by the server.

The runner now reuses the verified production MiniLM instance across cases.
It prints a heartbeat every 30 seconds with the active case, stage, stage time,
and total case time. `protected_current_case.json` records the latest checkpoint;
`protected_events.jsonl` persists observable messages and tool decisions during
each case, including interrupted cases. Checkpoint timestamps are historical;
they do not prove the process is still running.

Completed model steps and tool decisions are also printed explicitly. A case
ID remains the same throughout its multi-step workflow. For example, Qwen
generation 9 means eight model steps have already returned; it does not mean
the first request has been waiting that long. The Qwen timeout applies to each
HTTP request, while the separate whole-case deadline covers all steps together.
Display updates take effect when the runner is next launched.

Progress bars count attempts remaining at launch, separately for `new cases`
and `retries`. Each starts at zero and advances after an attempt is saved,
whether it succeeds or fails. Once new cases finish, a separate retry bar opens.
On resume, successful saved cases are excluded from both bars; unresolved
errors are queued only with `--retry-errors`. The startup summary still reports
unique saved cases, and `evaluated` and `errors` show outcome counts separately.
Thus 100% on a retry bar means every scheduled retry was attempted, not that
every case succeeded. An interrupted case restarts on the next launch.

Additional runtime controls:

| Flag | Default | Meaning |
|---|---:|---|
| `--heartbeat-seconds` | 30 | Progress update interval inside a case |
| `--llm-timeout-seconds` | 600 | Qwen HTTP timeout; automatic SDK retries disabled |
| `--case-timeout-seconds` | 3600 | Whole-case POSIX deadline; `0` disables it |
| `--retry-errors` | off | Run unfinished cases first, then retry saved errors/manual skips once at the end |
| `--max-output-tokens` | unset | Optional generation cap; changes the decoding budget |
| `--case-ids ID ...` | unset | Restrict selected cases to exact IDs for isolated reproduction |
| `--case-ids-file PATH` | unset | Restrict selected cases using a JSON `case_ids` manifest |

Explicit ID filters run after normal case selection, preserve benchmark order,
and reject unknown IDs. For the published unresolved manifest, select the full
97-clean/949-attacked pool first; do not combine it with a small `--limit`.
Use a **new output directory** for contributor reproduction so it cannot overwrite
or masquerade as the historical full run. The filter is recorded in metadata;
the default selection and middleware path are unchanged.

These are maximum waiting budgets, not minimum runtimes. The HTTP timeout
limits network inactivity; the case deadline also bounds accumulated generations
and verifier calls. Python delivers POSIX deadlines when native code returns
control, so this is not an operating-system process kill. A timed-out request
may continue computing on the model server until the server notices cancellation.
The existing Action Gate timeout remains 120 seconds and follows its configured
fail-closed behavior; it is separate from the Qwen/case budgets.

Runtime errors, malformed model calls, empty answers, truncated generations,
and historical manual skips are recorded as unevaluated outcomes. They are
excluded from ASR and utility denominators and reported separately with coverage.
They are never counted as successful defenses or clean-task middleware FPs.
Clean-task security interventions remain candidates for manual FP attribution:
the presence of a block alone cannot prove that the proposed action was legitimate.
IMP and Tool ASR in the existing report are aliases of AgentDojo's single returned
attack-success flag, not independently measured attack types.

Completed valid cases are preserved during resume. Retried attempts go into
`protected_attempts.jsonl`; the authoritative result file has one latest row per
case ID. A pre-recovery backup is saved on first resume. Checkpoint replacement
is atomic, and a run-folder lock prevents concurrent writers. A truncated final
JSONL line is backed up and recovered; corruption inside the file is rejected.
Resume settings are appended to `resume_history.jsonl` for reproducibility.
Historical empty/malformed generations cannot be reclassified without traces;
the improved diagnostics apply to subsequent cases.

While a phase is running, read `protected_metrics_live.json` and (after the
updated runner is launched) `protected_summary_live.md`. The final `metrics.json`
and `comparison_report.md` may still describe an earlier completed run until
the phase finishes. The live ASR is conditional on evaluated attacks; JSON also
includes unknown attack counts, attack coverage, and lower/upper ASR bounds over
attempted attacks. These bounds do not treat unknown outcomes as ground truth.
Terminal ASR uses two decimal places. The periodic ETA uses current-session
attempts, including retries, rather than treating previously saved cases as
having executed in the current session.

Known failures/manual skips are deferred behind all unfinished cases when
`--retry-errors` is used. Their IDs are recorded in
`protected_deferred_cases.json`. This scheduling deferral does not change
their evaluator outcomes or count them as blocked attacks. Newly failed cases
are saved for a subsequent explicit retry rather than retried indefinitely.

Inspect progress without loading models or using the network:

```bash
.venv/bin/python evaluation/agentdojo/inspect_run.py \
  evaluation/agentdojo/results/agentdojo_stageA_20clean_10attack_fix3
```

Press Ctrl+C once to stop. Finished cases remain saved, partial events remain
available, and the active interrupted case stays pending. Run the same command
again to resume. Use `--retry-errors` to revisit recorded runtime failures.

## Earlier 157-Case Snapshot

This is the historical 157-case snapshot, not the current full-run coverage.
The same result folder has since been extended; use `inspect_run.py` for counts.

Result folder:

```text
evaluation/agentdojo/results/agentdojo_stageA_20clean_10attack_fix3
```

Coverage:

```text
157 total cases
97 clean AgentDojo user tasks
60 injected attack cases
```

Metrics:

```text
Clean utility success: 39.18%
Attack Success Rate / IMP / Tool ASR: 0.00%
Middleware false positives: 5 / 97 = 5.15%
Model utility failures: 54 / 97 = 55.67%
False negatives: 0
```

Per-suite utility on the 157-case protected run:

```text
Workspace: 52.73%
Slack:     61.11%
Travel:    45.71%
Banking:   22.58%
```

## Metric Separation

Do not treat every failed clean AgentDojo task as an AegisVault false positive.

Middleware false positives count only cases where AegisVault incorrectly blocks
or modifies a legitimate clean action, such as a Sentinel or Action Gate block on
a goal-aligned tool call.

Model utility failures are separate. They include wrong dates, wrong tool
arguments, incomplete retrieval, arithmetic mistakes, duplicate tool calls, and
AgentDojo exact-match failures after AegisVault allowed execution.

This distinction matters because AegisVault is the security middleware around
the agent. It does not guarantee that the underlying local Qwen model completes
every AgentDojo clean task correctly.

Manual AgentDojo setup is required before running real Workspace, Slack,
Banking, and Travel suites. Follow [BENCHMARK_STATUS.md](BENCHMARK_STATUS.md).
