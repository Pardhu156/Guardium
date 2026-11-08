# Remaining 78 Cases: Analysis and Recovery

**Outcome update, 9 November 2026:** all 78 cases in the recovery manifest now
have evaluated outcomes. The recovery batch recorded 43 task successes and no
successful attacks or unresolved runtime failures. Together with the original
968 outcomes, protected coverage is 1,046/1,046. See the
[final aggregate snapshot](snapshots/2026-11-09/summary.json). The analysis and
commands below document the earlier diagnosis and reproduction procedure.

At the **5 November 2026** review, the newest local result records dated
from 29 October. At that point, no later run had resolved these failures.
No historical outcomes, labels, policies, thresholds, or published snapshots
were edited during this review.

## What actually ran

- 1,046 unique saved case records: 97 clean and 949 attacked combinations.
- 968 evaluated outcomes: 97 clean and 871 attacked combinations.
- 78 unresolved outcomes, all Workspace attacked cases.
- One observed successful attack among the evaluated attacks; unknown cases
  must not be reported as successful defenses.

The 78 cases are not 78 never-started tasks. Their latest records are failures.
Some have been attempted repeatedly, and the last run was interrupted during
`workspace_direct_user_task_26_injection_task_12`. Successful evaluated cases
should not be rerun in the original folder.

## Findings

| Latest recorded problem | Cases | Evidence and interpretation |
|---|---:|---|
| Actor request timeout | 58 | Many fail in generation 2/3 after about 29-31 KB of file results; recent requests reached the 600-second read budget |
| Actor connection failure | 10 | A connection failure was followed by rapid failures on subsequent cases; the runner previously kept trying despite the unavailable server |
| Malformed tool-call output | 10 | Saved responses contain an extra closing JSON brace after an otherwise complete object, often for file creation/appending or email sending |

The largest unresolved groups are Workspace user tasks 26, 35, and 38 (14 each),
27 and 34 (12 each). Smaller groups are tasks 13 (4), 29 (3), 36 (2), and 19,
23, 37 (1 each). These IDs are diagnostic grouping, not special-case runtime
rules; fixes must apply equally to all suites and both phases.

The old buffered transport provides no time-to-first-output measurement. It
cannot prove whether a timeout came from prompt evaluation, slow output
generation, queueing, or a stalled server. Large file results and CPU-only
historical inference explain the workload, not a proven root cause for every
timeout. Do not shorten benchmark tool results, hide injection text, or assume
that another timeout increase alone solves the problem.

An additional correctness bug was found: `--tool-output-format json` was wired
only to the baseline. The protected executor now receives the same existing
formatter. Default YAML remains unchanged, and all content remains present.

## Changes and boundaries

1. Optional streamed actor output (`--stream-model-output`). Full text is
   assembled before parsing. Character counts, request size in characters,
   and time to the first content chunk are saved. Partial/interrupted streams
   are saved for diagnosis but never parsed as completed tool actions. Missing
   finish markers and truncation remain runtime failures.
2. Optional one-generation format recovery (`--format-repair-attempts 1`). The
   malformed proposal does not execute. The same actor receives its failed
   proposal plus fixed formatting feedback and produces a new proposal. The
   strict parser remains unchanged, and the new proposal still requires Layer
   0, Sentinel, and Action Gate. A second malformed result ends the attempt.
   Connection errors, timeouts, and truncated generations are not automatically
   replayed; this avoids accidental side-effect retries.
3. Fail-fast on actor connection loss. The current failed attempt is persisted
   as unknown and the run exits with code 2. Subsequent cases remain untouched.
4. Shared output formatting in both phases, with a regression test proving
   structured data and long untrusted content are preserved.
5. New options are saved in metadata and per-attempt records. Existing defaults
   stay buffered with zero format-regeneration attempts. The optional format
   retry adds a model request and can change the agent trajectory; it is an
   execution protocol change, not a metric improvement already demonstrated.

No production middleware source code changed. The semantic Request/Response
Gates remain bypassed only in the existing AgentDojo path. Immutable Goal Vault,
production MiniLM, Sentinel, Action Gate, and risky-tool verification are retained.
No tool is executed by the transport/parser recovery code itself.

## Run one difficult case first

No full live benchmark was started by this review. Ollama was unavailable on
localhost:11434 when checked, so real-case recovery still needs validation.

Terminal 1, if Ollama is not already running:

```zsh
OLLAMA_CONTEXT_LENGTH=16384 ollama serve
```

Terminal 2, from the repository root:

```zsh
TOKENIZERS_PARALLELISM=false LOCAL_LLM_PORT=11434 .venv/bin/python evaluation/agentdojo/run_pilot_benchmark.py \
  --clean-limit 97 --attack-limit 949 \
  --balanced-by-suite --selection-strategy first --case-layout interleave-types \
  --case-ids workspace_direct_user_task_26_injection_task_0 \
  --phase protected --stream-model-output --format-repair-attempts 1 \
  --llm-timeout-seconds 1800 --case-timeout-seconds 3600 --heartbeat-seconds 30 \
  --output-dir evaluation/agentdojo/results/recovery_78_stream_v1
```

This uses a new result folder, preserving the original 968 evaluated outcomes.
An explicit larger actor timeout permits longer prompt evaluation; it does not
change the 120-second Action Gate verification budget or any security threshold.
It can take longer, not faster. If no characters arrive for a long time, it is
still waiting for output; rising character counts demonstrate generation
activity, not task success. A future context overflow still stops rather than
silently truncating the conversation.

## Then attempt the 78-case manifest

Only proceed after inspecting the smoke result. Reusing the new folder preserves
an evaluated smoke outcome; unresolved attempts need `--retry-errors`:

```zsh
TOKENIZERS_PARALLELISM=false LOCAL_LLM_PORT=11434 .venv/bin/python evaluation/agentdojo/run_pilot_benchmark.py \
  --clean-limit 97 --attack-limit 949 \
  --balanced-by-suite --selection-strategy first --case-layout interleave-types \
  --case-ids-file evaluation/agentdojo/snapshots/2026-10-29/unresolved_cases.json \
  --phase protected --stream-model-output --format-repair-attempts 1 \
  --llm-timeout-seconds 1800 --case-timeout-seconds 3600 --heartbeat-seconds 30 \
  --output-dir evaluation/agentdojo/results/recovery_78_stream_v1 \
  --resume --retry-errors
```

The initial bar may be `new cases` because this is a fresh experimental folder;
it is not rerunning the original successful 968 cases. Press Ctrl+C once to
interrupt. Do not start a second process in the same folder. On connection loss,
restore Ollama first; then rerun the same command. Offline flags are optional
only once the full embedding model is cached.

Inspect without loading models:

```zsh
.venv/bin/python evaluation/agentdojo/inspect_run.py evaluation/agentdojo/results/recovery_78_stream_v1
```

Keep raw model events/attempts local, including `model_request_started`,
`generation_progress`, `model_partial_response`, `model_response`, and
`format_retry`. Compare evaluated coverage, remaining unknowns, security
decisions, and latency, not just an exit code. This issue subset is selected
for failures and cannot establish general utility, ASR, or production readiness.
Do not merge its outcomes into the historical run automatically or claim a
controlled before/after result across differing protocols or hardware.

## Validation performed

Tests exercise streaming via the real OpenAI SDK against an in-process HTTP
mock, incomplete streams, timeouts/interruption, single bounded format retry,
connection-loss persistence/stop, lossless shared formatting, and real adapter
tool authorization with fake models. A regenerated allowed tool executes once;
a regenerated blocked tool never executes. These validate code behavior, not
live model recovery or future attack success rates.

Full suite after changes: **322 passed, 3 skipped** (previous baseline: 305
passed, 3 skipped). `git diff --check` passed. No new live-model benchmark was
run, so the saved coverage and attack metrics have not changed.
