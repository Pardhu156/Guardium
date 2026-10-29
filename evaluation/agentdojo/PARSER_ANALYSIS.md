# AgentDojo Tool-Call Parsing Analysis

Inspected on 2026-10-23. Result folder:
`evaluation/agentdojo/results/agentdojo_stageA_20clean_10attack_fix3`.

## Evidence

At inspection there were **789 unique saved rows**, of which **80** were runtime
failures or skips:

| Recorded failure | Cases |
|---|---:|
| Generated function call could not be parsed | 73 |
| Qwen HTTP timeout | 6 |
| Historical manual skip | 1 |

All 73 parsing failures happened at Qwen generation 1, before any tool execution
in those attempts. The failures were on attack combinations, not the 97 clean
tasks. The same user task is repeated with multiple injection tasks, explaining
clusters such as Workspace user tasks 27 and 35. Repeated first-step failures
do not establish that the injected text was encountered: these attempts never
reached a tool that could return it.

The installed AgentDojo 0.1.35 `local_llm` parser expects JSON between
`<function=name>` and `</function>`. It logs `broken JSON: ''` for an empty call
body. Workspace tasks such as listing drive files legitimately use `list_files`
with no arguments, but `<function=list_files></function>` fails that parser.

The previous harness correctly surfaced malformed calls as runtime failures,
but had two diagnostic gaps:

1. It raised before the outer observer persisted the failed raw model response.
2. Its broad `<function` substring check could classify a final answer merely
   mentioning a literal `<function>` tag as a malformed action.

Both gaps are fixed. Historical rejected raw completions cannot be reconstructed
from the old event logs. Therefore **73 is not a count of proven repairable
outputs**, and no historical rows have been relabelled or rescored.

## Changes

The isolated benchmark parser revision is `schema-envelope-v1`.

| Input form | Handling |
|---|---|
| Strict valid function call and JSON arguments | Preserve it |
| Empty closed/self-closing call | Supply `{}` only if the registered tool's actual Pydantic schema accepts `{}` |
| JSON code fence inside a call | Remove the fence without changing arguments |
| Valid JSON followed by one extra XML `>` delimiter | Normalize the call envelope |
| Valid JSON and missing closing function tag | Normalize the envelope |
| Missing required values | Reject; do not invent values |
| Extra JSON braces, trailing commas, incomplete values | Reject |
| Duplicate keys, NaN/infinity, overflow, non-object arguments | Reject |
| Multiple proposed calls | Reject ambiguity rather than silently choosing one |
| Ordinary final answer mentioning `<function>` | Preserve as a final answer |
| Timeout or empty whole-model answer | Remain runtime failures |

There is **no additional LLM repair request or retry**, no tool-name/argument
substitution, and no automatic execution in the parser. The official AgentDojo
parser receives the normalized envelope. Every actual proposed action still
passes through the existing Layer 0, Sentinel, and Action Gate pipeline.
Valid unknown-tool calls are left for Layer 0 to reject; the parser does not
register them or bypass security checks.

Raw completions are now persisted **before** parsing, with finish reason and
available token usage. Repairs and failures have explicit events. The parser
revision and repair counts are saved per case and in resume metadata so the
changed execution protocol is visible in the mixed historical/resumed run.
Baseline and protected runs use the same parser adapter.

## Verification and limits

Tests use deterministic model-response fixtures, plus real AgentDojo tool
schemas. They cover value preservation, required/unknown arguments, duplicate
keys, malformed JSON, raw-output persistence, and a repaired action still being
blocked by Layer 0 before execution.

Verification completed: **280 passed, 3 skipped** in the full suite; **61 passed**
in the focused parser/reliability/pipeline suite. The actual Workspace
`list_files` and Banking `get_user_info` schemas both accepted the empty-object
normalization in an offline schema check. CLI help and `git diff --check` passed.
No production middleware module, policy, threshold, benchmark label, or dataset
case was edited, and no Git commit was made.

Ollama was stopped at inspection, so no live model or full benchmark was run.
The parser fix does not repair Qwen reasoning, calculations, or timeouts, and
cannot guarantee that utility or ASR will improve. Recovered syntax may let more
proposals reach the security layers; measure their outcomes rather than assuming
the previous ASR remains unchanged. Report parser revisions separately where
possible. Earlier valid results are preserved and saved failures are retried
only when explicitly requested.

## Manual validation

Terminal 1:

```bash
OLLAMA_CONTEXT_LENGTH=16384 ollama serve
```

Terminal 2, from the repository root:

```bash
TOKENIZERS_PARALLELISM=false LOCAL_LLM_PORT=11434 \
  .venv/bin/python evaluation/agentdojo/check_tool_call_format.py \
  --suite workspace --user-task user_task_35
```

This makes one initial Qwen request with the actual suite tool schemas, saves
its raw output and parse result in `evaluation/agentdojo/results/format_check`,
and executes **zero tools**. It is a format diagnostic, not a security benchmark
or utility result. Look for `status: parsed` and `tools_executed: 0`.

Then resume the benchmark:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false LOCAL_LLM_PORT=11434 \
.venv/bin/python evaluation/agentdojo/run_pilot_benchmark.py \
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

Previously evaluated cases are preserved. Never-saved cases run first; saved
errors/skips get one attempt each at the end. The 73 old parse failures are not
retroactively repaired; their new attempts use the updated adapter.

## Follow-up: context capacity after a successful repair

The subsequent live trace for Workspace user task 35 confirmed that the empty
`list_files` call was repaired and executed through the normal gates. The first
model request used 4,597 prompt tokens. Its tool result contained 28,277
characters, expanding the next prompt to 13,651 tokens; other attempts reported
14,027, 13,701, and 13,951 tokens. Ollama's allocated context was 8,192.
This is a serving-capacity error after parsing succeeds, not a parser failure
or a middleware block. The model's locally reported maximum context metadata
was larger than the configured serving allocation.

The harness now recognizes the structured `exceed_context_size_error`, saves
the case with unknown outcomes and its prompt/context counts, and stops before
running further cases under the inadequate configuration. Restarting Ollama
with 16,384 tokens accommodates the observed prompts, with additional memory
cost. Future larger prompts may require another explicit increase. Model
weights, security thresholds, prompts, cases, and injected content are unchanged.
Record the mixed serving configuration when reporting historical/resumed results.

Post-context-fix verification: **285 passed, 3 skipped**. Regression tests cover
the actual nested HTTP error format, unknown outcome persistence, stopping before
later cases, and a clear configuration-error exit. These tests also caught a
checkpoint-creation regression in a brand-new run directory; it was fixed and
covered by a separate test. The user's existing result file was unaffected.
No live larger-context benchmark was started automatically.
