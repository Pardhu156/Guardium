# Long-Run Reliability Review

Reviewed on 2026-10-23 without stopping the active protected benchmark.
Result folder: `evaluation/agentdojo/results/agentdojo_stageA_20clean_10attack_fix3`.

## Confirmed remaining issues

At the start of this review, 798 unique cases were saved: 711 had evaluator
outcomes and 87 were errors/skips. Their recorded causes were:

| Cause | Cases | Assessment |
|---|---:|---|
| Historical first-response parser failures | 73 | Deferred for retry; original raw outputs unavailable |
| Qwen request timeouts | 7 | Model/server execution failures, not successful defenses |
| Earlier 8192-token context errors | 4 | Saved before the server was increased to 16384 |
| New malformed tool-call JSON | 2 | Actual extra JSON closing braces; correctly rejected |
| Historical manual skip | 1 | Unknown outcome |

The two newly captured malformed outputs were in Workspace user task 29,
injection tasks 1 and 3. Both contained an `append_to_file` call whose decoded
argument object was followed by an extra `}`. The parser deliberately retains
this as a failure instead of altering or guessing the proposed action.

The Mac had approximately 9 GB of occupied swap, and Qwen was using CPU only.
This may contribute to long requests; occupied swap alone does not prove active
paging. Close unused memory-heavy applications, while keeping Ollama and the
benchmark terminal running. More complex conversations can still exceed the
configured context or time budgets. A 16384-token context is not a guarantee
that all future tasks fit.

## Reporting corrections

1. The terminal bar rounded a small nonzero ASR to `0%`. It now uses two decimal
   places and shows `n/a` when no evaluated attacks exist.
2. The periodic ETA divided the current session's elapsed time by completed
   cases from older sessions. It now uses only attempts in the current session,
   including pending retries. It remains an estimate across variable workloads.
3. `metrics.json` and `comparison_report.md` were old 157-case reports because
   final reports are regenerated only when a phase finishes. During a run,
   use `protected_metrics_live.json`. The runner now also refreshes
   `protected_summary_live.md` after each case, without replacing an old
   baseline/protected comparison with an incomplete comparison.
4. Unknown attack outcomes are now explicitly counted, with attack evaluation
   coverage and lower/upper ASR bounds over attempted attacks. These bounds
   assume respectively none/all of the unknown attacks succeeded; they do not
   relabel failures. The standard ASR remains conditional on evaluated attacks.

The existing automatic FP/model-failure split is a trace heuristic, not a
verified causal attribution. Live summary labels now call them candidates.
IMP and Tool ASR remain aliases of the single AgentDojo attack-success result;
they are not independently measured attack types.

## Additional startup safeguards

Protected runs now check that each suite's configured Action Gate model is
installed at its evaluator endpoint, accepting Ollama's `:latest` alias.
Baseline-only runs and the proposal-only format check do not require a verifier.
Non-finite or invalid heartbeat, actor, verifier, and case timeout settings are
rejected before execution. No security threshold, model, prompt, policy, tool
implementation, benchmark case, or label was changed.

## Checks that passed

- Qwen and llama3.2 are installed.
- Qwen is allocated a 16384-token context.
- Saved case IDs are unique; the canonical result file is readable.
- Checkpoints are fresh, and the active case is making tool/model progress.
- Files are small (single-digit MB per main log); approximately 208 GiB of disk
  space was available. No disk-capacity issue was observed.
- The existing lock, atomic checkpoint, raw-output, deferred-retry, and
  context-error stop mechanisms remain in place.

No additional current configuration blocker was found. This does not guarantee
future model generations, context sizes, or infrastructure will succeed.
Review/tests ran alongside the benchmark, so this is not an isolated latency
measurement environment.

The active process was left running and its saved outcomes were not edited.
The reporting/startup changes apply on the next launch; do not restart an active
case solely to pick up these changes. No full benchmark or new live inference
request was started by this review, and no Git commit was made.
