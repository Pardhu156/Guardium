# Sentinel Runtime Monitor

Sentinel is a runtime monitoring module. It estimates whether a running agent is drifting away from the trusted goal stored by Goal Vault.

Sentinel does not:

- authorize actions
- execute tools
- modify Goal Vault
- call Request Gate, Action Gate, or Response Gate
- invoke an LLM
- make network calls

Action Gate remains the authorization component.

## Architecture

```text
Trusted Goal
  -> Reasoning Monitor
  -> Intent Monitor
  -> Action Monitor
  -> Risk Fusion
  -> EMA Drift Tracker
  -> Sentinel Decision
```

## Execution Object

Sentinel consumes one structured execution object instead of raw conversation history.

```python
from aegisvault.sentinel import SentinelExecutionState, ToolCallState

execution = SentinelExecutionState(
    reasoning="Qwen reasoning when available",
    current_intent="Summarize unread customer emails.",
    tool_call=ToolCallState(
        name="read_email",
        arguments={"folder": "inbox"},
    ),
)
```

Reasoning is optional. Sentinel continues with intent and action monitors when reasoning is unavailable.

## Monitors

Reasoning Monitor compares trusted goal to Qwen reasoning when available.

Intent Monitor compares trusted goal to the structured current intent.

Action Monitor normalizes the proposed tool call into text, then compares it to the trusted goal.

All monitors use the existing `GoalEmbedder` abstraction used by Goal Vault. By default, Sentinel uses `SentenceTransformerGoalEmbedder`; tests and applications can inject another `GoalEmbedder`.

## Fusion

Default weights:

- reasoning: `0.20`
- intent: `0.35`
- action: `0.45`

Unavailable monitors are ignored and remaining weights are renormalized.

## EMA

Sentinel maintains per-session EMA drift:

```text
ema = alpha * current_fused_drift + (1 - alpha) * previous_ema
```

Default `alpha` is `0.40`.

EMA state is internal to Sentinel and does not modify Goal Vault.

## Decision Thresholds

Defaults:

- `allow`: `< 0.25`
- `observe`: `0.25 - 0.45`
- `review`: `0.45 - 0.65`
- `block`: `>= 0.65`

The `block` Sentinel decision is a runtime signal only. It is not tool authorization.

## Usage

```python
from aegisvault.sentinel import SentinelMonitor, SentinelExecutionState, ToolCallState

sentinel = SentinelMonitor()

decision = sentinel.analyze(
    session_id="session-123",
    trusted_goal="Summarize unread customer emails.",
    execution=SentinelExecutionState(
        reasoning=None,
        current_intent="Summarize unread customer emails.",
        tool_call=ToolCallState(name="read_email", arguments={"folder": "inbox"}),
    ),
)
```

## Policy Configuration

Sentinel is disabled by default for backward compatibility. Existing policies without a `sentinel` section preserve the Stage 4.2/5.1 path.

```yaml
sentinel:
  enabled: true
  fail_mode: closed
  signals:
    reasoning: true
    intent: true
    action: true
  runtime:
    evaluate_before_every_tool: true
    require_trusted_goal: true
    audit_missing_signals: true
  enforcement:
    block_on_sentinel_block: true
    review_requires_action_gate_verification: true
```

When enabled in the protected tool path:

```text
Layer 0 tool validation
  -> Sentinel evaluation
  -> Action Gate authorization
  -> tool execution
```

Layer 0 blocks short-circuit before Sentinel. Sentinel `block` can prevent automatic tool execution when `block_on_sentinel_block` is true. Action Gate remains the final authorization component for non-blocked Sentinel decisions.

## Performance

Run the deterministic Sentinel micro-benchmark:

```bash
python evaluation/sentinel/benchmark_sentinel.py
```

This benchmark uses an injected fake embedder. It does not measure sentence-transformer model latency.

Run local smoke scenarios:

```bash
python evaluation/sentinel/smoke_sentinel_integration.py
```

## Remaining Runtime Integration Work

Future stages may feed Sentinel decisions into richer dashboards, live Qwen evaluation reports, or AgentDojo-style benchmark comparisons.
