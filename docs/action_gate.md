# AegisVault Stage 3.2: Action Gate

The Action Gate protects tool execution. Before a tool runs, AegisVault checks whether the proposed action aligns with the immutable original goal stored in the Goal Vault.

The gate must never evaluate only the tool name. Its decision uses:

- immutable goal anchor: original goal, normalized goal, goal embedding,
- proposed tool action: name, description, arguments,
- tool metadata: risk level, allowed domains, required permissions, side-effect level,
- runtime context: reasoning summary, previous approved action, session metadata,
- current application policy.

## Decision Flow

```text
Proposed tool action
    ↓
Retrieve GoalAnchor from Goal Vault
    ↓
Verify SHA-256 integrity
    ↓
Build action text from goal + action + metadata + context + policy
    ↓
Embed action text
    ↓
Cosine similarity with goal embedding
    ↓
High similarity  -> EXECUTE, no Ollama
Low similarity   -> BLOCK, no Ollama
Uncertain band   -> Ollama verifier
    ↓
EXECUTE / JUSTIFY / BLOCK
```

`JUSTIFY` pauses execution. The protected tool is not called automatically.

## Why Cosine Similarity

Cosine similarity gives a deterministic fast path. Clearly aligned actions can execute without an LLM call, and clearly misaligned actions can be blocked without an LLM call. Ollama is reserved for the uncertainty band where semantic and policy context need deeper inspection.

## Thresholds

Configure thresholds with `ActionGateConfig`:

```python
from aegisvault.runtime.action_gate import ActionGateConfig

config = ActionGateConfig(
    high_similarity=0.82,
    low_similarity=0.35,
    minimum_llm_confidence=0.75,
)
```

- `high_similarity`: similarity at or above this returns `EXECUTE`.
- `low_similarity`: similarity at or below this returns `BLOCK`.
- `minimum_llm_confidence`: low-confidence Ollama decisions become `JUSTIFY`.

Tune thresholds per application, tool inventory, and risk tolerance.

## Basic Usage

```python
from aegisvault.runtime.action_gate import ActionGate, ToolMetadata, SideEffectLevel
from aegisvault.runtime.goal_vault import GoalVault, RedisGoalVaultBackend

goal_vault = GoalVault(backend=RedisGoalVaultBackend.from_env())
goal_vault.commit_goal(
    session_id="session-123",
    application_name="email-assistant",
    goal="Summarize unread emails",
)

action_gate = ActionGate(goal_vault=goal_vault)

metadata = ToolMetadata(
    risk_level="low",
    allowed_domains=("email_assistant",),
    required_permissions=("gmail.read",),
    side_effect_level=SideEffectLevel.READ,
)

protected_tool = action_gate.protect_tool(
    read_unread_email,
    tool_metadata=metadata,
    policy=policy,
    tool_name="gmail.read",
    tool_description="Read unread email messages",
)

result = protected_tool("UNREAD", session_id="session-123")

if result.executed:
    print(result.result)
else:
    print(result.decision.verdict, result.decision.reason)
```

## Audit Events

Each decision emits an `ACTION_GATE_DECISION` event:

```json
{
  "event_type": "ACTION_GATE_DECISION",
  "session_id": "session-123",
  "tool": "gmail.read",
  "arguments": {"label": "UNREAD"},
  "similarity": 0.91,
  "ollama_called": false,
  "decision_source": "COSINE",
  "verdict": "EXECUTE",
  "confidence": 0.91,
  "reason": "Goal/action cosine similarity met the high execute threshold.",
  "latency_ms": 3.4
}
```

## Manual Checks

Verify Redis:

```bash
redis-cli ping
```

Verify Ollama:

```bash
curl http://localhost:11434/api/tags
ollama pull llama3.2
```

Run tests:

```bash
pytest tests/runtime/test_action_gate.py
```

Run the example:

```bash
python examples/action_gate_basic.py
```

Inspect audit logs from the example:

```bash
cat logs/action_gate_example.jsonl | python -m json.tool
```

## Limitations

Stage 3.2 does not include Sentinel Monitor, EMA, continuous reasoning monitoring, goal rewriting, multi-agent orchestration, LangChain, AutoGen, or mutable Goal Vault behavior. It protects proposed tool calls at the execution boundary only.
