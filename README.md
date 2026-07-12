# AegisVault

AegisVault is an open-source, domain-specific guardrail middleware for AI applications. It lets developers describe an application's intended domain and purpose in YAML, then uses that policy to stop out-of-domain user requests before they reach the application and stop out-of-domain generated responses before they reach the user.

Stage 1 protects synchronous Python callables that accept a `str` and return a `str`. The first evaluator implementation uses a local Ollama model through Ollama's HTTP API.

Stage 3.1 adds an immutable Goal Vault runtime component for committing the user's original session goal as a write-once, integrity-checked anchor. See [docs/goal_vault.md](docs/goal_vault.md).

Stage 3.2 adds an Action Gate for protecting tool execution against the immutable goal anchor. See [docs/action_gate.md](docs/action_gate.md).

## Stage 1 Scope

Included:

- YAML policy loading and validation
- Request Gate and Response Gate
- Ollama-based scope evaluator
- Confidence threshold handling
- Lightweight deterministic checks
- Structured JSON Lines audit logging
- Synchronous callable wrapper
- Unit tests with fake evaluators
- Minimal example application

Stage 3.1 runtime addition:

- Immutable Goal Vault with Redis and in-memory backends
- L2-normalized goal embeddings
- SHA-256 integrity commitments
- TTL-based write-once storage

Stage 3.2 runtime addition:

- Action Gate for proposed tool calls
- Cosine similarity shortcut for clear execute/block decisions
- Ollama verification only for uncertain actions
- `EXECUTE`, `JUSTIFY`, and `BLOCK` tool decisions

Deferred to later stages:

- EMA or adaptive scoring
- Sentinel Monitor
- Continuous semantic drift scoring
- LangChain integrations
- Web APIs, UI, Docker, and deployment code

## Architecture Flow

```text
User request
    ↓
Deterministic request checks
    ↓
Request Gate
    ↓
Ollama scope evaluator
    ↓
ALLOW / BLOCK / CLARIFY
    ↓
Protected callable is invoked only when allowed
    ↓
Generated response
    ↓
Deterministic response checks
    ↓
Response Gate
    ↓
Ollama scope evaluator
    ↓
ALLOW / BLOCK / REPLACE
    ↓
Final result
    ↓
Audit log
```

The gates depend on the abstract `ScopeEvaluator` interface, not directly on Ollama, so future evaluators can be added without changing the public `AegisVault` API.

## Installation For Local Development

AegisVault requires Python 3.11+.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For Goal Vault runtime dependencies:

```bash
pip install -e ".[runtime]"
```

If your `python3` already points to Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Ollama Prerequisites

Install and start Ollama locally, then pull the model configured in your policy:

```bash
ollama --version
ollama serve
ollama pull llama3.2
```

In another terminal, verify the local API:

```bash
curl http://localhost:11434/api/tags
```

## Example Policy

```yaml
version: "1.0"

application:
  name: ecommerce-support
  description: Customer support assistant for an ecommerce platform

purpose: >
  Help users with products, orders, shipping, returns and refunds.

allowed_topics:
  - products
  - orders
  - shipping
  - returns
  - refunds
  - payment status

blocked_topics:
  - programming
  - medical advice
  - legal advice
  - investment advice

gates:
  request:
    enabled: true
    allow_threshold: 0.80
    block_threshold: 0.80
    low_confidence_action: clarify
  response:
    enabled: true
    allow_threshold: 0.80
    block_threshold: 0.80
    low_confidence_action: block

evaluator:
  provider: ollama
  model: llama3.2
  base_url: http://localhost:11434
  timeout_seconds: 30
  temperature: 0
```

Domain behavior comes from the policy. The framework source does not hardcode ecommerce, medical, HR, or other domain rules.

## Basic Usage

```python
from aegisvault import AegisVault


def ecommerce_assistant(prompt: str) -> str:
    return f"Demo application received: {prompt}"


guard = AegisVault.from_policy("policies/ecommerce.yaml")
protected_app = guard.wrap(ecommerce_assistant)

result = protected_app("Where is my order?")

print(result.final_response)
print(result.application_called)
print(result.terminated_by)
```

Run the included example:

```bash
python examples/basic_usage.py
```

## Decisions

Request Gate:

- `ALLOW`: the request may reach the protected callable.
- `BLOCK`: the request is rejected before the protected callable is called.
- `CLARIFY`: the protected callable is not called; the user receives a safe clarification message.

Response Gate:

- `ALLOW`: the generated response is returned.
- `BLOCK`: the generated response is blocked and replaced with safe fallback text.
- `REPLACE`: the generated response is replaced with safe fallback text.

A wrapped callable returns a structured `GuardResult` with:

- `final_response`
- `request_decision`
- `response_decision`
- `application_called`
- `request_accepted`
- `response_accepted`
- `was_modified`
- `terminated_by`
- `original_response`

`terminated_by` is one of `REQUEST_GATE`, `RESPONSE_GATE`, or `APPLICATION`.

## Confidence Handling

AegisVault does not blindly accept evaluator verdicts.

Request logic:

```python
if verdict == ALLOW and confidence >= allow_threshold:
    final_verdict = ALLOW
elif verdict == BLOCK and confidence >= block_threshold:
    final_verdict = BLOCK
else:
    final_verdict = configured_low_confidence_action
```

Response logic:

```python
if verdict == ALLOW and confidence >= allow_threshold:
    final_verdict = ALLOW
elif verdict == BLOCK and confidence >= block_threshold:
    final_verdict = BLOCK
else:
    final_verdict = configured_low_confidence_action
```

LLM confidence is model-reported and is not statistically calibrated. Deterministic checks and runtime fallbacks record `null` confidence rather than inventing a value.

## Audit Logs

When audit logging is enabled, AegisVault writes one JSON object per line:

```json
{
  "event_id": "uuid",
  "timestamp": "ISO-8601 UTC",
  "application": "ecommerce-support",
  "gate": "request",
  "input_text": "Where is my order?",
  "verdict": "ALLOW",
  "confidence": 0.94,
  "reason": "The request concerns order tracking.",
  "latency_ms": 218.5,
  "evaluator": "ollama:llama3.2",
  "application_called": true,
  "session_id": null,
  "terminated_by": "APPLICATION",
  "metadata": {}
}
```

The log directory is created automatically. Full request and response text can be disabled in policy with `audit.include_request_text` and `audit.include_response_text`.

## Running Tests

Unit tests do not require Ollama:

```bash
pytest
```

Run Goal Vault tests only:

```bash
pytest tests/runtime
```

Run the optional Ollama integration test only when Ollama is running and the configured model is available:

```bash
AEGISVAULT_RUN_OLLAMA_TESTS=1 pytest -m ollama
```

## Current Limitations

- Stage 1 supports synchronous Python callables only.
- Ollama is the only built-in evaluator provider.
- Deterministic checks are intentionally lightweight.
- The middleware does not perform embeddings, semantic similarity, or complex security scanning.
- Runtime evaluator failures follow policy-configured fallback behavior.
- The package is not published to PyPI; use editable local installation for development.
