# AegisVault Stage 3.1: Immutable Goal Vault

Stage 3.1 adds an immutable runtime vault for anchoring the user's original goal at the start of a protected session. It is intentionally narrow: it stores a write-once, integrity-checked goal anchor that later stages can read, but it does not make semantic similarity decisions.

## What Is Included

- `GoalCommitRequest` input validation.
- Deterministic goal text normalization.
- SentenceTransformer embedding generation.
- L2 normalization before storage.
- SHA-256 integrity commitment over a canonical JSON payload.
- Redis-backed write-once storage with `SET NX EX`.
- In-memory backend for local tests and development.
- TTL-based expiration.
- Retrieval with integrity verification.
- Replaceable backend and audit sink interfaces.

## What Is Excluded

- No cosine similarity.
- No EMA or adaptive drift scoring.
- No Sentinel Monitor.
- No Action Gate.
- No Goal Vault updates or overwrites.
- No LangChain, agents, web server, or deployment layer.

## Public API

```python
from aegisvault.runtime.goal_vault import (
    GoalVault,
    RedisGoalVaultBackend,
    InMemoryGoalVaultBackend,
)

backend = RedisGoalVaultBackend.from_env()
vault = GoalVault(backend=backend)

anchor = vault.commit_goal(
    session_id="session-123",
    application_name="ecommerce-support",
    goal="I want to track my delayed order.",
    ttl_seconds=3600,
)

same_anchor = vault.get_anchor("session-123")
```

`commit_goal()` succeeds once per `session_id`. A second commit for the same live session raises `GoalAlreadyCommittedError`.

## Commitment Flow

```text
GoalCommitRequest
    ↓
Validate session id, application name, goal, TTL, metadata
    ↓
Normalize goal text
    ↓
Generate embedding with SentenceTransformer
    ↓
L2 normalize embedding
    ↓
Build GoalAnchor
    ↓
Compute SHA-256 over canonical JSON without integrity_hash
    ↓
Store anchor with backend write-once primitive
    ↓
Emit audit events
```

## Retrieval Flow

```text
session_id
    ↓
Backend get
    ↓
Deserialize stored JSON
    ↓
Recompute SHA-256 commitment
    ↓
Compare with stored integrity_hash
    ↓
Return GoalAnchor or raise GoalIntegrityError
```

## Redis Key Schema

Default key prefix:

```text
aegisvault:goal_anchor:
```

Full key:

```text
aegisvault:goal_anchor:<session_id>
```

The Redis backend stores each anchor with:

```text
SET <key> <json> NX EX <ttl_seconds>
```

`NX` gives atomic write-once behavior. `EX` ensures the anchor expires automatically.

## Stored JSON Schema

Each Redis value is one JSON object:

```json
{
  "schema_version": "1.0",
  "session_id": "session-123",
  "application_name": "ecommerce-support",
  "original_goal": "I want to track my delayed order.",
  "normalized_goal": "I want to track my delayed order.",
  "goal_embedding": [0.0123, -0.0456],
  "embedding_model": "all-MiniLM-L6-v2",
  "embedding_dimension": 384,
  "integrity_hash": "sha256-hex",
  "created_at": "2026-07-12T10:30:15.000000Z",
  "expires_at": "2026-07-12T11:30:15.000000Z",
  "metadata": {}
}
```

The default embedding model is `all-MiniLM-L6-v2` with dimension `384`. Tests use a deterministic fake embedder so CI does not download model weights.

## Integrity Commitment

The integrity hash is a SHA-256 digest over canonical JSON with:

- sorted keys,
- compact separators,
- UTF-8 text,
- deterministic float formatting for embedding values,
- all anchor fields except `integrity_hash`.

On retrieval, AegisVault recomputes the digest and compares it with `hmac.compare_digest`.

## Redis Configuration

`RedisGoalVaultBackend.from_env()` reads:

```bash
export AEGISVAULT_REDIS_HOST="localhost"
export AEGISVAULT_REDIS_PORT="6379"
export AEGISVAULT_REDIS_DB="0"
export AEGISVAULT_REDIS_USERNAME=""
export AEGISVAULT_REDIS_PASSWORD=""
export AEGISVAULT_REDIS_SSL="false"
export AEGISVAULT_REDIS_TTL_SECONDS="3600"
export AEGISVAULT_REDIS_KEY_PREFIX="aegisvault:goal_anchor:"
```

Windows PowerShell equivalent:

```powershell
$env:AEGISVAULT_REDIS_HOST="localhost"
$env:AEGISVAULT_REDIS_PORT="6379"
$env:AEGISVAULT_REDIS_DB="0"
$env:AEGISVAULT_REDIS_PASSWORD="your_password_here"
```

Do not commit Redis passwords or secrets to source control.

## Redis ACL Example

For a dedicated Redis user, allow only the key prefix and minimal commands:

```text
ACL SETUSER aegisvault on >replace_with_strong_password ~aegisvault:goal_anchor:* +set +get +del +ttl
```

If your Redis deployment requires authentication, pass the username and password through environment variables.

## Local Setup

Install runtime dependencies:

```bash
pip install -e ".[runtime]"
```

On Python 3.13, `sentence-transformers` may be unavailable because its `torch` dependency may not publish matching wheels for the environment yet. If you only need the Redis backend or Redis integration test, install the Redis extra:

```bash
pip install -e ".[redis]"
```

For production use with the default SentenceTransformer embedder, prefer Python 3.11 or 3.12:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[runtime]"
```

If using Redis locally:

```bash
brew install redis
brew services start redis
redis-cli ping
```

Run unit tests:

```bash
pytest tests/runtime
```

Run the optional live Redis integration test:

```bash
export AEGISVAULT_RUN_REDIS_TESTS=1
pytest tests/runtime/test_goal_vault_redis_integration.py
```

## Audit Events

The service emits structured events through the configured `AuditSink`:

- `GOAL_COMMIT_ATTEMPT`
- `GOAL_COMMITTED`
- `GOAL_DUPLICATE_REJECTED`
- `GOAL_RETRIEVED`
- `GOAL_INTEGRITY_VERIFIED`
- `GOAL_INTEGRITY_FAILED`
- `GOAL_DELETED`

Goal text is not logged by default. Set `include_goal_text_in_audit=True` only for trusted development or controlled evaluation runs.

## Limitations

The Goal Vault does not decide whether later text has drifted from the goal. It only stores and verifies an immutable goal anchor. Stage 3.2 can build Sentinel-style monitoring on top of this anchor without changing the storage contract.
