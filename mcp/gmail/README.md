# Gmail MCP Phase 2

This integration uses the official Python MCP SDK v2 and Gmail API client. It
is separate from AgentDojo and does not alter benchmark code or snapshots.

## Phase 2 verification

On 11 November 2026, the Gmail MCP client retrieved a real test message from
the authenticated demo inbox. The message said team lunch starts at noon.
Qwen (`qwen3:4b-instruct`) read the message through MCP and reported the
correct time in both baseline and protected modes. The protected run used
Redis Goal Vault, `all-MiniLM-L6-v2` embeddings, and `llama3.2` for scope
evaluation. Each run completed one read and proposed zero sends.

Separate stdio checks validated an allowlisted recipient in dry-run mode and
rejected a recipient outside the allowlist. The live Gmail check covered
reading only. These checks establish Phase 2 functionality with a normal
message. See the [manual setup](../../MANUAL_SETUP.md) to repeat them.

## Safety invariants

- `AEGISVAULT_GMAIL_DRY_RUN` defaults to `true`.
- `send_email` accepts one plain-text message, no attachment or forwarding data.
- The MCP server enforces `AEGISVAULT_GMAIL_RECIPIENT_ALLOWLIST` in baseline
  and protected modes.
- Live sending also requires the exact local acknowledgement documented in the
  source configuration. Do not enable it during Phase 2.
- Email bodies are returned with `content_trust=untrusted_external_data` and
  remain in `tool` messages, separate from system and user instructions.
- Protected mode invokes Layer 0, Sentinel, and Action Gate before the MCP
  client can call `send_email`.
- Audit records redact message bodies and hash recipient addresses. The local
  `logs/` directory is ignored by Git.

## Install and configure

```bash
python -m pip install -e ".[runtime,dev,gmail]"
cp mcp/gmail/.env.example .env.gmail
set -a
source .env.gmail
set +a
```

For a send dry run, edit `.env.gmail` and replace the test-recipient
placeholder with an address you control. Reading works without an approved
recipient. The repository's root `credentials.json`, root `token.json`, and
`.env.gmail` are ignored.

## Commands

```bash
# Interactive browser authorization (manual consent required)
python -m aegisvault.integrations.gmail_mcp.auth

# Start the stdio MCP server (normally spawned by the client)
python -m aegisvault.integrations.gmail_mcp.server

# Discover tools
python -m aegisvault.integrations.gmail_mcp.client list-tools

# Read latest inbox email (requires OAuth)
python -m aegisvault.integrations.gmail_mcp.client read-latest

# Validate a send without Gmail API transmission (does not require OAuth)
python -m aegisvault.integrations.gmail_mcp.client send-dry-run \
  --to "${AEGISVAULT_GMAIL_RECIPIENT_ALLOWLIST%%,*}" \
  --subject "AegisVault dry run" \
  --body "This message must not be sent."

# Run Qwen with dynamically discovered tools
python -m aegisvault.integrations.gmail_mcp.agent \
  --mode baseline "Read my latest email and summarize it."
python -m aegisvault.integrations.gmail_mcp.agent \
  --mode protected "Read my latest email and summarize it."

# Tests
python -m pytest -q tests/integrations/gmail_mcp
python -m pytest -q
```

Protected mode defaults to Redis Goal Vault. Ensure Redis, Ollama,
`qwen3:4b-instruct`, `llama3.2`, and `all-MiniLM-L6-v2` are available before a
manual protected run. For an isolated non-production smoke test only, set
`AEGISVAULT_GMAIL_GOAL_BACKEND=memory`.
