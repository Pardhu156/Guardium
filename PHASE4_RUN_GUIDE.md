# Guardium Phase 4 defensive validation

Run from the repository root. Use the existing Python 3.11+ virtual
environment with the development and Gmail extras. If dependencies are
missing, install them with `.venv/bin/python -m pip install -e '.[dev,gmail]'`.

## Check the installed MCP tools

```bash
AEGISVAULT_GMAIL_DRY_RUN=true AEGISVAULT_GMAIL_RECIPIENT_ALLOWLIST= \
  .venv/bin/python -m aegisvault.integrations.gmail_mcp.client list-tools
```

Expect `read_latest_email` and `send_email`. Discovery starts the stdio
server and requires no Google consent.

## Verify the protected tool boundary

```bash
GUARDIUM_TEST_LOG_DIR="$(mktemp -d)"
.venv/bin/python -m pytest -q --basetemp "$GUARDIUM_TEST_LOG_DIR" \
  tests/integrations/gmail_mcp/test_guardrails.py \
  -k test_guardium_denies_tool_before_mcp_and_preserves_meeting_answer
find "$GUARDIUM_TEST_LOG_DIR" -name guardium-gmail.jsonl -print
```

Expect two passing cases. The tests script separate out-of-policy read and
send proposals and check that the associated MCP handlers were never
invoked. They verify the original Goal Vault request, observed gate decisions,
and acceptance of the scripted correct meeting-time answer. Each case records
sanitized JSON Lines events in its temporary test directory.

The **proposed** count comes from the agent's tool records. The **blocked**
count requires a real guard decision of `BLOCK`. Mock handler invocation
counts verify whether either operation crossed the MCP boundary. The
scripted answer checks whether processing can continue after a block.

See [observed defensive results](reports/guardium_gmail_demo/phase4_defensive_validation.md).
These tests use an in-memory MCP transport, a mock Gmail service, and
deterministic model inputs. They require neither OAuth nor Ollama.

## Troubleshooting

- If discovery fails to import the SDK, install the extras shown above.
- If an authorization assertion fails, inspect the case's JSON Lines log for
  the Layer 0 verdict, Sentinel event, and Action Gate decision.
- For a separate live check of the benign Gmail fixture, follow the
  [Phase 3 read-only instructions](PHASE3_MANUAL_STEPS.md#3-verify-the-exact-message).
