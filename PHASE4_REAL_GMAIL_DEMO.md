# Guardium Phase 4 real Gmail delivery check

This guide runs the existing protected delivery check for a message you
explicitly authorize. Use two test Gmail accounts you control. The separate
[defensive checks](PHASE4_RUN_GUIDE.md) exercise denied proposals with mocked
handlers.

## 1. Prepare the local environment

Run from the repository root. Use the Phase 2 Python environment, Gmail
OAuth token, and local Redis. If Ollama is stopped, start it in a separate
terminal with `OLLAMA_HOST=127.0.0.1:11434 OLLAMA_CONTEXT_LENGTH=16384 ollama serve`.
Preflight expects `qwen3:4b-instruct` and `llama3.2`. If Google consent is
required, follow the [Phase 2 authorization steps](MANUAL_SETUP.md#1-authorize-the-test-account).

```bash
set -a
if [ -f .env.gmail ]; then source .env.gmail; fi
set +a
export AEGISVAULT_GMAIL_DRY_RUN=true
GUARDIUM_MAIL_MODULE=aegisvault.integrations.gmail_mcp.guardium_live_mail
printf 'Guardium test sender: '
read -r GUARDIUM_SENDER
printf 'Approved second test mailbox: '
read -r GUARDIUM_RECIPIENT
```

The recipient must be in the existing Gmail MCP allowlist. If it is missing,
explicitly approve that one test address for this terminal session:

```bash
export AEGISVAULT_GMAIL_RECIPIENT_ALLOWLIST="$GUARDIUM_RECIPIENT"
```

## 2. Preflight and dry run

```bash
.venv/bin/python -m "$GUARDIUM_MAIL_MODULE" preflight \
  --sender "$GUARDIUM_SENDER" --recipient "$GUARDIUM_RECIPIENT"
.venv/bin/python -m "$GUARDIUM_MAIL_MODULE" send \
  --sender "$GUARDIUM_SENDER" --recipient "$GUARDIUM_RECIPIENT"
```

`validated_not_sent` means the exact authorized message passed validation
and the tool returned before calling the Gmail send API. The report records
the Action Gate decision. If Guardium blocks the proposal, inspect the
decision and policy metadata in the report before proceeding.

## 3. Authorize one live message

```bash
.venv/bin/python -m "$GUARDIUM_MAIL_MODULE" send --live-send \
  --sender "$GUARDIUM_SENDER" --recipient "$GUARDIUM_RECIPIENT"
```

Review the displayed addresses, subject `Guardium Authorized Test`, and
body `GUARDIUM_LIVE_TEST_123`. Type `SEND` to confirm the exact message. The
existing Action Gate must authorize Qwen's matching proposal before it
reaches Gmail MCP. The CLI permits one dispatch per run.

## 4. Check the inbox and record the outcome

The CLI prints the path to `logs/guardium-live-mail/<run-id>/result.json`.
The same directory contains `audit.jsonl`. A `gmail_api_accepted` result
contains the Gmail API message ID. Check the receiving Inbox to verify
delivery. If you see the message from this run, record your confirmation:

```bash
printf 'Live run ID: '
read -r GUARDIUM_RUN_ID
.venv/bin/python -m "$GUARDIUM_MAIL_MODULE" report \
  "$GUARDIUM_RUN_ID" --confirm-inbox-delivery
```

Omit `--confirm-inbox-delivery` to view the report without marking receipt.
For `outcome_unknown`, inspect Gmail's Sent folder and the receiving Inbox
before retrying. The dispatch receipt and Response Gate verdict are separate
fields in the report.
