# Guardium authorized Gmail delivery check

Use the existing virtual environment and Gmail OAuth setup. Start local
Ollama with `qwen3:4b-instruct` and `llama3.2` installed. The protected
path uses the existing Guardium policy and defaults to Redis Goal Vault.

## 1. Choose your two test mailboxes

From the repository root, load the existing configuration and enter the
address of your second test mailbox. It must be one you control.

```bash
set -a
if [ -f .env.gmail ]; then source .env.gmail; fi
set +a
export AEGISVAULT_GMAIL_DRY_RUN=true
GUARDIUM_SENDER='guardiummcpdemo@gmail.com'
printf 'Second test mailbox: '
read -r GUARDIUM_RECIPIENT
```

If the recipient is missing from the existing allowlist, explicitly approve
it for this terminal session before running preflight:

```bash
export AEGISVAULT_GMAIL_RECIPIENT_ALLOWLIST="$GUARDIUM_RECIPIENT"
```

## 2. Verify the environment and dry-run the authorized request

```bash
.venv/bin/python -m aegisvault.integrations.gmail_mcp.guardium_live_mail \
  preflight --sender "$GUARDIUM_SENDER" --recipient "$GUARDIUM_RECIPIENT"
.venv/bin/python -m aegisvault.integrations.gmail_mcp.guardium_live_mail \
  send --sender "$GUARDIUM_SENDER" --recipient "$GUARDIUM_RECIPIENT"
```

Preflight checks the two Ollama models, the authenticated sender, the
existing recipient allowlist, and MCP tool discovery. The dry run runs Qwen
through the protected Guardium path. If its exact proposal is allowed and
validated, the report says `validated_not_sent`. A Guardium block is saved
with its actual decision and available policy information. Review the
recorded outcome before continuing.

## 3. Confirm one live send

```bash
.venv/bin/python -m aegisvault.integrations.gmail_mcp.guardium_live_mail \
  send --live-send --sender "$GUARDIUM_SENDER" --recipient "$GUARDIUM_RECIPIENT"
```

Review the displayed addresses, subject `Guardium Authorized Test`, and
body `GUARDIUM_LIVE_TEST_123`. Type `SEND` if you approve that exact
message. Qwen must propose matching arguments, and Guardium must authorize
the proposal before it reaches the existing Gmail MCP send tool. The
delivery check exposes only the send tool and can dispatch at most one
message per invocation.

The report is under `logs/guardium-live-mail/<run-id>/result.json`, alongside
the local audit log. `gmail_api_accepted` records an API message ID.
`outcome_unknown` means a call crossed the dispatch boundary without a
confirmed receipt. Check Gmail's Sent folder and the receiving Inbox before
retrying an uncertain result.

## 4. Record observed delivery

Open the second Gmail Inbox and look for the newly sent message. If it is
there, copy the run ID from the report path and run:

```bash
printf 'Live run ID: '
read -r GUARDIUM_RUN_ID
.venv/bin/python -m aegisvault.integrations.gmail_mcp.guardium_live_mail \
  report "$GUARDIUM_RUN_ID" --confirm-inbox-delivery
```

This records your inbox confirmation separately from Gmail API acceptance.
Omit `--confirm-inbox-delivery` to view the report without changing it.

For the separate scripted denial checks, see the
[Phase 4 defensive run guide](PHASE4_RUN_GUIDE.md).
