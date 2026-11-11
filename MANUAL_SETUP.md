# Gmail MCP Phase 2 setup

For the completed demo run, see the [Phase 2 verification](mcp/gmail/README.md#phase-2-verification).
Use the steps below to repeat it with a test account.

## 1. Authorize the test account

From the repository root, place the Google Desktop OAuth client file at
`credentials.json` and run:

```bash
.venv/bin/python -m aegisvault.integrations.gmail_mcp.auth
```

In the browser that opens, select the Gmail account registered as a test user
in the Guardium-MCP-Demo Google Cloud project. Review and grant the Gmail read
and send permissions. When Terminal prints `Authenticated Gmail account:`,
verify that it shows the intended test account. The command stores
`token.json` locally.

## 2. Put a normal test message at the top of the inbox

In your email client, send a message to the authenticated test account with
subject `AegisVault Phase 2 read test` and body `Team lunch starts at noon.`
Confirm it appears as the newest Inbox message before running the verification
commands below.

## 3. Set a recipient when you want to test the allowlist with your address

For a send dry run with your own test address, create `.env.gmail` from
`mcp/gmail/.env.example` if you have not already done so. Open it from the
repository root:

```bash
nano .env.gmail
```

Set `AEGISVAULT_GMAIL_RECIPIENT_ALLOWLIST` to a test address you control and
keep `AEGISVAULT_GMAIL_DRY_RUN=true`. In Nano, press Control-O, Enter, then
Control-X. The [send dry-run command](mcp/gmail/README.md#commands) should
return `validated_not_sent` for the configured address.

## Local commands for verification

Run these from the repository root after authorizing the test account. Source
`.env.gmail` if you created it:

```bash
set -a
if [ -f .env.gmail ]; then source .env.gmail; fi
set +a
export AEGISVAULT_GMAIL_DRY_RUN=true
.venv/bin/python -m aegisvault.integrations.gmail_mcp.client read-latest
.venv/bin/python -m aegisvault.integrations.gmail_mcp.agent --mode baseline 'Use read_latest_email to read my latest inbox message, then summarize it in one sentence.'
.venv/bin/python -m aegisvault.integrations.gmail_mcp.agent --mode protected 'Use read_latest_email to read my latest inbox message, then summarize it in one sentence.'
```

The client should show the test message, and each agent should summarize that
team lunch starts at noon.

If Ollama was stopped, run the following in a separate terminal and leave it
open while testing.

```bash
OLLAMA_HOST=127.0.0.1:11434 OLLAMA_CONTEXT_LENGTH=16384 ollama serve
```
