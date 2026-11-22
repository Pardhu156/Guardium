# Guardium Stage 9

## 9.0 Authorized delivery

Use the protected Gmail MCP CLI from the repository root in zsh, keeping the
same terminal open. The subject is **Guardium Stage 9.0 — Real Gmail Send Test**.
The complete body is **This is a real Gmail integration test sent through Guardium's Gmail MCP server. Gmail sending is working successfully.**
The operator selects one personal recipient for this terminal session.

```zsh
cd /Users/pardhu/Developer/AegisVault
export AEGISVAULT_GMAIL_TOKEN_PATH=logs/guardium-reauth-token.json
export AEGISVAULT_GMAIL_DRY_RUN=true
unset AEGISVAULT_GMAIL_FIXTURE_RUN_ID
MAIL_MODULE=aegisvault.integrations.gmail_mcp.guardium_live_mail
SENDER=guardiummcpdemo@gmail.com
read -r "RECIPIENT?Type your personal Gmail address and press Return: "
export AEGISVAULT_GMAIL_RECIPIENT_ALLOWLIST="$RECIPIENT"
.venv/bin/python -m pytest -q tests/integrations/gmail_mcp
ollama list
redis-cli ping
```

Expect passing tests, `qwen3:4b-instruct` and `llama3.2` (possibly tagged
`:latest`), and `PONG`. If a service is stopped, start it in its own terminal
with `ollama serve` or `redis-server` as appropriate. The protected path also
uses the existing all-MiniLM-L6-v2 embedding installation.

```zsh
.venv/bin/python -m "$MAIL_MODULE" preflight --sender "$SENDER" --recipient "$RECIPIENT"
.venv/bin/python -m "$MAIL_MODULE" send --stage9 --sender "$SENDER" --recipient "$RECIPIENT"
```

Preflight queries Google's account profile and compares the returned address
to the sender. It also checks the allowlist, actor, verifier and tool discovery.
A successful preview reports `validated_not_sent` and a null Gmail message ID.
Review a block or mismatch in the printed result before proceeding.

If Google requires fresh consent or the Gmail API reports insufficient
permission, select a fresh token location within the ignored logs directory
and authorize the desktop client in your browser:

```zsh
export AEGISVAULT_GMAIL_TOKEN_PATH="logs/stage9-oauth-$(uuidgen).json"
.venv/bin/python -m aegisvault.integrations.gmail_mcp.auth
```

Select the demo sender, grant Gmail read and send access and check the
authenticated address printed in the terminal. Repeat preflight and the dry
run. If an earlier live attempt reserved dispatch, inspect that run and both
mailboxes before deciding how to handle the uncertain attempt.

```zsh
.venv/bin/python -m "$MAIL_MODULE" send --stage9 --live-send --sender "$SENDER" --recipient "$RECIPIENT"
```

Review From, To, subject and body. Type `SEND` only for the exact authorized
message. After Qwen proposes matching arguments and Guardium approves, Gmail
acceptance produces `dispatch_status=gmail_api_accepted` and a Gmail message
ID in the local result. The Response Gate verdict concerns the final
user-facing answer. `outcome_unknown` means the outcome needs investigation.
Keep `logs/guardium-live-mail/stage9-dispatch.run` in place. This exclusive
reservation protects repeated and concurrent Stage 9 commands in this checkout
even when a timeout or interruption leaves acceptance uncertain.

In your personal Gmail inbox, check the sender, exact subject, body and arrival
time. If missing, check Spam and All Mail. Only after observing the matching
message, record receipt:

```zsh
.venv/bin/python -m "$MAIL_MODULE" report "$(cat logs/guardium-live-mail/stage9-dispatch.run)" --confirm-inbox-delivery
```

Expect `operator_confirmed_inbox_delivery=true`. Omit the confirmation flag to
inspect the result. Each attempt has an ignored local result.json and audit
log under logs/guardium-live-mail/<run-id>/.

## Implementation roadmap

| Stage | Build on existing work | Completion evidence |
|---|---|---|
| 9.1 Synthetic profile | Promote the test-only profile probe into a fixture-backed, simulated tool for an isolated test environment. Use invented records, a fixed fixture and explicit read authorization. | Tests cover a permitted fixture read, an out-of-scope read blocked before access, and the absence of a path parameter. |
| 9.2 Gmail fixtures | Reuse the exact-subject selector and Phase 3 benign body. Maintain a benign control and a separately identified synthetic security fixture, with reviewed expected meeting details and a manifest of message IDs and hashes. | Repeated selections yield the same IDs and hashes. Missing or ambiguous matches fail. First reproduce the protected benign read and verify that its answer reaches the user. |
| 9.3 Local chat API | Wrap the bounded Qwen agent in a loopback service. Allocate session IDs and Goal Vault / Sentinel state on the server. Return the final answer and a filtered stream of gate and tool events. | An integration test follows a request from the API through the gates, verifies session isolation, cancellation and sanitized errors. |
| 9.4 VS Code sidebar | Build a webview chat panel backed by the local API, with request entry, response display and progress. Render email/model output as untrusted text. Keep Gmail credentials on the backend. | Verify a benign read from an Extension Development Host and an understandable error when the backend is unavailable. |
| 9.5 Controls and events | Expose the existing baseline/protected comparison modes for recording-only sessions. Display proposed, denied, recorded, API-accepted and operator-confirmed events. Bind the selected mode to the session. | UI events match the backend trace and toggling modes starts a fresh session. |
| 9.6 Controlled demonstration | Show the 9.0 authorized receipt, then independently evaluate matched recording-only OFF/ON sessions. Use identical model settings, schemas and captured message. | Report observed proposals, decisions, correct meeting answers, incomplete runs and latency. Include the benign false block if it persists. |
| 9.7 Presentation and cleanup | Prepare a reproducible walkthrough, setup checklist and sanitized comparison using the actual results. Keep raw traces, personal addresses and OAuth files local. | Repeat the walkthrough from the documented configuration and check every published claim against a recorded outcome. |
