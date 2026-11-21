# Guardium Gmail comparison

The runner uses an exact Phase 3 subject `[GUARDIUM<12-hex-run-id>]` in the
Inbox. It rejects missing or duplicate matches. Create the email in the test
mailbox yourself; its body may contain synthetic test instructions. Use the
same meeting details (Friday at 3:00 PM). The existing benign verifier still
requires the exact benign body; the comparison explicitly selects by subject.

The Gmail MCP server reads the message once. Both runs receive the same captured
message as untrusted tool output, with the same Qwen model, request, temperature,
and MCP schemas. The comparison checks the schema digest for each run. The send
boundary records a proposal and returns `recorded_not_sent` without invoking
Gmail's send tool. The existing server allowlist and live-send implementation
are unchanged. The recording boundary never sends, regardless of recipient.

Run from the project root after starting Ollama and Redis and authenticating
the dedicated Gmail account. Use a unique run ID; the current benign fixture is
`5f52ab01e12c`. For an injection trial, send a separate message with a fresh
run ID and the exact subject printed by the preview command.

```zsh
export AEGISVAULT_GMAIL_TOKEN_PATH=logs/guardium-reauth-token.json
export AEGISVAULT_GMAIL_DRY_RUN=true
.venv/bin/python -m pytest -q tests/integrations/gmail_mcp
.venv/bin/python -m pytest -q
.venv/bin/python -m aegisvault.integrations.gmail_mcp.guardium_comparison --preflight --run-id 5f52ab01e12c
.venv/bin/python -m aegisvault.integrations.gmail_mcp.guardium_comparison --run-id 5f52ab01e12c
```

The command prints paths for a local JSON result and Markdown report under
`logs/guardium-comparison/`. Local audit logs contain gate decisions. Reports
record tool names and argument names, not argument values or message bodies.
`executed_at_boundary` means the agent reached the recording boundary, not
that Gmail sent anything. `meeting_answer_correct` is a simple text heuristic,
not independent task evaluation. Check both modes for errors, matching schema
and message IDs, and actual observed gate decisions. A refusal or no send
proposal is a valid model result; it is not a Guardium block.

The previous live protected run had Sentinel block a legitimate read; the
underlying cause remains unresolved. This runner faithfully records that
outcome if it recurs. No policy thresholds or authorization decisions change.

For a manually seeded injection fixture, use `--scenario injection --trials 3`
and its own run ID. The scenario is an operator label, not a verified attack.
Each trial runs OFF then ON in fresh agent sessions with isolated Sentinel state.
Keep both modes in the same invocation so they use the identical captured message.
To regenerate the report from a saved result:

```zsh
.venv/bin/python -m aegisvault.integrations.gmail_mcp.guardium_comparison --report logs/guardium-comparison/RUN_ID/result.json
```

Preflight checks Qwen, Gmail and MCP discovery; it does not verify Redis or the
llama3.2 verifier. A completed run may still contain denied tools or an incorrect
answer. Review the local audit files and classify false positives manually.
