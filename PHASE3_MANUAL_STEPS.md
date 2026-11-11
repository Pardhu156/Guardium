# Guardium Phase 3 manual steps

These steps verify a normal meeting email in your dedicated test mailbox.
Run the commands from the repository root using the Phase 2 virtual environment.

## 1. Preview the benign fixture

Generate a fresh ID so the verifier can locate one message by its subject.
Keep this terminal open for step 3.

```bash
RUN_ID=$(.venv/bin/python -c 'import secrets; print(secrets.token_hex(6))')
.venv/bin/python -m aegisvault.integrations.gmail_mcp.guardium_verify_fixture preview "$RUN_ID"
```

Success prints a subject starting with `Guardium Demo — Project Review Meeting`
and ending with `[GUARDIUM…]`. The body says the meeting is Friday at 3:00 PM.

## 2. Deliver the normal email

In Gmail, sign in to an account you control and click **Compose**. Set **To**
to your Guardium test mailbox. Copy the generated subject (without the
`Subject:` label) and the printed body into the corresponding fields. Click
**Send**, then open the Guardium test inbox and confirm the exact subject
appears once. This step places the known fixture in the Inbox for verification.

## 3. Verify the exact message

Run this in the same terminal. Replace the example account with your
authenticated Guardium test mailbox.

```bash
export AEGISVAULT_GMAIL_DRY_RUN=true
.venv/bin/python -m aegisvault.integrations.gmail_mcp.guardium_verify_fixture verify "$RUN_ID" \
  --account 'guardium-test@example.com'
```

Success prints `Guardium fixture verified. Gmail message ID:` followed by the
ID. If the command finds no match, it prints the expected subject. Search the
test Inbox for the `GUARDIUM` marker and its 12-character suffix. Use the
same run ID for preview and verification. If you reopened Terminal, set
`RUN_ID` to the suffix from the subject you already sent. If there are duplicates,
repeat from step 1 with a fresh ID. If Gmail requires new consent, repeat the
[Phase 2 authorization steps](MANUAL_SETUP.md#1-authorize-the-test-account).
No recipient allowlist change is needed for this read-only check.
