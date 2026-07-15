# AgentDojo Integration

This folder contains the Stage 6.1 compatibility layer for routing AgentDojo-like
benchmark tasks through AegisVault runtime security.

The adapter intentionally bypasses the semantic Request Gate and Response Gate.
It uses:

1. Layer 0 request sanity validation
2. Goal Vault initialization
3. Qwen or AgentDojo agent step
4. Layer 0 tool-call validation
5. Sentinel evaluation
6. Action Gate authorization
7. Tool execution
8. AgentDojo evaluator

The included smoke runner uses mock AgentDojo-style suites because the real
AgentDojo package is not vendored in this repository.

Run the mock compatibility smoke test:

```bash
.venv/bin/python evaluation/agentdojo/smoke_agentdojo_adapter.py
```

Manual AgentDojo setup is required before running real Workspace, Slack,
Banking, and Travel suites.

