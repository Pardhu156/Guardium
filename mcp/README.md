# AegisVault MCP Demo

This folder contains a small, optional MCP-style stdio integration. It is isolated
from the production AegisVault package and from AgentDojo.

## Flow

```text
User request
  -> MCP client
  -> MCP tool request: get_policy_summary
  -> AegisVault Action Gate
  -> ALLOW or BLOCK
  -> MCP tool execution only when allowed
```

## Tool

`get_policy_summary` returns a sanitized, read-only summary of the demo policy:

- application name and description
- purpose
- allowed topics
- blocked topics
- request/response gate enabled flags

Unknown tools are blocked and never executed.

## Run

From the repository root:

```bash
pip install -e .
python mcp/client.py get_policy_summary
```

Blocked-tool example:

```bash
python mcp/client.py unknown_tool
```

The demo uses an in-memory Goal Vault and a deterministic tiny embedder to keep
the MCP example fast and dependency-light. It still calls AegisVault's
`ActionGate.evaluate_action()` before the tool executes.

## Isolation

- The main `aegisvault` package does not import `mcp`.
- AgentDojo integration does not import or execute MCP code.
- No benchmark metrics depend on this folder.
- No production thresholds, policies, Qwen/Ollama settings, or embedding model
  configuration are changed by this demo.

