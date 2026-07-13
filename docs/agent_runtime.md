# Stage 4.0 Agent Runtime Validation

Stage 4.0 validates the local agent runtime infrastructure required before wrapping real agents with AegisVault.

It is intentionally independent from Request Gate, Response Gate, Goal Vault, Action Gate, Redis, Gemini, and Stage 4.1 domain agents.

## Model

Default model:

```text
qwen3:4b-instruct
```

This is a practical local Qwen3 Instruct tag in Ollama. The model is configurable:

```bash
python run_agent.py --model qwen3:8b "What time is it?"
```

Ollama's Qwen3 page lists Qwen3 as the latest generation of Qwen models and includes tool-capable tags. Larger tags such as `qwen3:30b` and `qwen3:235b` require much more memory, so the default is the smaller instruct model for local validation.

## Architecture

```text
Ollama
  ↓
Qwen Instruct
  ↓
AgentRuntime
  ↓
ToolRegistry
  ↓
Mock local tools
```

## Tools

Local mock tools:

- `get_time`
- `calculator`
- `weather`
- `read_text`
- `echo`
- `save_note`
- `list_notes`

No external APIs are called.

## CLI

List tools:

```bash
python run_agent.py --list-tools
```

Run once:

```bash
python run_agent.py --model qwen3:4b-instruct "Calculate 42 * 18."
```

Interactive mode:

```bash
python run_agent.py --interactive --model qwen3:4b-instruct --verbose
```

Trace log:

```bash
logs/agent_runtime_traces.jsonl
```

## Validation

Mock validation, no Ollama required:

```bash
python evaluation/agent_runtime/scripts/validate_agent_runtime.py --mock
```

Real Ollama validation:

```bash
ollama serve
ollama pull qwen3:4b-instruct
python evaluation/agent_runtime/scripts/validate_agent_runtime.py --model qwen3:4b-instruct
```

Reports are stored under:

```text
evaluation/agent_runtime/reports/<run_id>/
```

## Limitations

This stage validates tool-calling infrastructure only. It does not evaluate AegisVault middleware, real agents, email, document QA, ecommerce workflows, Redis, or guardrail enforcement.
