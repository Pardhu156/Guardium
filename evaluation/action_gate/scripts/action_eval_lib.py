"""Utilities for Stage 3.3 Goal Vault + Action Gate evaluation."""

from __future__ import annotations

import json
import math
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping
from uuid import uuid4

from aegisvault.policy import DomainPolicy, load_policy
from aegisvault.runtime.action_gate import (
    ActionDecisionSource,
    ActionEvaluator,
    ActionGate,
    ActionGateConfig,
    ActionGateDecision,
    ActionRuntimeContext,
    ActionVerdict,
    ProposedToolAction,
    SideEffectLevel,
    ToolExecutionResult,
    ToolMetadata,
)
from aegisvault.runtime.action_gate.exceptions import ActionEvaluatorError
from aegisvault.runtime.goal_vault import (
    GoalAlreadyCommittedError,
    GoalAnchor,
    GoalEmbedder,
    GoalVault,
    InMemoryGoalVaultBackend,
    RedisGoalVaultBackend,
)

DOMAIN_FILES = {
    "email_assistant": "email_assistant_actions.jsonl",
    "ecommerce": "ecommerce_actions.jsonl",
    "document_qa": "document_qa_actions.jsonl",
    "cross_domain": "cross_domain_actions.jsonl",
}
POLICY_FILES = {
    "email_assistant": "email_assistant.yaml",
    "ecommerce": "ecommerce.yaml",
    "document_qa": "document_qa.yaml",
    "cross_domain": "cross_domain.yaml",
}


@dataclass(frozen=True, slots=True)
class ActionEvalCase:
    id: str
    domain: str
    goal: str
    application_name: str
    action: ProposedToolAction
    tool_metadata: ToolMetadata
    runtime_context: ActionRuntimeContext
    expected_verdict: ActionVerdict
    category: str
    notes: str


class DeterministicActionEvalEmbedder(GoalEmbedder):
    """Deterministic 384-dimensional embedder for reproducible synthetic evaluation."""

    model_name = "deterministic-all-MiniLM-L6-v2-compatible"
    dimension = 384

    def embed(self, text: str) -> list[float]:
        lowered = text.lower()
        vector = [0.0] * self.dimension
        if any(signal in lowered for signal in ("rm -rf", "attacker", "delete_messages", "dump", "export all", "api_key")):
            vector[0] = -1.0
        elif any(signal in lowered for signal in ("send", "issue_refund", "modify", "create_coupon", "write_file", "save_summary", "annotate", "credit")):
            vector[0] = 0.6
            vector[1] = 0.8
        else:
            vector[0] = 1.0
        return vector


class RuleBasedActionVerifier(ActionEvaluator):
    """Fake Ollama verifier used by default so tests/evaluations are reproducible."""

    name = "fake-ollama-action-verifier"

    def __init__(self, *, fail: bool = False) -> None:
        self.calls = 0
        self.fail = fail

    def evaluate(
        self,
        *,
        goal_anchor: GoalAnchor,
        action: ProposedToolAction,
        tool_metadata: ToolMetadata,
        policy: DomainPolicy,
        runtime_context: ActionRuntimeContext | None = None,
        goal_similarity: float | None = None,
    ) -> ActionGateDecision:
        self.calls += 1
        if self.fail:
            raise ActionEvaluatorError("Ollama disabled for this evaluation run")
        args = json.dumps(dict(action.tool_arguments), sort_keys=True).lower()
        if any(signal in args for signal in ("attacker", "rm -rf", "secret", "api_key", "export all", "dump")):
            verdict = ActionVerdict.BLOCK
            reason = "Arguments contain generic destructive or exfiltration signals."
        elif tool_metadata.requires_approval or tool_metadata.side_effect_level in {SideEffectLevel.WRITE, SideEffectLevel.NETWORK}:
            verdict = ActionVerdict.JUSTIFY
            reason = "Related action has write/network side effects or requires approval."
        else:
            verdict = ActionVerdict.EXECUTE
            reason = "Action appears aligned with the stored goal and tool metadata."
        return ActionGateDecision(
            tool_name=action.tool_name,
            tool_arguments=action.tool_arguments,
            goal_similarity=goal_similarity,
            decision_source=ActionDecisionSource.OLLAMA,
            verdict=verdict,
            confidence=0.9,
            reason=reason,
            latency_ms=1.0,
            ollama_called=True,
            goal_session=goal_anchor.session_id,
            metadata={"evaluator": self.name},
        )


class FakeToolExecutor:
    """Generic fake executor that records whether execution happened."""

    def __init__(self) -> None:
        self.invocation_count = 0
        self.arguments_received: dict[str, Any] | None = None
        self.execution_timestamp: str | None = None

    def run(self, *, decision: ActionGateDecision, expected_to_execute: bool) -> ToolExecutionResult:
        if decision.verdict != ActionVerdict.EXECUTE:
            return ToolExecutionResult(decision=decision, executed=False)
        self.invocation_count += 1
        self.arguments_received = dict(decision.tool_arguments)
        self.execution_timestamp = utc_now()
        return ToolExecutionResult(
            decision=decision,
            executed=True,
            result={"ok": True, "tool": decision.tool_name, "expected_to_execute": expected_to_execute},
        )


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def generate_run_id() -> str:
    return f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:4]}"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} contains invalid JSON: {exc}") from exc
    return rows


def load_cases(dataset_dir: Path, domains: Iterable[str]) -> tuple[list[ActionEvalCase], dict[str, str]]:
    seen: set[str] = set()
    cases: list[ActionEvalCase] = []
    dataset_files: dict[str, str] = {}
    for domain in domains:
        filename = DOMAIN_FILES[domain]
        path = dataset_dir / filename
        dataset_files[domain] = str(path)
        for payload in read_jsonl(path):
            case = parse_case(payload)
            if case.id in seen:
                raise ValueError(f"duplicate action case id {case.id!r}")
            seen.add(case.id)
            cases.append(case)
    return cases, dataset_files


def parse_case(payload: Mapping[str, Any]) -> ActionEvalCase:
    try:
        expected = ActionVerdict(str(payload["expected_verdict"]))
    except Exception as exc:
        raise ValueError(f"invalid expected_verdict for case {payload.get('id')!r}") from exc
    metadata = payload["tool_metadata"]
    return ActionEvalCase(
        id=str(payload["id"]),
        domain=str(payload["domain"]),
        goal=str(payload["goal"]),
        application_name=str(payload["application_name"]),
        action=ProposedToolAction(
            tool_name=str(payload["tool_name"]),
            tool_description=str(payload["tool_description"]),
            tool_arguments=payload.get("tool_arguments", {}),
        ),
        tool_metadata=ToolMetadata(
            risk_level=str(metadata["risk_level"]),
            allowed_domains=tuple(metadata.get("allowed_domains", ())),
            required_permissions=tuple(metadata.get("required_permissions", ())),
            side_effect_level=SideEffectLevel(str(metadata.get("side_effect", metadata.get("side_effect_level", "read")))),
            requires_approval=bool(metadata.get("requires_approval", False)),
        ),
        runtime_context=ActionRuntimeContext(
            reasoning_summary=payload.get("runtime_context", {}).get("reasoning_summary"),
            session_metadata=payload.get("runtime_context", {}).get("session_metadata", {}),
        ),
        expected_verdict=expected,
        category=str(payload["category"]),
        notes=str(payload.get("notes", "")),
    )


def load_policies(policy_dir: Path, domains: Iterable[str]) -> tuple[dict[str, DomainPolicy], dict[str, str]]:
    policies = {}
    paths = {}
    for domain in domains:
        path = policy_dir / POLICY_FILES[domain]
        policies[domain] = load_policy(path)
        paths[domain] = str(path)
    return policies, paths


def build_goal_vault(backend: str, embedder: GoalEmbedder, ttl_seconds: int = 300) -> GoalVault:
    if backend == "redis":
        return GoalVault(backend=RedisGoalVaultBackend.from_env(), embedder=embedder, default_ttl_seconds=ttl_seconds)
    return GoalVault(backend=InMemoryGoalVaultBackend(), embedder=embedder, default_ttl_seconds=ttl_seconds)


def run_case(
    *,
    case: ActionEvalCase,
    policy: DomainPolicy,
    backend: str,
    config: ActionGateConfig,
    no_ollama: bool,
    run_index: int,
) -> dict[str, Any]:
    total_started = time.perf_counter()
    embedder = DeterministicActionEvalEmbedder()
    goal_vault = build_goal_vault(backend, embedder)
    verifier = RuleBasedActionVerifier(fail=no_ollama)
    gate = ActionGate(goal_vault=goal_vault, embedder=embedder, evaluator=verifier, config=config)
    session_id = f"{case.id}-run-{run_index}-{uuid4().hex[:8]}"

    commit_started = time.perf_counter()
    anchor = goal_vault.commit_goal(
        session_id=session_id,
        application_name=case.application_name,
        goal=case.goal,
        ttl_seconds=300,
        metadata={"case_id": case.id, "domain": case.domain},
    )
    commit_latency = elapsed_ms(commit_started)
    duplicate_rejected = False
    try:
        goal_vault.commit_goal(session_id=session_id, application_name=case.application_name, goal=case.goal, ttl_seconds=300)
    except GoalAlreadyCommittedError:
        duplicate_rejected = True

    retrieval_started = time.perf_counter()
    retrieved = goal_vault.get_anchor(session_id)
    retrieval_latency = elapsed_ms(retrieval_started)
    norm = math.sqrt(sum(value * value for value in retrieved.goal_embedding))
    ttl = goal_vault.get_ttl(session_id)

    action_started = time.perf_counter()
    decision = gate.evaluate_action(
        session_id=session_id,
        action=case.action,
        tool_metadata=case.tool_metadata,
        policy=policy,
        runtime_context=case.runtime_context,
    )
    action_latency = elapsed_ms(action_started)

    expected_to_execute = case.expected_verdict == ActionVerdict.EXECUTE
    executor = FakeToolExecutor()
    execution_result = executor.run(decision=decision, expected_to_execute=expected_to_execute)
    execution_match = (
        (case.expected_verdict == ActionVerdict.EXECUTE and executor.invocation_count == 1)
        or (case.expected_verdict != ActionVerdict.EXECUTE and executor.invocation_count == 0)
    )
    verdict_match = decision.verdict == case.expected_verdict
    total_latency = elapsed_ms(total_started)
    return {
        "case_id": case.id,
        "run_index": run_index,
        "domain": case.domain,
        "category": case.category,
        "goal": case.goal,
        "session_id": session_id,
        "expected_verdict": case.expected_verdict.value,
        "actual_verdict": decision.verdict.value,
        "matched": verdict_match and execution_match,
        "verdict_match": verdict_match,
        "execution_match": execution_match,
        "false_execute": case.expected_verdict != ActionVerdict.EXECUTE and decision.verdict == ActionVerdict.EXECUTE,
        "false_block": case.expected_verdict == ActionVerdict.EXECUTE and decision.verdict == ActionVerdict.BLOCK,
        "false_justify": case.expected_verdict in {ActionVerdict.EXECUTE, ActionVerdict.BLOCK} and decision.verdict == ActionVerdict.JUSTIFY,
        "tool_name": decision.tool_name,
        "tool_arguments": dict(decision.tool_arguments),
        "goal_similarity": decision.goal_similarity,
        "decision_source": decision.decision_source.value,
        "ollama_called": decision.ollama_called,
        "confidence": decision.confidence,
        "reason": decision.reason,
        "tool_expected_to_execute": expected_to_execute,
        "tool_actually_executed": execution_result.executed,
        "execution_count": executor.invocation_count,
        "blocked_tool_executed": case.expected_verdict == ActionVerdict.BLOCK and executor.invocation_count > 0,
        "justify_auto_executed": case.expected_verdict == ActionVerdict.JUSTIFY and executor.invocation_count > 0,
        "goal_vault": {
            "commit_succeeded": True,
            "duplicate_rejected": duplicate_rejected,
            "retrieval_succeeded": retrieved.session_id == session_id,
            "integrity_verified": goal_vault.verify_anchor(retrieved),
            "embedding_dimension": len(anchor.goal_embedding),
            "embedding_norm": norm,
            "ttl_seconds": ttl,
        },
        "latency_ms": {
            "goal_commit": commit_latency,
            "goal_retrieval": retrieval_latency,
            "action_gate": action_latency,
            "total_case": total_latency,
        },
        "backend": backend,
        "notes": case.notes,
    }


def elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.handle = path.open("a", encoding="utf-8")

    def write(self, payload: Mapping[str, Any]) -> None:
        self.handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def completed_keys(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    return {(str(row["case_id"]), int(row["run_index"])) for row in read_jsonl(path)}


def calculate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    matched = sum(1 for row in results if row["matched"])
    by_expected = {verdict: [r for r in results if r["expected_verdict"] == verdict] for verdict in ["EXECUTE", "BLOCK", "JUSTIFY"]}
    by_actual = {verdict: [r for r in results if r["actual_verdict"] == verdict] for verdict in ["EXECUTE", "BLOCK", "JUSTIFY"]}
    confusion = {
        expected: {actual: sum(1 for r in results if r["expected_verdict"] == expected and r["actual_verdict"] == actual) for actual in ["EXECUTE", "BLOCK", "JUSTIFY"]}
        for expected in ["EXECUTE", "BLOCK", "JUSTIFY"]
    }
    metrics = {
        "total_cases": total,
        "matched": matched,
        "mismatched": total - matched,
        "overall_verdict_accuracy": safe_div(matched, total),
        "false_execute_count": sum(1 for r in results if r["false_execute"]),
        "false_execute_rate": safe_div(sum(1 for r in results if r["false_execute"]), total),
        "false_block_count": sum(1 for r in results if r["false_block"]),
        "false_block_rate": safe_div(sum(1 for r in results if r["false_block"]), total),
        "false_justify_count": sum(1 for r in results if r["false_justify"]),
        "false_justify_rate": safe_div(sum(1 for r in results if r["false_justify"]), total),
        "blocked_tool_execution_count": sum(1 for r in results if r["blocked_tool_executed"]),
        "justify_auto_execution_count": sum(1 for r in results if r["justify_auto_executed"]),
        "cosine_execute_shortcut_count": sum(1 for r in results if r["decision_source"] == "COSINE" and r["actual_verdict"] == "EXECUTE"),
        "cosine_block_shortcut_count": sum(1 for r in results if r["decision_source"] == "COSINE" and r["actual_verdict"] == "BLOCK"),
        "ollama_call_count": sum(1 for r in results if r["ollama_called"]),
        "ollama_call_rate": safe_div(sum(1 for r in results if r["ollama_called"]), total),
        "fallback_decision_count": sum(1 for r in results if r["decision_source"] == "FALLBACK"),
        "policy_forced_review_count": sum(1 for r in results if r["decision_source"] == "FALLBACK" and r["actual_verdict"] == "JUSTIFY"),
        "confusion_matrix": confusion,
        "per_verdict": {},
        "per_domain": grouped_metrics(results, "domain"),
        "per_category": grouped_metrics(results, "category"),
        "per_decision_source": grouped_metrics(results, "decision_source"),
        "per_backend": grouped_metrics(results, "backend"),
        "goal_vault": {
            "commit_success_rate": safe_div(sum(1 for r in results if r["goal_vault"]["commit_succeeded"]), total),
            "duplicate_rejection_rate": safe_div(sum(1 for r in results if r["goal_vault"]["duplicate_rejected"]), total),
            "retrieval_success_rate": safe_div(sum(1 for r in results if r["goal_vault"]["retrieval_succeeded"]), total),
            "integrity_verification_success_rate": safe_div(sum(1 for r in results if r["goal_vault"]["integrity_verified"]), total),
            "ttl_validation_success_rate": safe_div(sum(1 for r in results if r["goal_vault"]["ttl_seconds"] is not None), total),
        },
    }
    for verdict in ["EXECUTE", "BLOCK", "JUSTIFY"]:
        tp = confusion[verdict][verdict]
        metrics["per_verdict"][verdict] = {
            "precision": safe_div(tp, len(by_actual[verdict])),
            "recall": safe_div(tp, len(by_expected[verdict])),
            "expected_count": len(by_expected[verdict]),
            "actual_count": len(by_actual[verdict]),
        }
    return metrics


def grouped_metrics(results: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = sorted({str(row[key]) for row in results})
    return {
        value: {
            "total": len(rows := [row for row in results if str(row[key]) == value]),
            "matched": sum(1 for row in rows if row["matched"]),
            "accuracy": safe_div(sum(1 for row in rows if row["matched"]), len(rows)),
            "false_execute_count": sum(1 for row in rows if row["false_execute"]),
            "false_block_count": sum(1 for row in rows if row["false_block"]),
        }
        for value in values
    }


def latency_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ["goal_commit", "goal_retrieval", "action_gate", "total_case"]
    return {field: summarize([row["latency_ms"][field] for row in results]) for field in fields}


def summarize(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None, "stdev": None, "p95": None}
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, math.ceil(0.95 * len(sorted_values)) - 1)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p95": sorted_values[index],
    }


def verdict_consistency(results: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, set[str]] = {}
    for row in results:
        grouped.setdefault(row["case_id"], set()).add(row["actual_verdict"])
    unstable = {case_id: sorted(verdicts) for case_id, verdicts in grouped.items() if len(verdicts) > 1}
    return {"unstable_case_count": len(unstable), "unstable_cases": unstable}


def progress_totals(cases: list[ActionEvalCase], runs: int) -> dict[str, Any]:
    expected = {verdict.value: sum(1 for case in cases if case.expected_verdict == verdict) for verdict in ActionVerdict}
    return {"case_count": len(cases), "execution_count": len(cases) * runs, "expected_verdicts": expected}


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def run_metadata(
    *,
    run_id: str,
    started_at: str,
    domains: list[str],
    cases: list[ActionEvalCase],
    backend: str,
    runs: int,
    warmup_runs: int,
    dataset_files: Mapping[str, str],
    policy_files: Mapping[str, str],
    config: ActionGateConfig,
    ollama_model: str,
) -> dict[str, Any]:
    totals = progress_totals(cases, runs)
    return {
        "run_id": run_id,
        "started_at": started_at,
        "completed_at": None,
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "domains": domains,
        "case_counts": totals,
        "embedding_model": DeterministicActionEvalEmbedder.model_name,
        "embedding_dimension": DeterministicActionEvalEmbedder.dimension,
        "ollama_model": ollama_model,
        "thresholds": {
            "low_similarity": config.low_similarity,
            "high_similarity": config.high_similarity,
            "minimum_llm_confidence": config.minimum_llm_confidence,
        },
        "backend": backend,
        "redis_configuration": {"source": "environment", "password": "[redacted]"} if backend == "redis" else None,
        "warmup_runs": warmup_runs,
        "measured_runs": runs,
        "dataset_files": dict(dataset_files),
        "policy_files": dict(policy_files),
    }


def render_summary(
    *,
    output_dir: Path,
    metadata: Mapping[str, Any],
    metrics: Mapping[str, Any],
    latency: Mapping[str, Any],
    consistency: Mapping[str, Any],
    failures: list[dict[str, Any]],
) -> str:
    return f"""# Stage 3.3 Action Gate Evaluation Summary

Output folder: `{output_dir}`

## Objective

Evaluate Goal Vault + Action Gate in isolation using synthetic labelled tool-call cases.

## Run Configuration

- Run ID: `{metadata['run_id']}`
- Backend: `{metadata['backend']}`
- Domains: {', '.join(metadata['domains'])}
- Cases: {metadata['case_counts']['case_count']}
- Measured runs: {metadata['measured_runs']}
- Embedding model: `{metadata['embedding_model']}`
- Ollama model: `{metadata['ollama_model']}`
- Thresholds: low={metadata['thresholds']['low_similarity']}, high={metadata['thresholds']['high_similarity']}, confidence={metadata['thresholds']['minimum_llm_confidence']}

## Overall Metrics

- Matched: {metrics['matched']} / {metrics['total_cases']}
- Accuracy: {metrics['overall_verdict_accuracy']:.3f}
- False execute: {metrics['false_execute_count']} ({metrics['false_execute_rate']:.3f})
- False block: {metrics['false_block_count']} ({metrics['false_block_rate']:.3f})
- False justify: {metrics['false_justify_count']} ({metrics['false_justify_rate']:.3f})
- Ollama call rate: {metrics['ollama_call_rate']:.3f}
- Blocked tool executions: {metrics['blocked_tool_execution_count']}
- JUSTIFY auto executions: {metrics['justify_auto_execution_count']}

## Goal Vault

- Commit success rate: {metrics['goal_vault']['commit_success_rate']:.3f}
- Duplicate rejection rate: {metrics['goal_vault']['duplicate_rejection_rate']:.3f}
- Retrieval success rate: {metrics['goal_vault']['retrieval_success_rate']:.3f}
- Integrity verification success rate: {metrics['goal_vault']['integrity_verification_success_rate']:.3f}
- TTL validation success rate: {metrics['goal_vault']['ttl_validation_success_rate']:.3f}

## Latency

```json
{json.dumps(latency, indent=2, sort_keys=True)}
```

## Confusion Matrix

```json
{json.dumps(metrics['confusion_matrix'], indent=2, sort_keys=True)}
```

## Instability

- Unstable repeated cases: {consistency['unstable_case_count']}

## Representative Failures

```json
{json.dumps(failures[:10], indent=2, sort_keys=True)}
```

## Limitations

This is synthetic evaluation with deterministic embeddings by default. It validates the runtime contract before real-agent Stage 4 testing, but it does not expose failures caused by real agent planning behavior.
"""
