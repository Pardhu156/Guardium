from __future__ import annotations

import json
from pathlib import Path

import pytest

from aegisvault.runtime.action_gate import ActionGateConfig, ActionVerdict
from evaluation.action_gate.scripts.action_eval_lib import (
    JsonlWriter,
    calculate_metrics,
    completed_keys,
    latency_summary,
    load_cases,
    load_policies,
    parse_case,
    progress_totals,
    read_jsonl,
    run_case,
    run_metadata,
    verdict_consistency,
)


DATASET_DIR = Path("evaluation/action_gate/datasets")
POLICY_DIR = Path("evaluation/action_gate/policies")


def test_dataset_loading_counts() -> None:
    cases, files = load_cases(DATASET_DIR, ["email_assistant", "ecommerce", "document_qa", "cross_domain"])
    assert len(cases) == 110
    assert len(files) == 4


def test_duplicate_id_detection(tmp_path: Path) -> None:
    path = tmp_path / "email_assistant_actions.jsonl"
    row = read_jsonl(DATASET_DIR / "email_assistant_actions.jsonl")[0]
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(tmp_path, ["email_assistant"])


def test_invalid_expected_verdict() -> None:
    row = read_jsonl(DATASET_DIR / "email_assistant_actions.jsonl")[0]
    row["expected_verdict"] = "MAYBE"
    with pytest.raises(ValueError):
        parse_case(row)


def test_policy_loading() -> None:
    policies, files = load_policies(POLICY_DIR, ["email_assistant", "document_qa"])
    assert policies["email_assistant"].application.name == "email-assistant"
    assert "document_qa.yaml" in files["document_qa"]


def test_run_case_execute_block_and_justify() -> None:
    cases, _ = load_cases(DATASET_DIR, ["email_assistant"])
    policies, _ = load_policies(POLICY_DIR, ["email_assistant"])
    config = ActionGateConfig(high_similarity=0.82, low_similarity=0.35, minimum_llm_confidence=0.75)

    execute = run_case(case=cases[0], policy=policies["email_assistant"], backend="memory", config=config, no_ollama=False, run_index=0)
    block = run_case(case=cases[12], policy=policies["email_assistant"], backend="memory", config=config, no_ollama=False, run_index=0)
    justify = run_case(case=cases[21], policy=policies["email_assistant"], backend="memory", config=config, no_ollama=False, run_index=0)

    assert execute["actual_verdict"] == "EXECUTE"
    assert execute["execution_count"] == 1
    assert block["actual_verdict"] == "BLOCK"
    assert block["execution_count"] == 0
    assert justify["actual_verdict"] == "JUSTIFY"
    assert justify["execution_count"] == 0


def test_goal_vault_checks_in_run_case() -> None:
    cases, _ = load_cases(DATASET_DIR, ["ecommerce"])
    policies, _ = load_policies(POLICY_DIR, ["ecommerce"])
    row = run_case(
        case=cases[0],
        policy=policies["ecommerce"],
        backend="memory",
        config=ActionGateConfig(),
        no_ollama=False,
        run_index=0,
    )
    assert row["goal_vault"]["duplicate_rejected"] is True
    assert row["goal_vault"]["integrity_verified"] is True
    assert row["goal_vault"]["embedding_dimension"] == 384
    assert row["goal_vault"]["ttl_seconds"] is not None


def test_metrics_false_counts_and_confusion_matrix() -> None:
    rows = [
        {"expected_verdict": "EXECUTE", "actual_verdict": "EXECUTE", "matched": True, "false_execute": False, "false_block": False, "false_justify": False, "blocked_tool_executed": False, "justify_auto_executed": False, "decision_source": "COSINE", "ollama_called": False, "domain": "d", "category": "c", "backend": "memory", "goal_similarity": 1.0, "goal_vault": {"commit_succeeded": True, "duplicate_rejected": True, "retrieval_succeeded": True, "integrity_verified": True, "ttl_seconds": 1}},
        {"expected_verdict": "BLOCK", "actual_verdict": "EXECUTE", "matched": False, "false_execute": True, "false_block": False, "false_justify": False, "blocked_tool_executed": True, "justify_auto_executed": False, "decision_source": "COSINE", "ollama_called": False, "domain": "d", "category": "c", "backend": "memory", "goal_similarity": 1.0, "goal_vault": {"commit_succeeded": True, "duplicate_rejected": True, "retrieval_succeeded": True, "integrity_verified": True, "ttl_seconds": 1}},
        {"expected_verdict": "EXECUTE", "actual_verdict": "BLOCK", "matched": False, "false_execute": False, "false_block": True, "false_justify": False, "blocked_tool_executed": False, "justify_auto_executed": False, "decision_source": "FALLBACK", "ollama_called": False, "domain": "x", "category": "k", "backend": "memory", "goal_similarity": 0.0, "goal_vault": {"commit_succeeded": True, "duplicate_rejected": True, "retrieval_succeeded": True, "integrity_verified": True, "ttl_seconds": 1}},
    ]
    metrics = calculate_metrics(rows)
    assert metrics["false_execute_count"] == 1
    assert metrics["false_block_count"] == 1
    assert metrics["blocked_tool_execution_count"] == 1
    assert metrics["confusion_matrix"]["BLOCK"]["EXECUTE"] == 1
    assert metrics["per_domain"]["d"]["total"] == 2


def test_latency_summary_and_p95() -> None:
    rows = [{"latency_ms": {"goal_commit": i, "goal_retrieval": i + 1, "action_gate": i + 2, "total_case": i + 3}} for i in range(1, 11)]
    summary = latency_summary(rows)
    assert summary["total_case"]["p95"] == 13
    assert summary["goal_commit"]["median"] == 5.5


def test_repeated_run_instability() -> None:
    rows = [{"case_id": "a", "actual_verdict": "EXECUTE"}, {"case_id": "a", "actual_verdict": "BLOCK"}]
    result = verdict_consistency(rows)
    assert result["unstable_case_count"] == 1


def test_jsonl_persistence_and_resume(tmp_path: Path) -> None:
    path = tmp_path / "case_results.jsonl"
    writer = JsonlWriter(path)
    writer.write({"case_id": "case-1", "run_index": 0})
    writer.close()
    assert read_jsonl(path) == [{"case_id": "case-1", "run_index": 0}]
    assert completed_keys(path) == {("case-1", 0)}


def test_progress_total_calculation() -> None:
    cases, _ = load_cases(DATASET_DIR, ["cross_domain"])
    totals = progress_totals(cases, runs=3)
    assert totals["case_count"] == 20
    assert totals["execution_count"] == 60
    assert sum(totals["expected_verdicts"].values()) == 20


def test_secret_redaction_in_run_metadata() -> None:
    cases, datasets = load_cases(DATASET_DIR, ["email_assistant"])
    _, policies = load_policies(POLICY_DIR, ["email_assistant"])
    metadata = run_metadata(
        run_id="run",
        started_at="now",
        domains=["email_assistant"],
        cases=cases,
        backend="redis",
        runs=1,
        warmup_runs=0,
        dataset_files=datasets,
        policy_files=policies,
        config=ActionGateConfig(),
        ollama_model="llama3.2",
    )
    serialized = json.dumps(metadata)
    assert "password" in serialized
    assert "[redacted]" in serialized
    assert "API_KEY" not in serialized


def test_expected_verdict_distribution_balanced() -> None:
    cases, _ = load_cases(DATASET_DIR, ["email_assistant", "ecommerce", "document_qa", "cross_domain"])
    counts = {verdict: sum(1 for case in cases if case.expected_verdict == verdict) for verdict in ActionVerdict}
    assert counts[ActionVerdict.EXECUTE] == 42
    assert counts[ActionVerdict.BLOCK] == 35
    assert counts[ActionVerdict.JUSTIFY] == 33
