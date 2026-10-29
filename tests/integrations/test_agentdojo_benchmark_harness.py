from __future__ import annotations

import json
from collections import Counter

import pytest

from aegisvault.runtime.goal_vault import GoalEmbeddingError
from evaluation.agentdojo import run_pilot_benchmark as harness


def test_agentdojo_suite_specific_tool_metadata_marks_send_as_risky() -> None:
    metadata = harness._agentdojo_tool_metadata("slack", object(), "send_slack_message")

    assert metadata.risk_level == "medium"
    assert metadata.side_effect_level.value == "write"
    assert metadata.requires_approval is True
    assert "strict_verification" in metadata.required_permissions


def test_agentdojo_suite_specific_tool_metadata_marks_search_as_low_risk() -> None:
    metadata = harness._agentdojo_tool_metadata("workspace", object(), "search_workspace")

    assert metadata.risk_level == "low"
    assert metadata.side_effect_level.value == "read"
    assert metadata.requires_approval is False


def test_agentdojo_banking_read_tools_are_not_treated_as_financial_writes() -> None:
    metadata = harness._agentdojo_tool_metadata("banking", object(), "get_most_recent_transactions")

    assert metadata.risk_level == "low"
    assert metadata.side_effect_level.value == "read"
    assert metadata.requires_approval is False


def test_agentdojo_banking_money_movement_remains_strict() -> None:
    metadata = harness._agentdojo_tool_metadata("banking", object(), "send_money")

    assert metadata.risk_level == "high"
    assert metadata.side_effect_level.value == "write"
    assert metadata.requires_approval is True
    assert "strict_verification" in metadata.required_permissions


def test_agentdojo_slack_read_messages_are_low_risk_reads() -> None:
    metadata = harness._agentdojo_tool_metadata("slack", object(), "read_channel_messages")

    assert metadata.risk_level == "low"
    assert metadata.side_effect_level.value == "read"
    assert metadata.requires_approval is False


def test_agentdojo_slack_send_messages_remain_strict() -> None:
    metadata = harness._agentdojo_tool_metadata("slack", object(), "send_channel_message")

    assert metadata.risk_level == "medium"
    assert metadata.side_effect_level.value == "write"
    assert metadata.requires_approval is True
    assert "strict_verification" in metadata.required_permissions


def test_agentdojo_slack_invites_are_not_read_only_fast_path() -> None:
    metadata = harness._agentdojo_tool_metadata("slack", object(), "invite_user_to_slack")

    assert metadata.risk_level == "medium"
    assert metadata.side_effect_level.value == "write"
    assert metadata.requires_approval is True


def test_agentdojo_travel_reservations_require_strict_verification() -> None:
    metadata = harness._agentdojo_tool_metadata("travel", object(), "reserve_hotel")

    assert metadata.risk_level == "high"
    assert metadata.side_effect_level.value == "write"
    assert metadata.requires_approval is True


def test_duplicate_result_rows_are_compacted(tmp_path) -> None:
    path = tmp_path / "protected_results.jsonl"
    rows = {
        "case-a": {"case_id": "case-a", "value": 2},
        "case-b": {"case_id": "case-b", "value": 3},
    }
    path.write_text(
        "\n".join(
            [
                json.dumps({"case_id": "case-a", "value": 1}),
                json.dumps({"case_id": "case-a", "value": 2}),
                json.dumps({"case_id": "case-b", "value": 3}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    harness._rewrite_unique_jsonl(path, rows)

    compacted = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert compacted == [{"case_id": "case-a", "value": 2}, {"case_id": "case-b", "value": 3}]


def test_scaled_sampler_selects_clean_and_attack_cases_without_duplicates() -> None:
    cases = harness.select_cases(
        limit=None,
        suites=("workspace", "slack", "banking", "travel"),
        attack="direct",
        clean_limit=20,
        attack_limit=100,
        balanced_by_suite=True,
        seed=7,
        benchmark_version="v1.2.2",
    )

    assert len(cases) == 120
    assert sum(1 for case in cases if case.case_type == "benign") == 20
    assert sum(1 for case in cases if case.case_type == "attack") == 100
    assert len({case.case_id for case in cases}) == 120
    clean_by_suite = Counter(case.suite for case in cases if case.case_type == "benign")
    attack_by_suite = Counter(case.suite for case in cases if case.case_type == "attack")
    assert clean_by_suite == {"workspace": 5, "slack": 5, "banking": 5, "travel": 5}
    assert attack_by_suite == {"workspace": 25, "slack": 25, "banking": 25, "travel": 25}


def test_scaled_sampler_can_interleave_clean_and_attack_cases() -> None:
    cases = harness.select_cases(
        limit=None,
        suites=("workspace", "slack", "banking", "travel"),
        attack="direct",
        clean_limit=4,
        attack_limit=4,
        balanced_by_suite=True,
        case_layout="interleave-types",
        seed=7,
        benchmark_version="v1.2.2",
    )

    assert len(cases) == 8
    assert [case.case_type for case in cases[:4]] == ["benign", "attack", "benign", "attack"]


def test_case_record_marks_legit_cases_with_null_injection() -> None:
    case = harness.select_cases(
        limit=1,
        suites=("workspace",),
        attack="direct",
        clean_limit=1,
        attack_limit=0,
        balanced_by_suite=True,
        seed=7,
        benchmark_version="v1.2.2",
    )[0]

    record = harness._case_record(case)
    assert record["case_type"] == "benign"
    assert record["injection_task_id"] is None


def test_first_selection_strategy_uses_agentdojo_order() -> None:
    cases = harness.select_cases(
        limit=None,
        suites=("workspace", "slack"),
        attack="direct",
        clean_limit=3,
        attack_limit=2,
        selection_strategy="first",
        seed=7,
        benchmark_version="v1.2.2",
    )

    assert [case.case_id for case in cases] == [
        "workspace_benign_user_task_0",
        "workspace_benign_user_task_1",
        "workspace_benign_user_task_3",
        "workspace_direct_user_task_0_injection_task_0",
        "workspace_direct_user_task_0_injection_task_1",
    ]


def test_missing_real_embedder_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenEmbedder:
        model_name = "all-MiniLM-L6-v2"
        dimension = 384

        def embed(self, text: str):
            raise GoalEmbeddingError("model unavailable")

    monkeypatch.setattr(harness, "_production_embedder", lambda: BrokenEmbedder())

    with pytest.raises(SystemExit, match="Production embedder unavailable"):
        harness._verify_production_embedder()


def test_action_rows_include_flattened_decision_fields(tmp_path) -> None:
    path = tmp_path / "actions.jsonl"
    row = {
        "phase": "protected",
        "case_id": "case-a",
        "suite": "workspace",
        "case_type": "benign",
        "user_task_id": "user_task_1",
        "injection_task_id": None,
        "utility": True,
        "injection_success": False,
        "middleware": {
            "action_traces": [
                {
                    "tool_name": "search_files",
                    "risk_classification": {
                        "risk_level": "low",
                        "side_effect_level": "read",
                        "requires_approval": False,
                    },
                    "sentinel": {"decision": "allow"},
                    "action_gate": {
                        "verdict": "EXECUTE",
                        "decision_source": "COSINE",
                        "ollama_called": False,
                        "goal_similarity": 0.91,
                    },
                    "final_result": "EXECUTE",
                    "executed": True,
                }
            ]
        },
    }

    harness._append_action_rows(path, row)

    written = json.loads(path.read_text(encoding="utf-8").strip())
    assert written["risk_level"] == "low"
    assert written["side_effect_level"] == "read"
    assert written["sentinel_verdict"] == "allow"
    assert written["action_gate_verdict"] == "EXECUTE"
    assert written["ollama_called"] is False


def test_metrics_separate_middleware_fp_from_model_utility_failure() -> None:
    rows = [
        {
            "case_id": "clean-model-fail",
            "suite": "workspace",
            "case_type": "benign",
            "utility": False,
            "middleware": {"action_traces": [{"final_result": "EXECUTE"}]},
        },
        {
            "case_id": "clean-blocked",
            "suite": "workspace",
            "case_type": "benign",
            "utility": False,
            "middleware": {"action_traces": [{"final_result": "BLOCK"}]},
        },
        {
            "case_id": "attack-safe",
            "suite": "workspace",
            "case_type": "attack",
            "utility": True,
            "injection_success": False,
            "middleware": {"action_traces": [{"final_result": "EXECUTE"}]},
        },
    ]

    metrics = harness._phase_metrics(rows)

    assert metrics["false_positives"] == 1
    assert metrics["middleware_false_positives"] == 1
    assert metrics["model_utility_failures"] == 1
    assert metrics["failure_breakdown"]["middleware_false_positive_cases"] == ["clean-blocked"]
    assert metrics["failure_breakdown"]["model_utility_failure_cases"] == ["clean-model-fail"]


def test_resume_eta_counts_only_this_sessions_attempts(monkeypatch, capsys) -> None:
    monkeypatch.setattr(harness.time, "perf_counter", lambda: 20.0)
    harness._print_progress(
        "protected", 800, 1046, [], 0.0, current_suite="workspace",
        session_attempts=2, pending_remaining=9,
    )
    assert "eta=90.0s" in capsys.readouterr().out


@pytest.mark.parametrize("option,value", [
    ("--heartbeat-seconds", "nan"),
    ("--llm-timeout-seconds", "inf"),
    ("--action-timeout-seconds", "0"),
    ("--action-timeout-seconds", "-1"),
    ("--case-timeout-seconds", "nan"),
    ("--case-timeout-seconds", "inf"),
])
def test_invalid_runtime_budgets_rejected_before_execution(option, value) -> None:
    with pytest.raises(SystemExit) as caught:
        harness._parse_args([option, value])
    assert caught.value.code == 2


def _fake_preflight(monkeypatch, installed):
    monkeypatch.setenv("LOCAL_LLM_PORT", "11434")

    def get(url, **kwargs):
        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                if url.endswith("/v1/models"):
                    return {"data": [{"id": name} for name in installed]}
                return {"models": [{"name": "qwen3:4b-instruct", "context_length": 16384}]}

        return Response()

    monkeypatch.setattr(harness.requests, "get", get)


def test_preflight_accepts_installed_gate_models_with_latest_alias(monkeypatch) -> None:
    _fake_preflight(monkeypatch, ["qwen3:4b-instruct", "llama3.2:latest"])
    args = harness._parse_args(["--phase", "protected"])
    harness._preflight(args)
    assert args._ollama_context_tokens == 16384


def test_preflight_refuses_missing_action_verifier_before_benchmark(monkeypatch) -> None:
    _fake_preflight(monkeypatch, ["qwen3:4b-instruct"])
    args = harness._parse_args(["--phase", "protected"])
    with pytest.raises(SystemExit, match="Action Gate model 'llama3.2'"):
        harness._preflight(args)


def test_baseline_does_not_require_action_verifier(monkeypatch) -> None:
    _fake_preflight(monkeypatch, ["qwen3:4b-instruct"])
    args = harness._parse_args(["--phase", "baseline"])
    harness._preflight(args)
