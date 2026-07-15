from __future__ import annotations

import json

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


def test_missing_real_embedder_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenEmbedder:
        model_name = "all-MiniLM-L6-v2"
        dimension = 384

        def embed(self, text: str):
            raise GoalEmbeddingError("model unavailable")

    monkeypatch.setattr(harness, "_production_embedder", lambda: BrokenEmbedder())

    with pytest.raises(SystemExit, match="Production embedder unavailable"):
        harness._verify_production_embedder()
