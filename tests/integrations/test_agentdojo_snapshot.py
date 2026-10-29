from __future__ import annotations

import json

import pytest

from evaluation.agentdojo.export_snapshot import export_snapshot
from evaluation.agentdojo import run_pilot_benchmark as harness


def _source(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    rows = [
        {"case_id": "clean", "suite": "workspace", "case_type": "benign", "utility": True},
        {"case_id": "attack", "suite": "workspace", "case_type": "attack", "injection_success": True},
        {"case_id": "unknown", "suite": "workspace", "case_type": "attack", "user_task_id": "user_task_1", "injection_task_id": "injection_task_1", "status": "runtime_failure", "error": "APITimeoutError secret-token", "prompt": "private input"},
    ]
    (run / "protected_results.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (run / "protected_metrics_live.json").write_text(json.dumps({"cases": 3, "evaluated_cases": 2, "middleware_false_positives": 0, "model_utility_failures": 0, "api_key": "secret-token"}))
    (run / "run_metadata.json").write_text(json.dumps({"benchmark_version": "v1.2.2", "attack": "direct", "environment": "secret-token"}))
    return run


def test_public_snapshot_excludes_raw_content_and_preserves_unknowns(tmp_path):
    run = _source(tmp_path)
    out = tmp_path / "public"
    result = export_snapshot(run, out)
    assert result["saved_cases"] == 3
    assert result["evaluated_cases"] == 2
    assert result["unresolved_cases"] == 1
    assert result["conditional_attack_success_rate"] == 1.0
    assert result["asr_lower_bound_over_attempted"] == 0.5
    assert result["asr_upper_bound_over_attempted"] == 1.0
    assert result["unresolved_by_category"] == {"model_request_timeout": 1}
    assert result["per_suite"]["workspace"] == {"saved": 3, "evaluated": 2, "unresolved": 1}
    for file in out.iterdir():
        assert "secret-token" not in file.read_text()
        assert "private input" not in file.read_text()
    manifest = json.loads((out / "unresolved_cases.json").read_text())
    assert manifest["case_ids"] == ["unknown"]
    assert harness._parse_args(["--case-ids-file", str(out / "unresolved_cases.json")]).case_ids == ["unknown"]


def test_snapshot_refuses_overwrite(tmp_path):
    run = _source(tmp_path)
    out = tmp_path / "public"
    export_snapshot(run, out)
    original = (out / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        export_snapshot(run, out)
    assert (out / "summary.json").read_bytes() == original


def test_snapshot_rejects_inconsistent_metrics(tmp_path):
    run = _source(tmp_path)
    (run / "protected_metrics_live.json").write_text(json.dumps({"cases": 3, "evaluated_cases": 3}))
    with pytest.raises(ValueError, match="disagree"):
        export_snapshot(run, tmp_path / "public")
    assert not (tmp_path / "public").exists()


@pytest.mark.parametrize("ids", [[], ["a", "a"], [1], "a", [""]])
def test_case_manifest_rejects_invalid_ids(tmp_path, ids):
    path = tmp_path / "ids.json"
    path.write_text(json.dumps({"case_ids": ids}))
    with pytest.raises(SystemExit):
        harness._parse_args(["--case-ids-file", str(path)])


def test_case_filter_preserves_original_order_and_rejects_unknown_ids():
    cases = [harness.PilotCase(name, "workspace", "user_task_0", None, None) for name in ["a", "b", "c"]]
    assert harness._filter_case_ids(cases, None) == cases
    assert [case.case_id for case in harness._filter_case_ids(cases, ["c", "a"])] == ["a", "c"]
    with pytest.raises(ValueError, match="not present"):
        harness._filter_case_ids(cases, ["typo"])
