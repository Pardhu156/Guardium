from __future__ import annotations

from pathlib import Path

from evaluation.sentinel.deterministic_validation import run_validation


def test_deterministic_validation_suite_passes_without_report() -> None:
    run = run_validation(write_report=False)
    assert run.metrics["failed"] == 0
    assert run.metrics["readiness"] == "READY FOR AGENTDOJO"
    assert run.metrics["layer0"]["false_positives"] == 0
    assert run.metrics["layer0"]["false_negatives"] == 0
    assert run.metrics["performance"]["embedding_model_instances"] == 1


def test_deterministic_validation_writes_report(tmp_path: Path) -> None:
    run = run_validation(output_dir=tmp_path)
    assert run.report_path is not None
    assert run.report_path.exists()
    assert (run.output_dir / "metrics.json").exists()
    assert (run.output_dir / "case_results.jsonl").exists()
    report = run.report_path.read_text(encoding="utf-8")
    assert "READY FOR AGENTDOJO" in report
    assert "Calibration Recommendations" in report
