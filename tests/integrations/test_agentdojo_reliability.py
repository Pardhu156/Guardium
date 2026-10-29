from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from agentdojo.agent_pipeline.llms.local_llm import LocalLLM
from openai import BadRequestError

from evaluation.agentdojo import run_pilot_benchmark as harness
from evaluation.agentdojo.inspect_run import inspect_run
from evaluation.agentdojo.reliability import (
    CaseObserver, CaseTimeout, CheckedLocalLLM, ContextWindowError, ModelExecutionError,
    ObservedElement, case_deadline, run_lock,
)


def _client(content: str = "Done", *, error: Exception | None = None, finish: str = "stop"):
    create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish, message=SimpleNamespace(content=content),
    )]), side_effect=error)
    client = Mock()
    client.with_options.return_value = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, create


def _query(llm):
    return llm.query("task", SimpleNamespace(functions={}), messages=[{"role": "user", "content": [{"type": "text", "content": "task"}]}])


def test_upstream_swallowed_http_error_is_not_a_safe_result():
    client, create = _client(error=TimeoutError("stalled request"))
    checked = CheckedLocalLLM(LocalLLM(client, "model"), 600, None)
    with pytest.raises(ModelExecutionError, match="TimeoutError"):
        _query(checked)
    create.assert_called_once()
    client.with_options.assert_called_once_with(timeout=600, max_retries=0)


def test_nested_context_error_surfaces_as_configuration_failure(tmp_path):
    detail = {"error": {"type": "exceed_context_size_error", "n_prompt_tokens": 13651, "n_ctx": 8192}}
    error = BadRequestError(
        "context error", response=httpx.Response(400, request=httpx.Request("POST", "http://localhost/v1/chat/completions")),
        body={"error": {"message": json.dumps(detail), "type": "invalid_request_error"}},
    )
    client, create = _client(error=error)
    observer = CaseObserver(tmp_path, "protected", "case-a", 30)
    checked = CheckedLocalLLM(LocalLLM(client, "model"), 600, None, observer)
    with pytest.raises(ContextWindowError) as caught:
        _query(checked)
    assert caught.value.prompt_tokens == 13651
    assert caught.value.context_tokens == 8192
    create.assert_called_once()
    events = [json.loads(line) for line in (tmp_path / "protected_events.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "context_window_error"


def test_ordinary_400_error_is_not_misclassified_as_context_error():
    error = BadRequestError(
        "invalid parameter", response=httpx.Response(400, request=httpx.Request("POST", "http://localhost/v1/chat/completions")),
        body={"error": {"message": "Invalid parameter", "type": "invalid_request_error"}},
    )
    client, _ = _client(error=error)
    with pytest.raises(ModelExecutionError) as caught:
        _query(CheckedLocalLLM(LocalLLM(client, "model"), 600, None))
    assert not isinstance(caught.value, ContextWindowError)


@pytest.mark.parametrize("content,finish", [("", "stop"), ('<function=send_email>{"x": }</function>', "stop"), ("partial", "length")])
def test_invalid_or_truncated_generation_is_a_runtime_failure(content, finish):
    client, _ = _client(content, finish=finish)
    with pytest.raises(ModelExecutionError):
        _query(CheckedLocalLLM(LocalLLM(client, "model"), 600, None))


def test_valid_generation_preserves_decoding_and_official_parser():
    client, create = _client('<function=search_email>{"query": "meeting"}</function>')
    result = _query(CheckedLocalLLM(LocalLLM(client, "model"), 600, None))
    assert result[3][-1]["tool_calls"][0].function == "search_email"
    assert create.call_args.kwargs["temperature"] == 0
    assert "max_tokens" not in create.call_args.kwargs


def test_output_cap_is_explicit_only():
    client, create = _client()
    _query(CheckedLocalLLM(LocalLLM(client, "model"), 600, 1024))
    assert create.call_args.kwargs["max_tokens"] == 1024


def test_case_deadline_escapes_upstream_exception_handler():
    with pytest.raises(CaseTimeout):
        with case_deadline(0.02):
            try:
                time.sleep(0.1)
            except Exception:
                pytest.fail("deadline was swallowed")
    with case_deadline(0):
        pass


def test_heartbeat_and_interruption_persist_current_stage(tmp_path, capsys):
    with pytest.raises(KeyboardInterrupt):
        with CaseObserver(tmp_path, "protected", "case-a", 0.01) as observer:
            observer.update("Qwen generation #1")
            time.sleep(0.04)
            raise KeyboardInterrupt
    status = json.loads((tmp_path / "protected_current_case.json").read_text())
    assert status["status"] == "interrupted"
    assert status["stage"] == "Qwen generation #1"
    events = [json.loads(line) for line in (tmp_path / "protected_events.jsonl").read_text().splitlines()]
    assert any(event["event"] == "heartbeat" for event in events)


def test_observed_element_preserves_result_and_checkpoints_messages(tmp_path):
    messages = [{"role": "assistant", "content": [{"type": "text", "content": "done"}]}]
    result = ("query", None, None, messages, {})
    observer = CaseObserver(tmp_path, "baseline", "case-a", 30)
    element = ObservedElement(SimpleNamespace(query=lambda: result), observer, "Qwen generation")
    assert element.query() == result
    events = [json.loads(line) for line in (tmp_path / "baseline_events.jsonl").read_text().splitlines()]
    assert next(event for event in events if event["event"] == "messages")["messages"] == messages
    assert events[-1]["event"] == "step_completed"
    assert events[-1]["completed_steps"] == 1


def test_completed_steps_and_tool_execution_are_visible(tmp_path, capsys):
    observer = CaseObserver(tmp_path, "protected", "case-a", 30)
    observer.update("Qwen generation #1")
    observer.event("step_completed", duration_seconds=120.0)
    observer.event("tool_decision", trace={"tool_name": "search", "final_result": "EXECUTE", "executed": True})
    observer.event("tool_decision", trace={"tool_name": "send", "final_result": "BLOCK", "executed": False})
    output = capsys.readouterr().out
    assert "completed Qwen generation #1 in 120.0s" in output
    assert "tool search -> EXECUTE" in output
    assert "tool send -> BLOCK" in output
    checkpoint = json.loads((tmp_path / "protected_current_case.json").read_text())
    assert checkpoint["executed_tools"] == 1
    assert checkpoint["completed_steps"] == 1


def test_many_short_model_steps_do_not_equal_one_stalled_request(tmp_path):
    result = ("query", None, None, [{"role": "assistant", "content": [{"type": "text", "content": "done"}]}], {})
    observer = CaseObserver(tmp_path, "protected", "case-a", 30)
    element = ObservedElement(SimpleNamespace(query=lambda: result), observer, "Qwen generation")
    for _ in range(8):
        assert element.query() == result
    assert observer.completed_steps == 8
    checkpoint = json.loads((tmp_path / "protected_current_case.json").read_text())
    assert checkpoint["stage"] == "Qwen generation #8"
    assert checkpoint["completed_steps"] == 8


def test_failed_cases_excluded_from_asr_and_middleware_fp():
    rows = [
        {"case_id": "safe", "suite": "travel", "case_type": "attack", "utility": True, "injection_success": False},
        {"case_id": "unsafe", "suite": "travel", "case_type": "attack", "utility": False, "injection_success": True},
        {"case_id": "manual", "suite": "travel", "case_type": "attack", "utility": False, "injection_success": False, "error": "ManualSkip"},
        {"case_id": "clean-error", "suite": "travel", "case_type": "benign", "utility": None, "error": "Timeout"},
    ]
    metrics = harness._phase_metrics(rows)
    assert metrics["attack_success_rate"] == 0.5
    assert metrics["attempted_attack_cases"] == 3
    assert metrics["unevaluated_attack_cases"] == 1
    assert metrics["asr_lower_bound_over_attempted"] == pytest.approx(1 / 3)
    assert metrics["asr_upper_bound_over_attempted"] == pytest.approx(2 / 3)
    assert metrics["runtime_failure_count"] == 2
    assert metrics["middleware_false_positives"] == 0
    assert metrics["model_utility_failures"] == 0
    assert metrics["per_suite"]["travel"]["asr"] == 0.5
    assert "manual" not in metrics["failure_breakdown"]["security_success_cases"]


def test_retry_only_failed_cases_without_duplicate_results(tmp_path, monkeypatch):
    cases = [harness.PilotCase(name, "travel", "user_task_0", None, None) for name in ["done", "retry", "new"]]
    rows = {
        "done": {"phase": "protected", "case_id": "done", "suite": "travel", "case_type": "benign", "utility": True},
        "retry": {"phase": "protected", "case_id": "retry", "suite": "travel", "case_type": "benign", "utility": False, "error": "ManualSkip"},
    }
    path = tmp_path / "protected_results.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows.values()))
    original = path.read_bytes()
    executed = []
    bars = []
    initial_counts = []
    descriptions = []
    original_tqdm = harness.tqdm

    def progress_bar(*args, **kwargs):
        initial_counts.append(kwargs["initial"])
        descriptions.append(kwargs["desc"])
        bar = original_tqdm(*args, **kwargs, disable=True)
        # tqdm skips its own counter updates when disabled; keep them observable.
        bar.update = lambda n=1: setattr(bar, "n", bar.n + n)
        bars.append(bar)
        return bar

    monkeypatch.setattr(harness, "tqdm", progress_bar)

    def run(phase, case, args):
        executed.append(case.case_id)
        return {"phase": phase, "case_id": case.case_id, "suite": case.suite, "case_type": case.case_type, "utility": True}

    monkeypatch.setattr(harness, "_run_case", run)
    args = harness._parse_args(["--retry-errors"])
    results = harness._run_phase("protected", cases, args, tmp_path)
    assert executed == ["new", "retry"]
    assert json.loads((tmp_path / "protected_deferred_cases.json").read_text())["case_ids"] == ["retry"]
    assert len(results) == 3
    assert len(harness._read_existing(path)) == 3
    summary = (tmp_path / "protected_summary_live.md").read_text()
    assert "Live snapshot" in summary
    assert "Evaluated cases: 3" in summary
    assert "Candidate middleware false positives" in summary
    assert initial_counts == [0, 0]
    assert [(bar.total, bar.n) for bar in bars] == [(1, 1), (1, 1)]
    assert descriptions == ["protected AgentDojo new cases", "protected AgentDojo retries"]
    assert path.with_suffix(".before_recovery.jsonl").read_bytes() == original
    harness._run_phase("protected", cases, args, tmp_path)
    assert executed == ["new", "retry"]
    assert initial_counts[2] == 0
    assert bars[2].total == bars[2].n == 0


def test_retry_only_progress_counts_attempts_even_when_retry_fails(tmp_path, monkeypatch):
    case = harness.PilotCase("retry", "travel", "user_task_0", None, None)
    path = tmp_path / "protected_results.jsonl"
    path.write_text(json.dumps({"case_id": "retry", "phase": "protected", "suite": "travel", "case_type": "benign", "error": "previous failure"}) + "\n")
    bars = []
    descriptions = []
    original_tqdm = harness.tqdm

    def progress_bar(*args, **kwargs):
        descriptions.append(kwargs["desc"])
        bar = original_tqdm(*args, **kwargs, disable=True)
        bar.update = lambda n=1: setattr(bar, "n", bar.n + n)
        bars.append(bar)
        return bar

    def fail(phase, case, args):
        raise RuntimeError("still unavailable")

    monkeypatch.setattr(harness, "tqdm", progress_bar)
    monkeypatch.setattr(harness, "_run_case", fail)
    rows = harness._run_phase("protected", [case], harness._parse_args(["--retry-errors"]), tmp_path)
    assert len(bars) == 1
    assert descriptions == ["protected AgentDojo retries"]
    assert bars[0].total == bars[0].n == 1
    assert len(rows) == len(harness._read_existing(path)) == 1
    assert rows[0]["status"] == "runtime_failure"


def test_case_timeout_is_saved_with_unknown_outcomes_and_partial_traces(tmp_path, monkeypatch):
    case = harness.PilotCase("case-a", "travel", "user_task_0", "injection_task_0", "direct")

    def run(phase, case, args):
        args._middleware_counts = {"action_traces": [{"tool_name": "search", "final_result": "EXECUTE"}]}
        raise CaseTimeout("deadline")

    monkeypatch.setattr(harness, "_run_case", run)
    row = harness._run_phase("protected", [case], harness._parse_args([]), tmp_path)[0]
    assert row["injection_success"] is None
    assert row["status"] == "runtime_failure"
    assert row["middleware"]["action_traces"]
    assert harness._phase_metrics([row])["evaluated_cases"] == 0


def test_context_limit_is_persisted_then_stops_before_later_cases(tmp_path, monkeypatch):
    cases = [harness.PilotCase(name, "workspace", "user_task_35", "injection_task_0", "direct") for name in ["large", "later"]]
    called = []

    def run(phase, case, args):
        called.append(case.case_id)
        raise ContextWindowError(13651, 8192)

    monkeypatch.setattr(harness, "_run_case", run)
    with pytest.raises(ContextWindowError):
        harness._run_phase("protected", cases, harness._parse_args([]), tmp_path)
    rows = harness._read_existing(tmp_path / "protected_results.jsonl")
    assert called == ["large"]
    assert list(rows) == ["large"]
    assert rows["large"]["injection_success"] is None
    assert rows["large"]["failure_category"] == "infrastructure_context_limit"
    assert rows["large"]["prompt_tokens"] == 13651
    metrics = json.loads((tmp_path / "protected_metrics_live.json").read_text())
    assert metrics["evaluated_cases"] == 0


def test_first_result_is_persisted_in_a_fresh_run_directory(tmp_path, monkeypatch):
    case = harness.PilotCase("first", "workspace", "user_task_0", None, None)
    monkeypatch.setattr(harness, "_run_case", lambda phase, case, args: {
        "phase": phase, "case_id": case.case_id, "suite": case.suite,
        "case_type": case.case_type, "utility": True, "injection_success": False,
    })
    harness._run_phase("protected", [case], harness._parse_args([]), tmp_path)
    rows = harness._read_existing(tmp_path / "protected_results.jsonl")
    assert rows["first"]["status"] == "completed"
    assert rows["first"]["utility"] is True


def test_cli_exits_clearly_on_context_configuration_error(tmp_path, monkeypatch, capsys):
    def run(*args):
        raise ContextWindowError(13651, 8192)

    monkeypatch.setattr(harness, "_main", run)
    assert harness.main(["--output-dir", str(tmp_path)]) == 2
    assert "larger OLLAMA_CONTEXT_LENGTH" in capsys.readouterr().err


def test_ctrl_c_leaves_unfinished_case_pending(tmp_path, monkeypatch):
    def interrupted(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(harness, "_run_case", interrupted)
    case = harness.PilotCase("case-a", "travel", "user_task_0", None, None)
    with pytest.raises(KeyboardInterrupt):
        harness._run_phase("protected", [case], harness._parse_args([]), tmp_path)
    assert harness._read_existing(tmp_path / "protected_results.jsonl") == {}


def test_interrupted_final_line_recovered_but_middle_corruption_rejected(tmp_path):
    path = tmp_path / "results.jsonl"
    path.write_text('{"case_id":"done"}\n{"case_id":')
    assert list(harness._read_existing(path)) == ["done"]
    assert path.with_name("results.jsonl.truncated_backup").exists()
    path.write_text('{bad}\n{"case_id":"done"}\n')
    with pytest.raises(ValueError, match="refusing to discard"):
        harness._read_existing(path)


def test_two_runners_cannot_write_same_folder(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another benchmark"):
            with run_lock(tmp_path):
                pytest.fail("lock should prevent concurrent writers")


def test_verified_production_embedder_is_reused():
    embedder = SimpleNamespace(model_name="all-MiniLM-L6-v2", dimension=384, embed=Mock(return_value=[1.0] * 384))
    assert harness._verify_production_embedder(embedder)["dimension"] == 384
    embedder.embed.assert_called_once()


def test_inspector_uses_unique_ids_and_excludes_skips(tmp_path):
    path = tmp_path / "protected_results.jsonl"
    path.write_text('\n'.join(json.dumps(row) for row in [
        {"case_id": "a", "suite": "travel", "case_type": "attack", "injection_success": False},
        {"case_id": "a", "suite": "travel", "case_type": "attack", "injection_success": True},
        {"case_id": "b", "suite": "travel", "case_type": "attack", "error": "ManualSkip"},
    ]))
    (tmp_path / "selected_cases.json").write_text(json.dumps({"cases": [{"case_id": "a"}, {"case_id": "b"}, {"case_id": "c"}]}))
    result = inspect_run(tmp_path)
    assert result["saved_unique_cases"] == 2
    assert result["attack_success_rate"] == 1.0
    assert result["remaining_including_retries"] == 2
