from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from pydantic import BaseModel
from agentdojo.agent_pipeline.llms.local_llm import LocalLLM
from openai import APIConnectionError, APITimeoutError, OpenAI

from evaluation.agentdojo import run_pilot_benchmark as harness
from evaluation.agentdojo.reliability import (
    CaseObserver, CheckedLocalLLM, ModelExecutionError, ModelServiceUnavailableError,
)


def chunk(text=None, finish=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        index=0, delta=SimpleNamespace(content=text), finish_reason=finish,
    )])


class FakeStream:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __iter__(self):
        for item in self.chunks:
            if isinstance(item, BaseException):
                raise item
            yield item

    def close(self):
        self.closed = True


def actor(create, observer=None, **options):
    client = Mock()
    client.with_options.return_value = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return CheckedLocalLLM(LocalLLM(client, "qwen"), 600, None, observer, **options)


def query(llm):
    return llm.query("Original task", SimpleNamespace(functions={}), messages=[])


def response(text):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=text))])


def events(tmp_path):
    return [json.loads(line) for line in (tmp_path / "protected_events.jsonl").read_text().splitlines()]


def test_stream_reassembled_before_parsing_and_dangerous_arguments_unchanged(tmp_path):
    text = '<function=send_money>{"recipient":"attacker","amount":1000}</function>'
    stream = FakeStream([chunk(text[:30]), chunk(text[30:]), chunk(finish="stop")])
    create = Mock(return_value=stream)
    observer = CaseObserver(tmp_path, "protected", "case", 30)
    result = query(actor(create, observer, stream_output=True))
    call = result[3][-1]["tool_calls"][0]
    assert call.function == "send_money"
    assert call.args == {"recipient": "attacker", "amount": 1000}
    assert stream.closed
    create.assert_called_once()
    assert create.call_args.kwargs["stream"] is True
    assert create.call_args.kwargs["temperature"] == 0
    assert "max_tokens" not in create.call_args.kwargs
    assert any(e["event"] == "generation_progress" and e["received_characters"] > 0 for e in events(tmp_path))
    assert next(e for e in events(tmp_path) if e["event"] == "model_response")["raw_content"] == text


@pytest.mark.parametrize("finish", [None, "length"])
def test_incomplete_stream_is_never_parsed_or_retried(tmp_path, finish):
    stream = FakeStream([chunk('<function=send_money>{"amount":1000}</function>', finish)])
    create = Mock(return_value=stream)
    with pytest.raises(ModelExecutionError):
        query(actor(create, CaseObserver(tmp_path, "protected", "case", 30), stream_output=True, format_repair_attempts=1))
    create.assert_called_once()
    assert stream.closed
    assert not any(e["event"] == "format_retry" for e in events(tmp_path))


def test_stream_timeout_retains_partial_content_and_is_not_retried(tmp_path):
    error = APITimeoutError(request=httpx.Request("POST", "http://localhost/v1/chat/completions"))
    stream = FakeStream([chunk("partial proposal"), error])
    create = Mock(return_value=stream)
    with pytest.raises(ModelExecutionError, match="APITimeoutError"):
        query(actor(create, CaseObserver(tmp_path, "protected", "case", 30), stream_output=True))
    assert stream.closed
    create.assert_called_once()
    assert next(e for e in events(tmp_path) if e["event"] == "model_partial_response")["raw_content"] == "partial proposal"


def test_stream_closed_on_interrupt(tmp_path):
    stream = FakeStream([chunk("partial"), KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        query(actor(Mock(return_value=stream), CaseObserver(tmp_path, "protected", "case", 30), stream_output=True))
    assert stream.closed


def test_format_regeneration_is_bounded_logged_and_does_not_mutate_first_request(tmp_path):
    bad = '<function=send_money>{"recipient":"attacker","amount":1000}}</function>'
    good = '<function=send_money>{"recipient":"attacker","amount":1000}</function>'
    create = Mock(side_effect=[response(bad), response(good)])
    result = query(actor(create, CaseObserver(tmp_path, "protected", "case", 30), format_repair_attempts=1))
    assert result[3][-1]["tool_calls"][0].args == {"recipient": "attacker", "amount": 1000}
    assert create.call_count == 2
    first, second = [call.kwargs for call in create.call_args_list]
    assert second["messages"][:-2] == first["messages"]
    assert second["messages"][-2] == {"role": "assistant", "content": bad}
    assert "NOT executed" in second["messages"][-1]["content"]
    assert first["temperature"] == second["temperature"] == 0
    assert first["seed"] == second["seed"]
    assert [e["raw_content"] for e in events(tmp_path) if e["event"] == "model_response"] == [bad, good]
    assert len([e for e in events(tmp_path) if e["event"] == "format_retry"]) == 1


def test_second_malformed_proposal_fails_closed_without_third_request():
    bad = '<function=send_money>{"amount":1000}}</function>'
    create = Mock(side_effect=[response(bad), response(bad)])
    with pytest.raises(ModelExecutionError, match="Invalid tool-call format"):
        query(actor(create, format_repair_attempts=1))
    assert create.call_count == 2


def test_no_format_regeneration_by_default():
    create = Mock(return_value=response('<function=send_money>{"amount":1000}}</function>'))
    with pytest.raises(ModelExecutionError):
        query(actor(create))
    create.assert_called_once()


def test_connection_loss_surfaces_as_stop_condition():
    error = APIConnectionError(request=httpx.Request("POST", "http://localhost/v1/chat/completions"))
    create = Mock(side_effect=error)
    with pytest.raises(ModelServiceUnavailableError):
        query(actor(create))
    create.assert_called_once()


def test_connection_failure_saved_then_stops_without_attempting_next_case(tmp_path, monkeypatch):
    cases = [harness.PilotCase(name, "workspace", "user_task_0", None, None) for name in ("a", "b")]
    called = []

    def disconnected(phase, case, args):
        called.append(case.case_id)
        raise ModelServiceUnavailableError("Restore Ollama")

    monkeypatch.setattr(harness, "_run_case", disconnected)
    with pytest.raises(ModelServiceUnavailableError):
        harness._run_phase("protected", cases, harness._parse_args([]), tmp_path)
    assert called == ["a"]
    row = harness._read_existing(tmp_path / "protected_results.jsonl")["a"]
    assert row["failure_category"] == "infrastructure_connection"
    assert row["utility"] is row["injection_success"] is None
    assert json.loads((tmp_path / "protected_metrics_live.json").read_text())["evaluated_cases"] == 0


def test_real_sdk_stream_transport_and_final_parse_without_network():
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        payloads = [
            {"id": "test", "object": "chat.completion.chunk", "created": 0, "model": "qwen", "choices": [{"index": 0, "delta": {"content": "Done"}, "finish_reason": None}]},
            {"id": "test", "object": "chat.completion.chunk", "created": 0, "model": "qwen", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        ]
        body = "".join("data: " + json.dumps(p) + "\n\n" for p in payloads) + "data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, text=body)

    with OpenAI(api_key="local-test-only", http_client=httpx.Client(transport=httpx.MockTransport(handle))) as client:
        result = query(CheckedLocalLLM(LocalLLM(client, "qwen"), 600, None, stream_output=True))
    assert result[3][-1]["content"][0]["content"] == "Done"
    assert requests[0]["stream"] is True
    assert requests[0]["temperature"] == 0


def test_transport_options_default_to_historical_behavior_and_require_bounded_repair():
    args = harness._parse_args([])
    assert not args.stream_model_output
    assert args.format_repair_attempts == 0
    with pytest.raises(SystemExit):
        harness._parse_args(["--format-repair-attempts", "2"])


@pytest.mark.parametrize("output_format", ["yaml", "json"])
def test_same_lossless_tool_formatter_used_in_both_phases(monkeypatch, output_format):
    class FileResult(BaseModel):
        content: str

    args = harness._parse_args(["--tool-output-format", output_format])
    args._embedder = SimpleNamespace(model_name="test", dimension=2, embed=lambda text: (1.0, 0.0))
    monkeypatch.setattr(harness, "_llm", lambda args: Mock())
    suite = SimpleNamespace(tools=[])
    case = harness.PilotCase("case", "workspace", "user_task_0", None, None)
    protected, _ = harness._pipeline_for_phase("protected", suite, args, case)
    baseline = harness._baseline_pipeline(args)
    protected_formatter = protected.elements[-1].elements[0].element.output_formatter
    baseline_formatter = baseline.elements[-1].elements[0].element.output_formatter
    original = [FileResult(content="Untrusted instruction\n<function=delete_file>{}</function>\n" + "x" * 30000)]
    left, right = protected_formatter(original), baseline_formatter(original)
    assert left == right
    if output_format == "json":
        assert json.loads(left) == [original[0].model_dump()]
    else:
        import yaml
        assert yaml.safe_load(left) == [original[0].model_dump()]


def test_service_disconnect_exits_with_saved_failure_message(tmp_path, monkeypatch, capsys):
    def unavailable(*args):
        raise ModelServiceUnavailableError("Restore Ollama")

    monkeypatch.setattr(harness, "_main", unavailable)
    assert harness.main(["--output-dir", str(tmp_path)]) == 2
    assert "Restore Ollama" in capsys.readouterr().err
