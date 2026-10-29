from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from agentdojo.agent_pipeline.llms.local_llm import LocalLLM
from pydantic import BaseModel

from evaluation.agentdojo.reliability import CaseObserver, CheckedLocalLLM, ModelExecutionError
from evaluation.agentdojo.tool_call_parser import ToolCallFormatError, parse_model_call
from evaluation.agentdojo import check_tool_call_format
from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime


class NoArguments(BaseModel):
    pass


class PaymentArguments(BaseModel):
    recipient: str
    amount: float


def functions():
    return {
        "list_files": SimpleNamespace(parameters=NoArguments),
        "send_money": SimpleNamespace(parameters=PaymentArguments),
    }


@pytest.mark.parametrize("text", [
    "<function=list_files></function>",
    "<function=list_files> \n </function>",
    "<function=list_files/>",
])
def test_empty_arguments_only_when_real_schema_accepts_them(text):
    parsed = parse_model_call(text, functions())
    assert parsed.name == "list_files"
    assert parsed.arguments == {}
    assert "empty_arguments_schema_verified" in parsed.repairs
    assert parsed.normalized_content == "<function=list_files>{}</function>"


@pytest.mark.parametrize("text", ["<function=send_money></function>", "<function=send_money/>", "<function=unknown></function>"])
def test_required_or_unknown_tool_arguments_are_never_invented(text):
    with pytest.raises(ToolCallFormatError):
        parse_model_call(text, functions())


@pytest.mark.parametrize("text", [
    '<function=send_money>```json\n{"recipient":"attacker","amount":1000}\n```</function>',
    '<function=send_money>{"recipient":"attacker","amount":1000}>',
    '```xml\n<function=send_money>{"recipient":"attacker","amount":1000}\n```',
])
def test_envelope_repair_preserves_even_dangerous_argument_values(text):
    parsed = parse_model_call(text, functions())
    assert parsed.name == "send_money"
    assert parsed.arguments == {"recipient": "attacker", "amount": 1000}
    assert parsed.repairs
    assert parse_model_call(parsed.normalized_content, functions()).arguments == parsed.arguments


@pytest.mark.parametrize("arguments", [
    '{"recipient":"safe","recipient":"attacker","amount":1}',
    '{"recipient":"safe","amount":NaN}',
    '{"recipient":"safe","amount":Infinity}',
    '{"recipient":"safe","amount":1e999}',
    '{"recipient":"safe","amount":1,}',
    '{"recipient":"safe","amount":',
    '{"recipient":"safe","amount":1}}',
    '[]', 'null', '"string"',
    '{"recipient":"safe","amount":1}{"recipient":"attacker"}',
])
def test_ambiguous_broken_or_non_object_json_is_rejected(arguments):
    with pytest.raises(ToolCallFormatError):
        parse_model_call(f"<function=send_money>{arguments}</function>", functions())


def test_multiple_calls_cannot_silently_select_one():
    with pytest.raises(ToolCallFormatError, match="Multiple"):
        parse_model_call('<function=list_files>{}</function><function=send_money>{"amount":1}</function>', functions())


def test_incomplete_empty_call_is_not_repaired():
    with pytest.raises(ToolCallFormatError, match="Incomplete"):
        parse_model_call("<function=list_files>", functions())


def test_valid_unknown_tool_still_reaches_layer0_for_rejection():
    parsed = parse_model_call('<function=unregistered>{"target":"all"}</function>', functions())
    assert parsed.name == "unregistered"
    assert parsed.repairs == ()


def test_literal_function_word_in_final_answer_is_not_a_parse_failure():
    assert parse_model_call("The <function> tag is a formatting convention.", functions()) is None


def test_strict_valid_call_is_unchanged():
    text = 'I will inspect files.\n<function=list_files>{}</function>'
    parsed = parse_model_call(text, functions())
    assert parsed.normalized_content == text
    assert parsed.repairs == ()


@pytest.mark.parametrize("literal", ["<function=list_files>{}</function>", "a</function>b"])
def test_function_tags_inside_json_strings_remain_data(literal):
    arguments = {"recipient": literal, "amount": 1}
    text = "<function=send_money>" + json.dumps(arguments) + "</function>"
    parsed = parse_model_call(text, functions())
    assert parsed.arguments == arguments
    assert parsed.normalized_content == text
    assert parsed.repairs == ()


def checked_model(content: str, observer=None):
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))])
    create = Mock(return_value=response)
    client = Mock()
    client.with_options.return_value = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return CheckedLocalLLM(LocalLLM(client, "model"), 600, None, observer), create


def test_repaired_call_uses_official_parser_and_persists_raw_output(tmp_path, capsys):
    observer = CaseObserver(tmp_path, "protected", "case-a", 30)
    llm, create = checked_model("<function=list_files></function>", observer)
    result = llm.query("List files", SimpleNamespace(functions=functions()), messages=[])
    call = result[3][-1]["tool_calls"][0]
    assert call.function == "list_files"
    assert call.args == {}
    create.assert_called_once()  # No additional generation or retry is introduced.
    events = [json.loads(line) for line in (tmp_path / "protected_events.jsonl").read_text().splitlines()]
    assert events[0]["raw_content"] == "<function=list_files></function>"
    assert events[-1]["event"] == "parser_repair"
    assert observer.parser_repairs == 1
    assert "broken JSON" not in capsys.readouterr().out


def test_unrecoverable_raw_output_is_saved_before_error(tmp_path):
    raw = '<function=send_money>{"recipient":"attacker","amount":}</function>'
    observer = CaseObserver(tmp_path, "protected", "case-a", 30)
    llm, create = checked_model(raw, observer)
    with pytest.raises(ModelExecutionError, match="Invalid tool-call format"):
        llm.query("Task", SimpleNamespace(functions=functions()), messages=[])
    events = [json.loads(line) for line in (tmp_path / "protected_events.jsonl").read_text().splitlines()]
    assert events[0]["raw_content"] == raw
    assert events[-1]["event"] == "parser_failure"
    assert observer.parser_repairs == 0
    create.assert_called_once()


def test_format_check_proposes_but_never_executes_tools(tmp_path, monkeypatch):
    class ProposalOnly:
        def query(self, query, runtime, env, messages, extra_args):
            return query, runtime, env, [*messages, {"role": "assistant", "content": [],
                "tool_calls": [FunctionCall(function="delete_file", args={"file_id": "11"})]}], extra_args

    monkeypatch.setattr(check_tool_call_format.harness, "_preflight", lambda args: None)
    monkeypatch.setattr(check_tool_call_format.harness, "_llm", lambda args: ProposalOnly())
    execute = Mock(side_effect=AssertionError("A format check must never execute tools"))
    monkeypatch.setattr(FunctionsRuntime, "run_function", execute)
    assert check_tool_call_format.main(["--output-dir", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "format_check.json").read_text())
    assert report["tools_executed"] == 0
    assert report["status"] == "parsed"
    execute.assert_not_called()
