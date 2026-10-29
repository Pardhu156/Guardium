"""Execution diagnostics for the benchmark, independent of security decisions."""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from tqdm.auto import tqdm

from evaluation.agentdojo.tool_call_parser import PARSER_REVISION, ToolCallFormatError, parse_model_call


class CaseTimeout(BaseException):
    """Escape upstream exception handlers when the whole case exceeds its budget."""


class ModelExecutionError(RuntimeError):
    """A failed generation must not look like an evaluated, safe answer."""


class ContextWindowError(ModelExecutionError):
    """The serving configuration cannot accept an intact benchmark conversation."""

    def __init__(self, prompt_tokens: int | None, context_tokens: int | None) -> None:
        self.prompt_tokens = prompt_tokens
        self.context_tokens = context_tokens
        super().__init__(
            f"Ollama context limit exceeded: prompt={prompt_tokens}, context={context_tokens}. "
            "No benchmark content was truncated. Restart Ollama with a larger "
            "OLLAMA_CONTEXT_LENGTH and resume; this case has an unknown outcome."
        )


def _context_window_error(exc: Exception) -> ContextWindowError | None:
    """Read the structured Ollama error nested inside an OpenAI-compatible error."""
    def inspect(value: Any, depth: int = 0) -> dict[str, Any] | None:
        if depth > 6:
            return None
        if isinstance(value, str):
            try:
                return inspect(json.loads(value), depth + 1)
            except json.JSONDecodeError:
                return None
        if not isinstance(value, dict):
            return None
        if value.get("type") == "exceed_context_size_error":
            return value
        for key in ("error", "message"):
            found = inspect(value.get(key), depth + 1)
            if found is not None:
                return found
        return None

    detail = inspect(getattr(exc, "body", None))
    if detail is None:
        return None
    prompt = detail.get("n_prompt_tokens")
    context = detail.get("n_ctx")
    return ContextWindowError(
        prompt if isinstance(prompt, int) and not isinstance(prompt, bool) else None,
        context if isinstance(context, int) and not isinstance(context, bool) else None,
    )


@contextmanager
def run_lock(run_dir: Path) -> Iterator[None]:
    """Prevent two benchmark processes from replacing the same checkpoints."""
    import fcntl

    with (run_dir / ".benchmark.lock").open("a", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another benchmark process is using {run_dir}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def atomic_json(path: Path, payload: Any) -> None:
    """Replace a checkpoint only after its complete contents have been written."""
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class CaseObserver:
    """Persist current work and show heartbeats even inside a long model call."""

    def __init__(self, run_dir: Path, phase: str, case_id: str, interval: float) -> None:
        self.run_dir = run_dir
        self.phase = phase
        self.case_id = case_id
        self.interval = interval
        self.started = time.monotonic()
        self.stage_started = self.started
        self.stage = "initializing"
        self.status = "running"
        self.completed_steps = 0
        self.executed_tools = 0
        self.parser_repairs = 0
        self._stop = threading.Event()
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> CaseObserver:
        self.update("loading task")
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self.status = "interrupted" if isinstance(exc, KeyboardInterrupt) else "error" if exc else "completed"
        self.event("case_finished", error_type=type(exc).__name__ if exc else None)

    def update(self, stage: str) -> None:
        with self._lock:
            self.stage = stage
            self.stage_started = time.monotonic()
            self.event("stage")

    def event(self, kind: str, **details: Any) -> None:
        with self._lock:
            if kind == "step_completed":
                self.completed_steps += 1
            if kind == "tool_decision":
                self.executed_tools += int(bool(details["trace"].get("executed")))
            if kind == "parser_repair":
                self.parser_repairs += 1
            now = time.monotonic()
            payload = {
                "timestamp": datetime.now(UTC).isoformat(),
                "phase": self.phase,
                "case_id": self.case_id,
                "status": self.status,
                "stage": self.stage,
                "elapsed_seconds": round(now - self.started, 1),
                "stage_seconds": round(now - self.stage_started, 1),
                "completed_steps": self.completed_steps,
                "executed_tools": self.executed_tools,
                "parser_repairs": self.parser_repairs,
                "event": kind,
                **details,
            }
            with (self.run_dir / f"{self.phase}_events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, default=_json_default) + "\n")
            atomic_json(self.run_dir / f"{self.phase}_current_case.json", {k: v for k, v in payload.items() if k != "messages"})
            if kind == "tool_decision":
                trace = details["trace"]
                tqdm.write(
                    f"[{self.phase}] {self.case_id} | tool {trace.get('tool_name')} "
                    f"-> {trace.get('final_result')} | executed={bool(trace.get('executed'))} "
                    f"| total executed={self.executed_tools}"
                )
            elif kind == "step_completed":
                tqdm.write(
                    f"[{self.phase}] {self.case_id} | completed {self.stage} "
                    f"in {details['duration_seconds']:.1f}s"
                )
            elif kind == "parser_repair":
                tqdm.write(
                    f"[{self.phase}] {self.case_id} | parser repair for {details['tool_name']}: "
                    + ", ".join(details["repairs"])
                )

    def _heartbeat(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                self.event("heartbeat")
                tqdm.write(
                    f"[{self.phase}] {self.case_id} | {self.stage} | "
                    f"case={time.monotonic() - self.started:.0f}s "
                    f"stage={time.monotonic() - self.stage_started:.0f}s "
                    f"| completed steps={self.completed_steps} executed tools={self.executed_tools}"
                )
            except OSError as exc:
                tqdm.write(f"Heartbeat persistence failed: {type(exc).__name__}")
                return


class ObservedElement(BasePipelineElement):
    """Delegate an AgentDojo element while checkpointing observable messages."""

    def __init__(self, element: Any, observer: CaseObserver | None, stage: str) -> None:
        self.element = element
        self.observer = observer
        self.stage = stage
        self.calls = 0

    def query(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.observer:
            self.observer.update(f"{self.stage} #{self.calls}")
        started = time.monotonic()
        result = self.element.query(*args, **kwargs)
        if self.observer:
            self.observer.event("messages", messages=result[3])
            self.observer.event("step_completed", duration_seconds=time.monotonic() - started)
        return result


class _CheckedCompletions:
    def __init__(self, create: Callable[..., Any], max_tokens: int | None, observer: CaseObserver | None) -> None:
        self.create = create
        self.max_tokens = max_tokens
        self.failure: Exception | None = None
        self.observer = observer
        self.functions: dict[str, Any] = {}
        self.proposed_call = False

    def __call__(self, **kwargs: Any) -> Any:
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        try:
            response = self.create(**kwargs)
            choice = response.choices[0]
            content = choice.message.content
            if self.observer:
                usage = getattr(response, "usage", None)
                self.observer.event(
                    "model_response",
                    raw_content=content,
                    finish_reason=choice.finish_reason,
                    usage=usage.model_dump() if hasattr(usage, "model_dump") else None,
                    parser_revision=PARSER_REVISION,
                )
            if choice.finish_reason == "length":
                raise ModelExecutionError("Qwen output truncated at the generation/context limit")
            if isinstance(content, str):
                try:
                    parsed = parse_model_call(content, self.functions)
                except ToolCallFormatError as exc:
                    if self.observer:
                        self.observer.event("parser_failure", reason=str(exc), parser_revision=PARSER_REVISION)
                    raise ModelExecutionError(f"Invalid tool-call format: {exc}") from exc
                self.proposed_call = parsed is not None
                if parsed and parsed.repairs:
                    choice.message.content = parsed.normalized_content
                    if self.observer:
                        self.observer.event(
                            "parser_repair", tool_name=parsed.name, repairs=parsed.repairs,
                            normalized_content=parsed.normalized_content, parser_revision=PARSER_REVISION,
                        )
            return response
        except Exception as exc:
            context_error = _context_window_error(exc)
            if context_error is not None:
                self.failure = context_error
                if self.observer:
                    self.observer.event(
                        "context_window_error", prompt_tokens=context_error.prompt_tokens,
                        context_tokens=context_error.context_tokens,
                    )
                raise context_error from exc
            self.failure = exc
            if self.observer and not isinstance(exc, ModelExecutionError):
                self.observer.event("model_request_failure", error_type=type(exc).__name__)
            raise


class CheckedLocalLLM(BasePipelineElement):
    """Expose errors swallowed by AgentDojo's local model implementation."""

    def __init__(self, llm: Any, timeout: float, max_tokens: int | None, observer: CaseObserver | None = None) -> None:
        self.llm = llm
        client = llm.client.with_options(timeout=timeout, max_retries=0)
        self.completions = _CheckedCompletions(client.chat.completions.create, max_tokens, observer)
        # LocalLLM only needs this resource; the official parser and prompts remain in use.
        llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.completions)))

    def query(self, *args: Any, **kwargs: Any) -> Any:
        self.completions.failure = None
        self.completions.proposed_call = False
        runtime = kwargs.get("runtime", args[1] if len(args) > 1 else None)
        self.completions.functions = getattr(runtime, "functions", {})
        result = self.llm.query(*args, **kwargs)
        if self.completions.failure is not None:
            exc = self.completions.failure
            if isinstance(exc, ModelExecutionError):
                raise exc
            raise ModelExecutionError(f"Local model request failed ({type(exc).__name__}): {exc}") from exc
        message = result[3][-1]
        content = message.get("content") or []
        text = content if isinstance(content, str) else "".join(block.get("content", "") for block in content)
        if not text.strip() and not message.get("tool_calls"):
            raise ModelExecutionError("Local model returned an empty answer")
        if self.completions.proposed_call and not message.get("tool_calls"):
            raise ModelExecutionError("AgentDojo could not parse the generated function call")
        return result


@contextmanager
def case_deadline(seconds: float) -> Iterator[None]:
    """Bound a whole case on POSIX, including handlers that swallow Exception."""
    if seconds <= 0:
        yield
        return
    if not hasattr(signal, "setitimer") or threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Case deadlines require a POSIX main thread; use --case-timeout-seconds 0 otherwise")

    def expired(signum: int, frame: Any) -> None:
        raise CaseTimeout(f"Case exceeded {seconds:g} seconds; evaluator outcome is unknown")

    previous_handler = signal.signal(signal.SIGALRM, expired)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    started = time.monotonic()
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, max(0.001, previous_timer[0] - (time.monotonic() - started)), previous_timer[1])
