"""Real AgentDojo pipeline element protected by AegisVault.

This module imports AgentDojo lazily at runtime so the base AegisVault package
does not require AgentDojo unless this integration is used.
"""

from __future__ import annotations

from ast import literal_eval
from typing import Any, Callable

from pydantic import BaseModel

from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement

from aegisvault.integrations.agentdojo.adapter import AgentDojoAdapterConfig
from aegisvault.layer0 import Layer0Validator
from aegisvault.policy.models import DomainPolicy
from aegisvault.runtime.action_gate import (
    ActionGate,
    ActionGateConfig,
    ActionRuntimeContext,
    ActionVerdict,
    ProposedToolAction,
    SideEffectLevel,
    ToolMetadata,
)
from aegisvault.runtime.action_gate.evaluators import ActionEvaluator
from aegisvault.runtime.goal_vault import GoalEmbedder, GoalVault, InMemoryGoalVaultBackend
from aegisvault.sentinel import SentinelConfig, SentinelExecutionState, SentinelMonitor, ToolCallState


def build_aegisvault_agentdojo_pipeline(*args: Any, **kwargs: Any) -> Any:
    """Build an AgentDojo pipeline whose tool executor is protected by AegisVault.

    Keyword arguments match :class:`AegisVaultAgentDojoToolsExecutor`, plus:
    - ``llm``: an AgentDojo ``BasePipelineElement`` LLM component.
    - ``system_message``: optional AgentDojo system prompt.
    - ``max_iters``: tool loop iterations.
    """

    from agentdojo.agent_pipeline.agent_pipeline import AgentPipeline, load_system_message
    from agentdojo.agent_pipeline.basic_elements import InitQuery, SystemMessage
    from agentdojo.agent_pipeline.tool_execution import ToolsExecutionLoop

    llm = kwargs.pop("llm")
    system_message = kwargs.pop("system_message", None) or load_system_message(None)
    max_iters = kwargs.pop("max_iters", 15)
    policy = kwargs["policy"]
    config = kwargs["config"]
    goal_vault = kwargs.get("goal_vault")
    embedder = kwargs.get("embedder")
    if goal_vault is None:
        goal_vault = GoalVault(backend=InMemoryGoalVaultBackend(), embedder=embedder)
        kwargs["goal_vault"] = goal_vault
    initializer = AegisVaultAgentDojoRequestInitializer(policy=policy, config=config, goal_vault=goal_vault)
    executor = AegisVaultAgentDojoToolsExecutor(*args, **kwargs)
    pipeline = AgentPipeline(
        [
            SystemMessage(system_message),
            InitQuery(),
            initializer,
            llm,
            ToolsExecutionLoop([executor, llm], max_iters=max_iters),
        ]
    )
    pipeline.name = f"{getattr(llm, 'name', 'agentdojo')}-aegisvault"
    return pipeline


class AegisVaultAgentDojoRequestInitializer(BasePipelineElement):
    """Layer 0 request sanity validation and Goal Vault initialization before LLM execution."""

    def __init__(self, *, policy: DomainPolicy, config: AgentDojoAdapterConfig, goal_vault: GoalVault) -> None:
        self.policy = policy
        self.config = config
        self.goal_vault = goal_vault
        self.layer0 = Layer0Validator(policy=policy)
        self._initialized_sessions: set[str] = set()

    def query(
        self,
        query: str,
        runtime: Any,
        env: Any = None,
        messages: Any = (),
        extra_args: dict[str, Any] | None = None,
    ) -> tuple[str, Any, Any, Any, dict[str, Any]]:
        extra_args = extra_args or {}
        session_id = _session_id(extra_args, query)
        decision = self.layer0.validate_request(
            session_id=session_id,
            request_text=query,
            domain=self.config.domain,
            metadata={"suite": self.config.suite_name},
        )
        extra_args["aegisvault_session_id"] = session_id
        extra_args["aegisvault_request_allowed"] = decision.allowed
        if not decision.allowed:
            raise RuntimeError(f"Layer 0 request block: {decision.reason}")
        if session_id not in self._initialized_sessions:
            self.goal_vault.commit_goal(
                session_id=session_id,
                application_name=self.policy.application.name,
                goal=query,
                metadata={"suite": self.config.suite_name},
            )
            self._initialized_sessions.add(session_id)
        return query, runtime, env, messages, extra_args


class AegisVaultAgentDojoToolsExecutor(BasePipelineElement):
    """AgentDojo ``ToolsExecutor`` replacement that protects every tool call."""

    def __init__(
        self,
        *,
        policy: DomainPolicy,
        config: AgentDojoAdapterConfig,
        goal_vault: GoalVault | None = None,
        embedder: GoalEmbedder | None = None,
        action_evaluator: ActionEvaluator | None = None,
        sentinel_monitor: SentinelMonitor | None = None,
        action_config: ActionGateConfig | None = None,
        tool_metadata_resolver: Callable[[Any, str], ToolMetadata] | None = None,
        decision_sink: Callable[[dict[str, Any]], None] | None = None,
        tool_output_formatter: Callable[[Any], str] | None = None,
    ) -> None:
        from agentdojo.agent_pipeline.tool_execution import tool_result_to_str

        self.policy = policy
        self.config = config
        self.output_formatter = tool_output_formatter or tool_result_to_str
        self.goal_vault = goal_vault or GoalVault(backend=InMemoryGoalVaultBackend(), embedder=embedder)
        self.embedder = embedder or self.goal_vault.embedder
        self.layer0 = Layer0Validator(policy=policy)
        self.sentinel_monitor = sentinel_monitor
        self.tool_metadata_resolver = tool_metadata_resolver
        self.decision_sink = decision_sink
        self.action_gate = ActionGate(
            goal_vault=self.goal_vault,
            embedder=self.embedder,
            evaluator=action_evaluator,
            config=action_config,
        )
        self._initialized_sessions: set[str] = set()
        self._previous_actions: dict[str, str] = {}

    def query(
        self,
        query: str,
        runtime: Any,
        env: Any = None,
        messages: Any = (),
        extra_args: dict[str, Any] | None = None,
    ) -> tuple[str, Any, Any, Any, dict[str, Any]]:
        from agentdojo.agent_pipeline.llms.google_llm import EMPTY_FUNCTION_NAME
        from agentdojo.types import ChatToolResultMessage, text_content_block_from_string

        extra_args = extra_args or {}
        if not self._has_tool_calls(messages):
            return query, runtime, env, messages, extra_args

        session_id = _session_id(extra_args, query)
        if not extra_args.get("aegisvault_request_allowed"):
            request_decision = self.layer0.validate_request(
                session_id=session_id,
                request_text=query,
                domain=self.config.domain,
                metadata={"suite": self.config.suite_name},
            )
            if not request_decision.allowed:
                return query, runtime, env, [*messages, _blocked_message("layer0_request", request_decision.reason)], extra_args
            self._ensure_goal(session_id=session_id, objective=query)
        tool_messages = []
        for index, tool_call in enumerate(messages[-1]["tool_calls"], start=1):
            if tool_call.function == EMPTY_FUNCTION_NAME:
                tool_messages.append(_tool_message(tool_call, "", "Empty function name provided. Provide a valid function name."))
                continue
            result, error = self._execute_agentdojo_tool(
                session_id=session_id,
                runtime=runtime,
                env=env,
                tool_call=tool_call,
                query=query,
                messages=messages,
                step_index=index,
            )
            formatted = "" if error else self.output_formatter(result)
            tool_messages.append(_tool_message(tool_call, formatted, error))
        return query, runtime, env, [*messages, *tool_messages], extra_args

    def _execute_agentdojo_tool(
        self,
        *,
        session_id: str,
        runtime: Any,
        env: Any,
        tool_call: Any,
        query: str,
        messages: Any,
        step_index: int,
    ) -> tuple[Any, str | None]:
        if tool_call.function not in runtime.functions:
            decision = self.layer0.validate_tool_call(
                session_id=session_id,
                tool_name=tool_call.function,
                arguments=dict(tool_call.args),
                domain=self.config.domain,
                tool_catalog=_tool_catalog(runtime),
            )
            return "", decision.reason

        args = _coerce_args(dict(tool_call.args))
        layer0_decision = self.layer0.validate_tool_call(
            session_id=session_id,
            tool_name=tool_call.function,
            arguments=args,
            domain=self.config.domain,
            tool_catalog=_tool_catalog(runtime),
        )
        if not layer0_decision.allowed:
            return "", layer0_decision.reason

        tool_metadata = self._resolve_tool_metadata(runtime.functions[tool_call.function], tool_call.function)
        sentinel_decision = None
        if self.policy.sentinel.enabled and self.policy.sentinel.runtime.evaluate_before_every_tool:
            monitor = self.sentinel_monitor or SentinelMonitor(embedder=self.embedder, config=_sentinel_config(self.policy))
            anchor = self.goal_vault.get_anchor(session_id)
            sentinel_decision = monitor.analyze(
                session_id=session_id,
                trusted_goal=anchor.original_goal,
                execution=SentinelExecutionState(
                    session_id=session_id,
                    reasoning=_observable_context_from_messages(messages),
                    current_intent=query,
                    tool_call=ToolCallState(name=tool_call.function, arguments=args),
                    step_index=step_index,
                    metadata={
                        "trusted_goal": anchor.original_goal,
                        "previous_action": self._previous_actions.get(session_id),
                        "tool_outputs": _tool_outputs_from_messages(messages),
                    },
                ),
            )
            if (
                sentinel_decision.decision.value == "block"
                and self.policy.sentinel.enforcement.block_on_sentinel_block
            ):
                self._record_trace(
                    session_id=session_id,
                    query=query,
                    tool_call=tool_call,
                    args=args,
                    tool_metadata=tool_metadata,
                    sentinel_decision=sentinel_decision,
                    action_decision=None,
                    final_result="BLOCK",
                    reason=sentinel_decision.reason,
                    executed=False,
                )
                return "", sentinel_decision.reason

        action = ProposedToolAction(
            tool_name=tool_call.function,
            tool_description=getattr(runtime.functions[tool_call.function], "description", tool_call.function),
            tool_arguments=args,
        )
        decision = self.action_gate.evaluate_action(
            session_id=session_id,
            action=action,
            tool_metadata=tool_metadata,
            policy=self.policy,
            runtime_context=ActionRuntimeContext(
                reasoning_summary=_observable_context_from_messages(messages),
                previous_approved_action=self._previous_actions.get(session_id),
                current_intent=query,
                step_index=step_index,
                sentinel_decision=sentinel_decision,
                session_metadata={"suite": self.config.suite_name},
            ),
        )
        if decision.verdict != ActionVerdict.EXECUTE:
            self._record_trace(
                session_id=session_id,
                query=query,
                tool_call=tool_call,
                args=args,
                tool_metadata=tool_metadata,
                sentinel_decision=sentinel_decision,
                action_decision=decision,
                final_result=decision.verdict.value,
                reason=decision.reason,
                executed=False,
            )
            return "", decision.reason
        result = runtime.run_function(env, tool_call.function, args)
        self._previous_actions[session_id] = tool_call.function
        self._record_trace(
            session_id=session_id,
            query=query,
            tool_call=tool_call,
            args=args,
            tool_metadata=tool_metadata,
            sentinel_decision=sentinel_decision,
            action_decision=decision,
            final_result="EXECUTE",
            reason=decision.reason,
            executed=True,
        )
        return result

    def _ensure_goal(self, *, session_id: str, objective: str) -> None:
        if session_id in self._initialized_sessions:
            return
        self.goal_vault.commit_goal(
            session_id=session_id,
            application_name=self.policy.application.name,
            goal=objective,
            metadata={"suite": self.config.suite_name},
        )
        self._initialized_sessions.add(session_id)

    def _has_tool_calls(self, messages: Any) -> bool:
        return (
            bool(messages)
            and messages[-1]["role"] == "assistant"
            and messages[-1]["tool_calls"] is not None
            and len(messages[-1]["tool_calls"]) > 0
        )

    def _resolve_tool_metadata(self, function: Any, tool_name: str) -> ToolMetadata:
        if self.tool_metadata_resolver is not None:
            return self.tool_metadata_resolver(function, tool_name)
        return _tool_metadata(function)

    def _record_trace(
        self,
        *,
        session_id: str,
        query: str,
        tool_call: Any,
        args: dict[str, Any],
        tool_metadata: ToolMetadata,
        sentinel_decision: Any,
        action_decision: Any,
        final_result: str,
        reason: str,
        executed: bool,
    ) -> None:
        if self.decision_sink is None:
            return
        self.decision_sink(
            {
                "session_id": session_id,
                "suite": self.config.suite_name,
                "current_intent": query,
                "tool_name": tool_call.function,
                "tool_arguments": _sanitize_args(args),
                "risk_classification": {
                    "risk_level": tool_metadata.risk_level,
                    "side_effect_level": tool_metadata.side_effect_level.value,
                    "requires_approval": tool_metadata.requires_approval,
                    "required_permissions": list(tool_metadata.required_permissions),
                },
                "sentinel": _sentinel_trace(sentinel_decision),
                "action_gate": _action_trace(action_decision),
                "final_result": final_result,
                "reason": reason,
                "executed": executed,
            }
        )


def _tool_message(tool_call: Any, content: str, error: str | None) -> Any:
    from agentdojo.types import ChatToolResultMessage, text_content_block_from_string

    return ChatToolResultMessage(
        role="tool",
        content=[text_content_block_from_string(content)],
        tool_call_id=tool_call.id,
        tool_call=tool_call,
        error=error,
    )


def _blocked_message(kind: str, reason: str) -> Any:
    from agentdojo.functions_runtime import FunctionCall

    return _tool_message(FunctionCall(function=kind, args={}, id=kind), "", reason)


def _session_id(extra_args: dict[str, Any], query: str) -> str:
    for key in ("session_id", "task_id", "user_task_id"):
        value = extra_args.get(key)
        if value:
            return str(value)
    return "agentdojo-" + str(abs(hash(query)))


def _coerce_args(args: dict[str, Any]) -> dict[str, Any]:
    output = dict(args)
    for key, value in output.items():
        if isinstance(value, str) and _is_string_list(value):
            output[key] = literal_eval(value)
    return output


def _is_string_list(value: str) -> bool:
    try:
        return isinstance(literal_eval(value), list)
    except (ValueError, SyntaxError):
        return False


def _tool_catalog(runtime: Any) -> dict[str, Any]:
    return {name: {"description": function.description, "parameters": _schema(function.parameters)} for name, function in runtime.functions.items()}


def _schema(model: type[BaseModel]) -> dict[str, Any]:
    if hasattr(model, "model_json_schema"):
        return model.model_json_schema()
    return {}


def _tool_metadata(function: Any) -> ToolMetadata:
    return ToolMetadata(
        risk_level="medium",
        side_effect_level=SideEffectLevel.WRITE if function.name.startswith(("send_", "delete_", "book_", "transfer_")) else SideEffectLevel.READ,
    )


def _observable_context_from_messages(messages: Any) -> str | None:
    parts: list[str] = []
    for message in list(messages or [])[-8:]:
        role = _message_get(message, "role")
        content = _message_get(message, "content")
        tool_calls = _message_get(message, "tool_calls")
        if content:
            parts.append(f"{role}: {_compact_content(content)}")
        if tool_calls:
            names = [getattr(call, "function", "") for call in tool_calls]
            parts.append(f"{role}: proposed_tools={','.join(name for name in names if name)}")
        error = _message_get(message, "error")
        if error:
            parts.append(f"{role}: tool_error={error}")
    return "\n".join(parts) if parts else None


def _tool_outputs_from_messages(messages: Any) -> list[str]:
    outputs: list[str] = []
    for message in list(messages or [])[-8:]:
        if _message_get(message, "role") == "tool":
            content = _message_get(message, "content")
            if content:
                outputs.append(_compact_content(content))
    return outputs


def _message_get(message: Any, key: str) -> Any:
    if isinstance(message, dict):
        return message.get(key)
    try:
        return message[key]
    except Exception:
        return getattr(message, key, None)


def _compact_content(content: Any) -> str:
    if isinstance(content, str):
        return content[:2000]
    if isinstance(content, list):
        values = []
        for item in content:
            if isinstance(item, dict):
                values.append(str(item.get("content") or item.get("text") or item)[:500])
            else:
                values.append(str(item)[:500])
        return " ".join(values)[:2000]
    return str(content)[:2000]


def _sanitize_args(args: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in args.items():
        lowered = str(key).lower()
        if any(secret in lowered for secret in ("password", "token", "secret", "key")):
            sanitized[str(key)] = "[REDACTED]"
        else:
            sanitized[str(key)] = _truncate(value)
    return sanitized


def _truncate(value: Any) -> Any:
    if isinstance(value, str):
        return value[:1000]
    if isinstance(value, dict):
        return {str(key): _truncate(raw_value) for key, raw_value in value.items()}
    if isinstance(value, list | tuple):
        return [_truncate(item) for item in value[:20]]
    return value


def _sentinel_trace(decision: Any) -> dict[str, Any] | None:
    if decision is None:
        return None
    return {
        "verdict": decision.decision.value,
        "confidence": decision.confidence,
        "reasoning_similarity": decision.reasoning_similarity,
        "intent_similarity": decision.intent_similarity,
        "action_similarity": decision.action_similarity,
        "reasoning_drift": decision.reasoning_drift,
        "intent_drift": decision.intent_drift,
        "action_drift": decision.action_drift,
        "fused_risk": decision.fused_risk,
        "ema_risk": decision.ema_risk,
        "reason": decision.reason,
    }


def _action_trace(decision: Any) -> dict[str, Any] | None:
    if decision is None:
        return None
    return {
        "verdict": decision.verdict.value,
        "confidence": decision.confidence,
        "goal_similarity": decision.goal_similarity,
        "decision_source": decision.decision_source.value,
        "ollama_called": decision.ollama_called,
        "reason": decision.reason,
    }


def _sentinel_config(policy: DomainPolicy) -> SentinelConfig:
    return SentinelConfig(
        reasoning_weight=policy.sentinel.reasoning_weight,
        intent_weight=policy.sentinel.intent_weight,
        action_weight=policy.sentinel.action_weight,
        ema_alpha=policy.sentinel.ema_alpha,
        allow_threshold=policy.sentinel.allow_threshold,
        observe_threshold=policy.sentinel.observe_threshold,
        review_threshold=policy.sentinel.review_threshold,
    )
