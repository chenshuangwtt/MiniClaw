"""Shared async-first execution engine for portable and native agent modes."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from miniclaw.agent.context import ContextManager
from miniclaw.agent.executor import Observation, ToolExecutor
from miniclaw.agent.parser import OutputParser, ParseError
from miniclaw.agent.prompts import build_full_prompt, build_memory_prompt
from miniclaw.agent.recovery import RecoveryManager
from miniclaw.agent.state import FinalAnswer, RunStatus, ToolCall
from miniclaw.agent.trace import StepTrace, TraceLogger
from miniclaw.llm.base import BaseLLM, LLMResponse
from miniclaw.memory.base import MemoryBackend, NullMemoryBackend
from miniclaw.memory.extractor import MemoryExtractor
from miniclaw.memory.manager import MemoryManager
from miniclaw.redaction import redact_secrets
from miniclaw.tools.audit import AuditLogger
from miniclaw.tools.permissions import PermissionPolicy
from miniclaw.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from miniclaw.storage.sqlite_store import SQLiteStore

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    """Stable outcome of one agent task."""

    success: bool
    answer: str | None
    error: str | None
    trace: TraceLogger
    steps_taken: int
    status: RunStatus = RunStatus.CREATED
    run_id: str = ""

    def __repr__(self) -> str:
        if self.success:
            return f"<AgentResult success answer={self.answer!r} steps={self.steps_taken}>"
        return f"<AgentResult failed error={self.error!r} steps={self.steps_taken}>"


class AgentEngine:
    """The single execution core behind synchronous and asynchronous APIs."""

    def __init__(
        self,
        llm: BaseLLM,
        registry: ToolRegistry,
        max_steps: int = 10,
        max_errors: int = 3,
        max_tool_calls: int | None = None,
        max_repeated_tool_calls: int = 3,
        max_run_seconds: float | None = None,
        context: ContextManager | None = None,
        recovery: RecoveryManager | None = None,
        memory_backend: MemoryBackend | None = None,
        memory_extractor: MemoryExtractor | None = None,
        memory_manager: MemoryManager | None = None,
        permission_policy: PermissionPolicy | None = None,
        audit_logger: AuditLogger | None = None,
        store: SQLiteStore | None = None,
        session_id: int | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.audit_logger = audit_logger
        self.permission_policy = permission_policy or PermissionPolicy()
        self.executor = ToolExecutor(registry, self.permission_policy, audit_logger)
        self.parser = OutputParser()
        self.parse_mode = "portable"
        self.max_steps = max_steps
        self.max_errors = max_errors
        self.max_tool_calls = max_tool_calls if max_tool_calls is not None else max_steps
        self.max_repeated_tool_calls = max_repeated_tool_calls
        self.max_run_seconds = max_run_seconds
        self.context = context if context is not None else ContextManager()
        self.recovery = recovery if recovery is not None else RecoveryManager(max_errors=max_errors)
        self.memory_backend = memory_backend if memory_backend is not None else NullMemoryBackend()
        self.memory_extractor = (
            memory_extractor if memory_extractor is not None else MemoryExtractor()
        )
        self.memory_manager = (
            memory_manager
            if memory_manager is not None
            else MemoryManager(self.memory_backend, self.memory_extractor)
        )
        self.status = RunStatus.CREATED
        self.run_id = ""
        self._cancel_requested = False
        self.store = store
        self.session_id = session_id
        self._task = ""
        self._started_at = 0.0
        self._tool_call_counts: dict[str, int] = {}
        self._tool_calls_taken = 0
        self._pending_action: dict[str, Any] | None = None

    async def run_async(self, user_task: str, user_id: str = "default") -> AgentResult:
        """Execute a task without blocking the caller's event loop."""
        self.run_id = uuid.uuid4().hex
        self._task = user_task
        self._started_at = time.monotonic()
        self._tool_call_counts = {}
        self._tool_calls_taken = 0
        self._pending_action = None
        # Context belongs to the engine/session, so every turn is recorded once
        # before either transport builds its request.
        self.context.add_message("user", user_task, user_turn=True)
        if self.store is not None:
            self.store.create_run(self.run_id, user_task, self.session_id)
        self._set_status(RunStatus.RUNNING)
        self._checkpoint(0, TraceLogger(), None)
        memories = await self._search_memories(user_task, user_id)
        if self.parse_mode == "native":
            result = await self._run_native(user_task, memories)
        elif self.parse_mode == "auto":
            result = await self._run_native(user_task, memories, allow_portable_fallback=True)
        else:
            result = await self._run_portable(user_task, memories)
        if result.success and result.answer:
            await self._maybe_save_memory(user_task, user_id)
        return result

    async def resume_async(
        self, run_id: str, user_id: str = "default", approve_pending: bool = False
    ) -> AgentResult:
        """Continue a durable portable/native run from its latest checkpoint.

        Pending side-effecting actions are never replayed implicitly. They are
        only dispatched after this caller explicitly passes ``approve_pending``.
        """
        if self.store is None:
            raise RuntimeError(
                "恢复任务需要 SQLiteStore（A SQLiteStore is required to resume a run.）"
            )
        run = self.store.get_run(run_id)
        checkpoint = self.store.get_latest_checkpoint(run_id)
        if run is None or checkpoint is None:
            raise ValueError(
                f"任务 '{run_id}' 没有可恢复的 Checkpoint（No resumable checkpoint exists）。"
            )
        if run["status"] in {
            RunStatus.COMPLETED.value,
            RunStatus.FAILED.value,
        }:
            raise ValueError(
                f"任务 '{run_id}' 已处于终态，不能恢复（terminal and cannot be resumed）。"
            )

        state = checkpoint["state"]
        pending = state.get("pending_action")
        trace = TraceLogger()
        trace.steps = [StepTrace.model_validate(step) for step in state.get("trace", [])]
        self.context.restore(state.get("context", []))
        self.run_id = run_id
        self._task = str(state.get("task", run["task"]))
        self.parse_mode = str(state.get("mode", "portable"))
        self._started_at = time.monotonic()
        self._tool_calls_taken = sum(1 for step in trace.steps if step.tool_name)
        self._tool_call_counts = {}
        self._pending_action = None
        self._set_status(RunStatus.RUNNING)

        if pending:
            if self.parse_mode == "native":
                self._pending_action = pending
                return self._result(
                    False,
                    None,
                    "Native 待处理工具调用需要人工核对，不会自动重放（manual reconciliation required）。",
                    trace,
                    len(trace.steps),
                    RunStatus.WAITING_APPROVAL,
                )
            if pending.get("state") == "in_flight":
                error = "待处理工具可能已执行但结果未保存，必须人工核对（manual reconciliation required）。"
                return self._result(False, None, error, trace, len(trace.steps), RunStatus.FAILED)
            if not approve_pending:
                self._pending_action = pending
                return self._result(
                    False,
                    None,
                    "任务正在等待对待处理工具调用的显式审批（waiting for explicit approval）。",
                    trace,
                    len(trace.steps),
                    RunStatus.WAITING_APPROVAL,
                )
            name = str(pending["tool_name"])
            arguments = dict(pending["arguments"])
            was_approval_required = name in self.permission_policy.approval_required_tools
            if was_approval_required:
                self.permission_policy.approval_required_tools.remove(name)
            try:
                observation = await self._execute_budgeted(name, arguments)
            finally:
                if was_approval_required:
                    self.permission_policy.approval_required_tools.add(name)
            step = len(trace.steps) + 1
            trace.log_step(
                step=step,
                parsed_action="approved_tool_call",
                tool_name=name,
                arguments=arguments,
                observation=observation.output if observation.success else None,
                error=observation.error,
            )
            self._append_portable_observation(
                ToolCall(tool_name=name, arguments=arguments), observation
            )
            self._checkpoint(step, trace, observation)
            if not observation.success:
                return self._result(False, None, observation.error, trace, step)

        memories = await self._search_memories(self._task, user_id)
        start_step = len(trace.steps) + 1
        if self.parse_mode == "native":
            result = await self._run_native(
                self._task, memories, trace=trace, start_step=start_step
            )
        else:
            result = await self._run_portable(
                self._task, memories, trace=trace, start_step=start_step
            )
        if result.success and result.answer:
            await self._maybe_save_memory(self._task, user_id)
        return result

    def cancel(self) -> None:
        """Request cooperative cancellation before the next model/tool step."""
        self._cancel_requested = True

    def reset_cancel(self) -> None:
        """Clear a previous cancellation request before intentionally reusing an engine."""
        self._cancel_requested = False

    async def _run_portable(
        self,
        task: str,
        memories: list[str],
        trace: TraceLogger | None = None,
        start_step: int = 1,
    ) -> AgentResult:
        trace = trace or TraceLogger()
        errors = 0
        last_tool_failed = False
        schemas = self._portable_schemas()
        for step in range(start_step, self.max_steps + 1):
            if self._run_timed_out():
                return self._result(
                    False,
                    None,
                    "任务超过时间预算（Run exceeded its time budget.）",
                    trace,
                    step - 1,
                )
            cancelled = self._cancel_result(trace, step - 1)
            if cancelled:
                return cancelled
            if self.context.should_compress():
                self.context.compress()
            prompt = build_full_prompt(
                task, tools=schemas, history=self._portable_history(), memories=memories
            )
            try:
                raw = await asyncio.to_thread(self.llm.generate, prompt)
                parsed = self.parser.parse(raw)
            except ParseError as exc:
                errors += 1
                trace.log_step(
                    step=step,
                    model_output=getattr(exc, "raw", ""),
                    parsed_action="parse_error",
                    error=str(exc),
                )
                self._inject_recovery(str(exc))
                if errors >= self.max_errors:
                    return self._result(False, None, f"Aborted after {errors} errors.", trace, step)
                continue
            except Exception as exc:
                errors += 1
                trace.log_step(step=step, parsed_action="llm_error", error=f"LLM error: {exc}")
                if errors >= self.max_errors:
                    return self._result(
                        False, None, f"Aborted after {errors} errors: {exc}", trace, step
                    )
                continue

            if isinstance(parsed, FinalAnswer):
                trace.log_step(
                    step=step,
                    model_output=raw,
                    parsed_action="final_answer",
                    observation=parsed.answer,
                )
                self.context.add_message("assistant", parsed.answer)
                if last_tool_failed:
                    return self._result(
                        False,
                        None,
                        "任务因未解决的工具失败而结束（Task ended after an unresolved tool failure.）",
                        trace,
                        step,
                    )
                return self._result(True, parsed.answer, None, trace, step)

            self._pending_action = {
                "state": "in_flight",
                "tool_name": parsed.tool_name,
                "arguments": parsed.arguments,
            }
            self._checkpoint(step, trace, None)
            observation = await self._execute_budgeted(parsed.tool_name, parsed.arguments)
            self._pending_action = None
            trace.log_step(
                step=step,
                model_output=raw,
                parsed_action="tool_call",
                tool_name=parsed.tool_name,
                arguments=parsed.arguments,
                observation=observation.output if observation.success else None,
                error=observation.error,
            )
            self._append_portable_observation(parsed, observation)
            self._checkpoint(step, trace, observation)
            if (
                observation.error_kind == "permission"
                and observation.error
                and "approval" in observation.error
            ):
                self._pending_action = {
                    "state": "awaiting_approval",
                    "tool_name": parsed.tool_name,
                    "arguments": parsed.arguments,
                }
                return self._result(
                    False, None, observation.error, trace, step, RunStatus.WAITING_APPROVAL
                )
            if observation.success:
                errors = 0
                last_tool_failed = False
            else:
                errors += 1
                last_tool_failed = True
                if errors >= self.max_errors:
                    return self._result(False, None, observation.error, trace, step)

        return self._result(
            False,
            None,
            f"超过最大执行步数（Exceeded maximum steps: {self.max_steps}）。",
            trace,
            self.max_steps,
        )

    async def _run_native(
        self,
        task: str,
        memories: list[str],
        allow_portable_fallback: bool = False,
        trace: TraceLogger | None = None,
        start_step: int = 1,
    ) -> AgentResult:
        trace = trace or TraceLogger()
        tools = self.registry.to_openai_tools()
        errors = 0
        trace_step = start_step - 1
        last_tool_failed = False
        for round_number in range(1, self.max_steps + 1):
            if self._run_timed_out():
                return self._result(
                    False,
                    None,
                    "任务超过时间预算（Run exceeded its time budget.）",
                    trace,
                    round_number - 1,
                )
            cancelled = self._cancel_result(trace, round_number - 1)
            if cancelled:
                return cancelled
            messages = self._prepare_native_request(task, memories, tools)
            if messages is None:
                trace_step += 1
                error = "Native 请求在压缩后仍超过上下文预算（context budget exceeded）。"
                trace.log_step(step=trace_step, parsed_action="context_error", error=error)
                return self._result(False, None, error, trace, round_number)
            try:
                response = await asyncio.to_thread(self.llm.chat, messages, tools)
            except Exception as exc:
                errors += 1
                trace_step += 1
                trace.log_step(
                    step=trace_step, parsed_action="llm_error", error=f"LLM error: {exc}"
                )
                if errors >= self.max_errors:
                    return self._result(
                        False, None, f"Aborted after {errors} errors: {exc}", trace, round_number
                    )
                continue

            if not response.has_tool_calls:
                if allow_portable_fallback and response.content:
                    try:
                        self.parser.parse(response.content)
                    except ParseError:
                        pass
                    else:
                        self.parse_mode = "portable"
                        return await self._run_portable(task, memories)
                trace_step += 1
                trace.log_step(
                    step=trace_step,
                    model_output=response.content,
                    parsed_action="final_answer",
                    observation=response.content,
                )
                self.context.add_message("assistant", response.content)
                if last_tool_failed:
                    return self._result(
                        False,
                        None,
                        "任务因未解决的工具失败而结束（Task ended after an unresolved tool failure.）",
                        trace,
                        round_number,
                    )
                return self._result(True, response.content, None, trace, round_number)

            self.context.add_message(
                "assistant", response.content, tool_calls=_tool_call_payloads(response)
            )
            for call in response.tool_calls:
                trace_step += 1
                self._pending_action = {
                    "state": "in_flight",
                    "tool_name": call.name,
                    "arguments": call.arguments,
                }
                self._checkpoint(trace_step, trace, None)
                observation = await self._execute_budgeted(call.name, call.arguments)
                self._pending_action = None
                trace.log_step(
                    step=trace_step,
                    model_output=response.content,
                    parsed_action="tool_call",
                    tool_name=call.name,
                    arguments=call.arguments,
                    observation=observation.output if observation.success else None,
                    error=observation.error,
                )
                content = json.dumps(
                    {
                        "success": observation.success,
                        "output": observation.output,
                        "error": observation.error,
                    },
                    ensure_ascii=False,
                    default=str,
                )
                self.context.add_message("tool", content, tool_name=call.name, tool_call_id=call.id)
                self._checkpoint(trace_step, trace, observation)
                if (
                    observation.error_kind == "permission"
                    and observation.error
                    and "approval" in observation.error
                ):
                    self._pending_action = {
                        "state": "awaiting_approval",
                        "tool_name": call.name,
                        "arguments": call.arguments,
                    }
                    return self._result(
                        False,
                        None,
                        observation.error,
                        trace,
                        round_number,
                        RunStatus.WAITING_APPROVAL,
                    )
                if observation.success:
                    errors = 0
                    last_tool_failed = False
                else:
                    errors += 1
                    last_tool_failed = True
                    if errors >= self.max_errors:
                        return self._result(False, None, observation.error, trace, round_number)

        return self._result(
            False,
            None,
            f"超过最大执行步数（Exceeded maximum steps: {self.max_steps}）。",
            trace,
            self.max_steps,
        )

    async def _execute(self, name: str, arguments: dict[str, Any]) -> Observation:
        return await asyncio.to_thread(self.executor.execute, name, arguments)

    async def _execute_budgeted(self, name: str, arguments: dict[str, Any]) -> Observation:
        """Apply repeat and total-call budgets before invoking a tool."""
        self._tool_calls_taken += 1
        call_key = f"{name}:{json.dumps(arguments, sort_keys=True, default=str)}"
        self._tool_call_counts[call_key] = self._tool_call_counts.get(call_key, 0) + 1
        if self._tool_calls_taken > self.max_tool_calls:
            return Observation(
                tool_name=name,
                success=False,
                error=f"工具调用次数超过预算（Tool-call budget exceeded: {self.max_tool_calls}）。",
                error_kind="budget",
            )
        if self._tool_call_counts[call_key] > self.max_repeated_tool_calls:
            return Observation(
                tool_name=name,
                success=False,
                error=(
                    "已阻止重复的相同工具调用（Repeated identical tool call blocked "
                    f"after {self.max_repeated_tool_calls} executions）。"
                ),
                error_kind="budget",
            )
        return await self._execute(name, arguments)

    def _run_timed_out(self) -> bool:
        return self.max_run_seconds is not None and (
            time.monotonic() - self._started_at > self.max_run_seconds
        )

    def _append_portable_observation(self, call: ToolCall, observation: Observation) -> None:
        if observation.success:
            self.context.add_observation(call.tool_name, call.arguments, observation.output)
            return
        self.context.add_message("assistant", f"Called {call.tool_name}({call.arguments})")
        self.context.add_message(
            "tool",
            observation.error or "未知工具错误（Unknown tool error）",
            tool_name=call.tool_name,
        )
        self._inject_recovery(observation.error or "未知工具错误（Unknown tool error）")

    def _native_messages(self, task: str, memories: list[str]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": "You are MiniClaw, a careful agent. Use supplied native tools when needed. After tool results, continue or provide a final answer.",
            }
        ]
        memory_text = build_memory_prompt(memories)
        if memory_text:
            messages.append({"role": "system", "content": memory_text})
        for message in self.context.get_messages():
            role = message.get("role")
            if role == "summary":
                messages.append({"role": "system", "content": message.get("content", "")})
            elif role in {"system", "user", "assistant", "tool"}:
                native_message: dict[str, Any] = {
                    "role": role,
                    "content": message.get("content", ""),
                }
                if role == "assistant" and message.get("tool_calls"):
                    native_message["tool_calls"] = message["tool_calls"]
                if role == "tool" and message.get("tool_call_id"):
                    native_message["tool_call_id"] = message["tool_call_id"]
                if role == "tool" and message.get("name"):
                    native_message["name"] = message["name"]
                messages.append(native_message)
        return messages

    def _prepare_native_request(
        self, task: str, memories: list[str], tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]] | None:
        """Build a provider-valid native request within the actual request budget."""
        # A checkpoint created by an older release may not include the task in
        # its context. Add it once so reconstruction remains valid.
        if not any(
            message.get("role") == "user" and message.get("content") == task
            for message in self.context.get_messages()
        ):
            self.context.add_message("user", task)

        for _ in range(3):
            messages = self._native_messages(task, memories)
            if self._native_request_tokens(messages, tools) <= self.context.max_context_tokens:
                return messages
            if not self.context.compress(force=True):
                break
        messages = self._native_messages(task, memories)
        if self._native_request_tokens(messages, tools) <= self.context.max_context_tokens:
            return messages
        return None

    def _native_request_tokens(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> int:
        """Estimate the whole provider payload, including function schemas."""
        return self.llm.count_tokens(messages) + len(json.dumps(tools, default=str)) // 4

    def _portable_schemas(self) -> list[dict[str, Any]]:
        return [
            schema for name in self.registry.list() if (schema := self.registry.get_schema(name))
        ]

    def _portable_history(self) -> list[dict[str, Any]]:
        messages = self.context.get_messages()
        latest_user_turn = max(
            (
                index
                for index, message in enumerate(messages)
                if message.get("role") == "user" and message.get("user_turn")
            ),
            default=-1,
        )
        history: list[dict[str, Any]] = []
        for index, message in enumerate(messages):
            role = message.get("role")
            if role not in {"user", "assistant", "tool", "summary"}:
                continue
            if index == latest_user_turn:
                continue
            history.append(
                {
                    "step": "",
                    "action": f"{role}: {message.get('content', '')}",
                    "observation": message.get("content", "") if role == "tool" else "",
                }
            )
        return history

    async def _search_memories(self, task: str, user_id: str) -> list[str]:
        try:
            return await asyncio.to_thread(self.memory_manager.search, task, user_id, 5)
        except Exception as exc:
            logger.warning("Memory search failed: %s", exc)
            return []

    async def _maybe_save_memory(self, task: str, user_id: str) -> None:
        try:
            await asyncio.to_thread(
                self.memory_manager.maybe_add_from_task,
                task,
                "",
                user_id,
            )
        except Exception as exc:
            logger.warning("Memory save failed: %s", exc)

    def _inject_recovery(self, error: str) -> None:
        self.context.add_message("user", f"[Recovery] {error}")

    def _cancel_result(self, trace: TraceLogger, steps: int) -> AgentResult | None:
        if not self._cancel_requested:
            return None
        trace.log_step(
            step=max(steps + 1, 1),
            parsed_action="cancelled",
            error="任务已由调用方取消（Run cancelled by caller.）",
        )
        return self._result(
            False,
            None,
            "任务已由调用方取消（Run cancelled by caller.）",
            trace,
            steps,
            RunStatus.CANCELLED,
        )

    def _result(
        self,
        success: bool,
        answer: str | None,
        error: str | None,
        trace: TraceLogger,
        steps: int,
        status: RunStatus | None = None,
    ) -> AgentResult:
        final_status = status or (RunStatus.COMPLETED if success else RunStatus.FAILED)
        self._set_status(final_status, error)
        self._checkpoint(steps, trace, None)
        return AgentResult(success, answer, error, trace, steps, final_status, self.run_id)

    def _set_status(self, status: RunStatus, error: str | None = None) -> None:
        self.status = status
        if self.store is not None and self.run_id:
            self.store.update_run(self.run_id, status.value, error)

    def _checkpoint(
        self, sequence: int, trace: TraceLogger, observation: Observation | None
    ) -> None:
        if self.store is None or not self.run_id:
            return
        state: dict[str, Any] = {
            "task": self._task,
            "mode": self.parse_mode,
            "status": self.status.value,
            "context": self.context.get_messages(),
            "trace": [step.model_dump(mode="json") for step in trace.steps],
        }
        if self._pending_action is not None:
            state["pending_action"] = self._pending_action
        if observation is not None:
            state["last_observation"] = observation.model_dump(mode="json")
        self.store.save_checkpoint(self.run_id, sequence, redact_secrets(state))


def _tool_call_payloads(response: LLMResponse) -> list[dict[str, Any]]:
    return [
        {
            "id": call.id,
            "type": "function",
            "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
        }
        for call in response.tool_calls
    ]


def _assistant_tool_message(response: LLMResponse) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": response.content or None,
        "tool_calls": _tool_call_payloads(response),
    }
