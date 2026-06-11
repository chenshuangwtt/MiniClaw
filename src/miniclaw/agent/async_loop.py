"""AsyncAgentLoop — async variant of the agent loop.

Mirrors :class:`AgentLoop` but wraps blocking calls (LLM, tool execution)
with ``asyncio.to_thread()`` so the event loop stays responsive.

Usage::

    import asyncio
    from miniclaw.agent.async_loop import AsyncAgentLoop

    loop = AsyncAgentLoop(llm=my_llm, registry=my_registry)
    result = asyncio.run(loop.run("List all Python files"))
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from miniclaw.agent.context import ContextManager
from miniclaw.agent.executor import ToolExecutor
from miniclaw.agent.parser import OutputParser, ParseError
from miniclaw.agent.prompts import build_full_prompt
from miniclaw.agent.recovery import RecoveryManager
from miniclaw.agent.state import AgentOutput, FinalAnswer, ToolCall
from miniclaw.agent.trace import TraceLogger
from miniclaw.llm.base import BaseLLM
from miniclaw.memory.base import MemoryBackend, NullMemoryBackend
from miniclaw.memory.extractor import MemoryExtractor
from miniclaw.tools.audit import AuditLogger
from miniclaw.tools.permissions import PermissionPolicy
from miniclaw.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

# Re-export AgentResult for convenience
from miniclaw.agent.loop import AgentResult  # noqa: E402

DEFAULT_MAX_STEPS = 10
DEFAULT_MAX_ERRORS = 3


class AsyncAgentLoop:
    """Async variant of the agent loop.

    All blocking operations (LLM calls, tool execution, memory search)
    are wrapped with ``asyncio.to_thread()`` so the event loop stays
    responsive.

    Attributes:
        max_steps: Maximum number of tool-call rounds before forced stop.
        max_errors: Maximum consecutive parse/execution errors before aborting.
    """

    def __init__(
        self,
        llm: BaseLLM,
        registry: ToolRegistry,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_errors: int = DEFAULT_MAX_ERRORS,
        context: ContextManager | None = None,
        recovery: RecoveryManager | None = None,
        memory_backend: MemoryBackend | None = None,
        memory_extractor: MemoryExtractor | None = None,
        permission_policy: PermissionPolicy | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.audit_logger = audit_logger
        self.permission_policy = permission_policy or PermissionPolicy()
        self.executor = ToolExecutor(
            registry,
            permission_policy=self.permission_policy,
            audit_logger=audit_logger,
        )
        self.parser = OutputParser()
        self.parse_mode = "auto"  # "auto" | "portable" | "native"
        self.max_steps = max_steps
        self.max_errors = max_errors
        self.context = context or ContextManager()
        self.recovery = recovery or RecoveryManager(max_errors=max_errors)
        self.memory_backend = memory_backend or NullMemoryBackend()
        self.memory_extractor = memory_extractor or MemoryExtractor()

    async def run(self, user_task: str, user_id: str = "default") -> AgentResult:
        """Run the agent loop on a user task.

        Args:
            user_task: The task description from the user.
            user_id: User identifier for memory retrieval and storage.

        Returns:
            An ``AgentResult`` with the final answer (or error) and full trace.
        """
        # Retrieve related memories (safe — never crashes)
        related_memories = await self._search_memories(user_task, user_id)

        # Run the core loop
        result = await self._run_loop(user_task, related_memories)

        # Save memory if the task is worth remembering (safe — never crashes)
        if result.success and result.answer:
            await self._maybe_save_memory(user_task, result.answer, user_id)

        return result

    async def _run_loop(self, user_task: str, related_memories: list[str]) -> AgentResult:
        """Core loop — separated from run() so memory logic stays clean."""
        trace = TraceLogger()
        error_count = 0
        tool_schemas = self._get_tool_schemas()

        for step in range(1, self.max_steps + 1):
            # 0. Compress context if needed
            if self.context.should_compress():
                self.context.compress()

            # 1. Build prompt from context
            prompt = self._build_prompt(user_task, tool_schemas, related_memories)

            # 2. Call LLM
            try:
                raw_output, parsed = await self._call_and_parse(prompt, tool_schemas)
            except Exception as exc:
                error_count += 1
                trace.log_step(
                    step=step,
                    error=f"LLM error: {exc}",
                    parsed_action="llm_error",
                )
                logger.error("Step %d: LLM call failed: %s", step, exc)
                if error_count >= self.max_errors:
                    return AgentResult(
                        success=False,
                        answer=None,
                        error=f"Aborted after {error_count} consecutive errors (last: {exc}).",
                        trace=trace,
                        steps_taken=step,
                    )
                continue

            # 3. Handle parse failure
            if parsed is None:
                error_count += 1
                trace.log_step(
                    step=step,
                    model_output=raw_output,
                    parsed_action="parse_error",
                    error="Could not parse LLM output",
                )
                logger.warning("Step %d: Parse failed for output: %s", step, raw_output[:100])

                recovery_result = self.recovery.handle_invalid_json(raw_output)
                if recovery_result.get("status") == "repaired":
                    try:
                        parsed = self.parser.parse(recovery_result["output"])
                    except ParseError:
                        self._inject_recovery(step, recovery_result.get("error", "parse failed"))
                        if error_count >= self.max_errors:
                            return self._abort_result(error_count, trace, step)
                        continue
                else:
                    self._inject_recovery(step, recovery_result.get("error", "parse failed"))
                    if error_count >= self.max_errors:
                        return self._abort_result(error_count, trace, step)
                    continue

            # 4. Handle FinalAnswer
            if isinstance(parsed, FinalAnswer):
                trace.log_step(
                    step=step,
                    model_output=raw_output,
                    parsed_action="final_answer",
                    observation=parsed.answer,
                )
                self.context.add_message("assistant", parsed.answer)
                return AgentResult(
                    success=True,
                    answer=parsed.answer,
                    error=None,
                    trace=trace,
                    steps_taken=step,
                )

            # 5. Handle ToolCall
            assert isinstance(parsed, ToolCall)

            if self.registry.get(parsed.tool_name) is None:
                error_count += 1
                recovery_msg = self.recovery.handle_unknown_tool(parsed.tool_name, self.registry)
                trace.log_step(
                    step=step,
                    model_output=raw_output,
                    parsed_action="tool_call",
                    tool_name=parsed.tool_name,
                    arguments=parsed.arguments,
                    error=recovery_msg["error"],
                )
                self._inject_recovery(step, recovery_msg["error"])
                if error_count >= self.max_errors:
                    return self._abort_result(error_count, trace, step)
                continue

            obs = await asyncio.to_thread(self.executor.execute, parsed.tool_name, parsed.arguments)

            trace.log_step(
                step=step,
                model_output=raw_output,
                parsed_action="tool_call",
                tool_name=parsed.tool_name,
                arguments=parsed.arguments,
                observation=obs.output if obs.success else None,
                error=obs.error if not obs.success else None,
            )

            # 6. Append to context
            if obs.success:
                error_count = 0
                self.context.add_observation(parsed.tool_name, parsed.arguments, obs.output)
            else:
                error_count += 1
                recovery_msg = self.recovery.handle_tool_error(
                    parsed.tool_name, obs.error or "Unknown error"
                )
                self.context.add_message(
                    "assistant", f"Called {parsed.tool_name}({parsed.arguments})"
                )
                self.context.add_message("tool", recovery_msg["error"], tool_name=parsed.tool_name)
                if error_count >= self.max_errors:
                    return self._abort_result(error_count, trace, step)

        # Max steps exceeded
        return AgentResult(
            success=False,
            answer=None,
            error=f"Exceeded maximum steps ({self.max_steps}).",
            trace=trace,
            steps_taken=self.max_steps,
        )

    def _build_prompt(
        self,
        user_task: str,
        tool_schemas: list[dict[str, Any]],
        memories: list[str] | None = None,
    ) -> str:
        """Build the full prompt using ContextManager."""
        return build_full_prompt(
            user_task,
            tools=tool_schemas,
            history=self._context_to_history(),
            memories=memories,
        )

    async def _call_and_parse(
        self, prompt: str, tool_schemas: list[dict[str, Any]]
    ) -> tuple[str, AgentOutput | None]:
        """Call LLM and parse the response. Returns (raw_output, parsed).

        Supports three modes:
            - ``native``: Use ``llm.chat()`` with tools, parse tool_calls directly.
            - ``portable``: Use ``llm.generate()``, parse JSON from text.
            - ``auto``: Try native first; if no tool_calls, fall back to portable.

        Returns:
            Tuple of (raw_output_string, parsed_output_or_None).
            If parsing fails, ``parsed`` is ``None`` and caller should handle.
        """
        mode = self.parse_mode

        # Native mode: use chat() with tools
        if mode == "native" and tool_schemas:
            return await self._call_native(prompt, tool_schemas)

        # Portable mode: use generate() + JSON parsing
        if mode == "portable":
            return await self._call_portable(prompt)

        # Auto mode: try native if tools available, else portable
        if mode == "auto" and tool_schemas and hasattr(self.llm, "chat"):
            raw, parsed = await self._call_native(prompt, tool_schemas)
            if parsed is not None and isinstance(parsed, ToolCall):
                return raw, parsed
            # No native tool_calls — parse the content as portable JSON
            try:
                parsed = self.parser.parse(raw)
            except Exception:
                parsed = None
            return raw, parsed

        return await self._call_portable(prompt)

    async def _call_native(
        self, prompt: str, tool_schemas: list[dict[str, Any]]
    ) -> tuple[str, AgentOutput | None]:
        """Call LLM using native tool calling (OpenAI-style)."""
        messages = [{"role": "user", "content": prompt}]
        response = await asyncio.to_thread(self.llm.chat, messages, tool_schemas)
        raw_output = response.content or ""

        if response.has_tool_calls:
            tc = response.tool_calls[0]
            parsed = self.parser.parse_native(
                response.content,
                [
                    {
                        "id": tc.id,
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                ],
            )
            return raw_output, parsed

        # No tool_calls — return None to signal caller to try portable
        return raw_output, None

    async def _call_portable(self, prompt: str) -> tuple[str, AgentOutput | None]:
        """Call LLM using portable JSON protocol."""
        raw_output = await asyncio.to_thread(self.llm.generate, prompt)
        try:
            parsed = self.parser.parse(raw_output)
        except ParseError:
            parsed = None
        return raw_output, parsed

    async def _search_memories(self, user_task: str, user_id: str) -> list[str]:
        """Search memory backend for related memories. Never crashes."""
        try:
            return await asyncio.to_thread(self.memory_backend.search, user_task, user_id, 5)
        except Exception as exc:
            logger.warning("Memory search failed: %s", exc)
            return []

    async def _maybe_save_memory(self, user_task: str, answer: str, user_id: str) -> None:
        """Save task to memory if the extractor says it's worth remembering. Never crashes."""
        try:
            if await asyncio.to_thread(self.memory_extractor.should_remember, user_task):
                memories = await asyncio.to_thread(self.memory_extractor.extract, user_task)
                for mem in memories:
                    await asyncio.to_thread(self.memory_backend.add, mem, user_id)
        except Exception as exc:
            logger.warning("Memory save failed: %s", exc)

    def _context_to_history(self) -> list[dict[str, Any]]:
        """Convert context messages to the history format expected by prompts."""
        history = []
        for msg in self.context.get_messages():
            if msg.get("role") in ("user", "assistant", "tool"):
                history.append(
                    {
                        "step": "",
                        "action": msg.get("content", ""),
                        "observation": msg.get("content", "") if msg.get("role") == "tool" else "",
                    }
                )
        return history

    def _inject_recovery(self, step: int, error_msg: str) -> None:
        """Inject a recovery hint into the context."""
        self.context.add_message("user", f"[Recovery] {error_msg}")

    def _abort_result(self, error_count: int, trace: TraceLogger, step: int) -> AgentResult:
        """Build an abort AgentResult."""
        abort = self.recovery.handle_consecutive_failures(error_count)
        error_msg = abort["error"] if abort else f"Aborted after {error_count} errors."
        return AgentResult(
            success=False,
            answer=None,
            error=error_msg,
            trace=trace,
            steps_taken=step,
        )

    def _get_tool_schemas(self) -> list[dict[str, Any]]:
        """Collect all registered tool schemas."""
        schemas = []
        for name in self.registry.list():
            schema = self.registry.get_schema(name)
            if schema:
                schemas.append(schema)
        return schemas
