"""Recovery manager — retry logic and repair prompts for LLM failures."""

from __future__ import annotations

import time
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


class RecoveryManager:
    """Wraps LLM calls with retry + repair-prompt injection.

    Handles three failure modes:
    1. **Transient errors** (network, rate-limit) → exponential backoff retry.
    2. **Unparseable responses** → inject a repair prompt and retry.
    3. **Max retries exceeded** → raise the last error or return a fallback.

    Attributes:
        max_retries: Maximum retry attempts for transient errors.
        backoff_base: Base delay (seconds) for exponential backoff.
        max_repair_attempts: Max attempts to repair an unparseable response.
    """

    REPAIR_PROMPT = (
        "Your previous response could not be parsed. "
        "Please reply with ONLY a valid JSON object. "
        "Do not include any text before or after the JSON."
    )

    def __init__(
        self,
        max_retries: int = 3,
        backoff_base: float = 1.0,
        max_repair_attempts: int = 2,
    ) -> None:
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.max_repair_attempts = max_repair_attempts

    def call_with_retry(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Call *fn* with exponential-backoff retry on transient errors.

        Args:
            fn: The callable to invoke (e.g., ``llm.chat``).
            *args, **kwargs: Forwarded to *fn*.

        Returns:
            The return value of *fn* on success.

        Raises:
            The last exception if all retries are exhausted.
        """
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    delay = self.backoff_base * (2 ** attempt)
                    logger.warning(
                        "Attempt %d/%d failed: %s — retrying in %.1fs",
                        attempt + 1,
                        self.max_retries + 1,
                        exc,
                        delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        "All %d attempts failed. Last error: %s",
                        self.max_retries + 1,
                        exc,
                    )
        raise last_exc  # type: ignore[misc]

    def get_repair_messages(
        self,
        messages: list[dict[str, Any]],
        bad_response: str,
    ) -> list[dict[str, Any]]:
        """Build a message list that asks the LLM to repair its output.

        Appends the bad response and a repair instruction to the
        original message list.

        Args:
            messages: Original messages sent to the LLM.
            bad_response: The unparseable response text.

        Returns:
            A new message list with the repair prompt appended.
        """
        repair_msgs = list(messages)
        repair_msgs.append({"role": "assistant", "content": bad_response})
        repair_msgs.append({"role": "user", "content": self.REPAIR_PROMPT})
        return repair_msgs
