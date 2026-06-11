"""Tool timeout and cancellation support.

Provides a generic mechanism for running tool functions with a wall-clock
timeout and cooperative cancellation.  Uses ``concurrent.futures.ThreadPoolExecutor``
so it works on all platforms (including Windows, where ``signal.alarm`` is
unavailable).

Usage::

    from miniclaw.tools.timeout import run_with_timeout, CancellationToken, TimedOutError

    token = CancellationToken()
    try:
        result = run_with_timeout(my_fn, args=("arg1",), timeout=30, token=token)
    except TimedOutError:
        print("Tool timed out")

    # Cancel from another thread:
    token.cancel()
"""

from __future__ import annotations

import concurrent.futures
import logging
from dataclasses import dataclass
from typing import Any, Callable

logger = logging.getLogger(__name__)


class TimedOutError(Exception):
    """Raised when a tool call exceeds its timeout."""


@dataclass
class CancellationToken:
    """Cooperative cancellation token.

    Pass to ``run_with_timeout()`` and call ``cancel()`` from another
    thread to request cancellation.  The running function should check
    ``token.cancelled`` periodically to exit early.

    Attributes:
        cancelled: Whether cancellation has been requested.
    """

    cancelled: bool = False

    def cancel(self) -> None:
        """Request cancellation."""
        self.cancelled = True

    def check(self) -> None:
        """Raise ``CancelledError`` if cancellation has been requested."""
        if self.cancelled:
            raise concurrent.futures.CancelledError("Tool execution cancelled")


@dataclass
class ToolTimeout:
    """Configuration for a tool's timeout behaviour.

    Attributes:
        seconds: Maximum wall-clock seconds for the tool call.
        on_timeout: What to do on timeout — ``"error"`` raises
            ``TimedOutError``, ``"cancel"`` returns ``None``.
    """

    seconds: float = 30.0
    on_timeout: str = "error"  # "error" | "cancel"


def run_with_timeout(
    fn: Callable[..., Any],
    args: tuple[Any, ...] = (),
    kwargs: dict[str, Any] | None = None,
    timeout: float = 30.0,
    token: CancellationToken | None = None,
) -> Any:
    """Run *fn* in a thread with a wall-clock timeout.

    Args:
        fn: The function to call.
        args: Positional arguments.
        kwargs: Keyword arguments.
        timeout: Maximum seconds before raising ``TimedOutError``.
        token: Optional cancellation token.

    Returns:
        The return value of *fn*.

    Raises:
        TimedOutError: If *fn* doesn't complete within *timeout* seconds.
        CancelledError: If *token* was cancelled before completion.
        Exception: Any exception raised by *fn* is re-raised.
    """
    if kwargs is None:
        kwargs = {}

    # Check cancellation before even starting
    if token is not None:
        token.check()

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn, *args, **kwargs)

    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        if token is not None:
            token.cancel()
        future.cancel()
        logger.warning("Tool call timed out after %.1fs", timeout)
        raise TimedOutError(f"Tool timed out after {timeout}s") from None
    except concurrent.futures.CancelledError:
        raise
    except Exception:
        # Re-raise the original exception from fn
        raise
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
