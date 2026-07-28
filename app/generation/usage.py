"""
Per-request token-usage handoff (contextvar).

A generation provider is a SINGLETON shared by every concurrent request. Stashing
the last call's token usage on the instance (`provider.last_usage`) races: on the
streaming path the SSE generator suspends between token yields, and another request
can overwrite that shared attribute in the gap — so the first request would read a
DIFFERENT user's usage/cost.

A ContextVar is isolated per asyncio Task (and per `copy_context()`), so each
request reads back exactly the usage its own generation produced. The provider
writes it; the generation seam (factory) reads it right after the call — no shared
mutable state, no cross-request contamination.
"""
from __future__ import annotations

from contextvars import ContextVar
from typing import Optional

last_usage: ContextVar[Optional[dict]] = ContextVar("gen_last_usage", default=None)


def set_last_usage(usage: Optional[dict]) -> None:
    """Record the token usage the CURRENT request's last generation call reported."""
    last_usage.set(usage)


def get_last_usage() -> Optional[dict]:
    """The token usage from this request's last generation call, or None."""
    return last_usage.get()
