"""
Trace-context propagation: the current span stack and the active trace id.

Both live in contextvars, NOT module globals. Async requests interleave on one
thread, so a module global would let one request's span become another's parent.
A contextvar is per-asyncio-Task, so each request (and each `asyncio.to_thread`
call, which copies the context) sees its own isolated stack.

Cross-PROCESS propagation is different: the API and the arq worker share no
memory, so the stack does not travel between them. The trace_id (a plain string)
is carried across that boundary in the job kwargs and re-seeded here on the worker
side — see app/workers/tasks.py.
"""
from __future__ import annotations

import contextvars
import uuid
from typing import Any, Optional, Tuple

# The nesting stack: the last element is the current (innermost) span. Immutable
# tuples + tokens make push/pop safe under interleaving without locks.
_span_stack: contextvars.ContextVar[Tuple[Any, ...]] = contextvars.ContextVar(
    "obs_span_stack", default=()
)
# The active trace id, seeded by the root span (or handed in across a process
# boundary). None outside any trace.
_trace_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "obs_trace_id", default=None
)

# The prompt-template version that built the CURRENT request's prompt. Set by
# build_grounded_prompt (in the prompt_build span) so the generate span can record
# which template version served the answer. Git remains the source of truth for the
# template body; this is only the identifier that flows to the trace.
_prompt_version: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "obs_prompt_version", default=None
)


def new_trace_id() -> str:
    """Mint a fresh trace id (uuid4 hex, 32 chars)."""
    return uuid.uuid4().hex


# ── Span stack ────────────────────────────────────────────────────────────────

def current_span() -> Optional[Any]:
    """The innermost active span, or None if no span is open in this context."""
    stack = _span_stack.get()
    return stack[-1] if stack else None


def push_span(span: Any) -> contextvars.Token:
    """Push `span` as the new innermost span; returns a token for pop_span()."""
    stack = _span_stack.get()
    return _span_stack.set(stack + (span,))


def pop_span(token: contextvars.Token) -> None:
    """Restore the stack to before the matching push_span()."""
    _span_stack.reset(token)


# ── Trace id ──────────────────────────────────────────────────────────────────

def get_trace_id() -> Optional[str]:
    return _trace_id.get()


def set_trace_id(trace_id: str) -> contextvars.Token:
    """Set the active trace id; returns a token for reset_trace_id()."""
    return _trace_id.set(trace_id)


def reset_trace_id(token: contextvars.Token) -> None:
    _trace_id.reset(token)


# ── Prompt version ────────────────────────────────────────────────────────────

def set_prompt_version(version: Optional[str]) -> None:
    """Record which prompt-template version built the current request's prompt."""
    _prompt_version.set(version)


def get_prompt_version() -> Optional[str]:
    return _prompt_version.get()
