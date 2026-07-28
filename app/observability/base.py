"""
Tracing SEAM — protocols and no-op implementations (Phase 9A, part 1 of 2).

This module defines the *contract* the rest of the app instruments against. It is
deliberately vendor-neutral: NOTHING here (or anywhere outside this package)
imports a tracing SDK. Phase 9B swaps NullTracer for a real backend behind the
same protocols, with zero changes to the instrumented call sites.

Design rules baked into the contract:
  * FAIL-OPEN. A tracer implementation may do anything, including raise. The
    fail-open wrapper in app/observability/__init__.py individually guards every
    call into a Span/Tracer so a tracing failure can never break a request.
  * A Span is always usable. Callers never write `if span is not None`.
  * Tracing never alters control flow — business exceptions are recorded and
    re-raised unchanged by the wrapper, never here.
"""
from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Protocol, runtime_checkable

# Env var holding the salt mixed into hashed user ids. Kept out of Settings so the
# observability package stays self-contained and importable in isolation.
# TRACE_USER_SALT is the canonical name (Phase 9B); TRACE_SALT stays accepted as an
# alias so 9A deployments keep working.
TRACE_USER_SALT_ENV = "TRACE_USER_SALT"
TRACE_SALT_ENV = "TRACE_SALT"


@runtime_checkable
class Span(Protocol):
    """A unit of work in a trace.

    update() attaches structured fields (ids, counts, token usage, latencies,
    model names, …). score() attaches a named numeric evaluation (groundedness,
    confidence, …). Both are best-effort and must never raise to the caller — the
    fail-open wrapper guarantees that regardless of the implementation.
    """

    def update(self, **fields: Any) -> None: ...

    def score(self, name: str, value: float, comment: Optional[str] = None) -> None: ...


@runtime_checkable
class Tracer(Protocol):
    """Creates spans and flushes buffered data.

    span() is a context manager yielding a Span. Implementations receive the
    parent span (the concrete inner span object, or None for a root) and the
    active trace_id so they can build the parent/child tree and correlate across
    processes. flush() is the ONLY blocking call and runs only at shutdown / at
    the end of each worker job — never on the request path.
    """

    def span(
        self,
        name: str,
        *,
        parent: Any = None,
        trace_id: Optional[str] = None,
        **input_fields: Any,
    ) -> Iterator[Span]:  # actually a context manager yielding Span
        ...

    def flush(self) -> None: ...


class NullSpan:
    """A Span that does nothing. Valid, usable, and free of side effects."""

    __slots__ = ()

    def update(self, **fields: Any) -> None:
        return None

    def score(self, name: str, value: float, comment: Optional[str] = None) -> None:
        return None


# Shared singleton — NullSpan is stateless, so one instance is enough.
NULL_SPAN = NullSpan()


class NullTracer:
    """A Tracer that records nothing. The default whenever tracing is disabled."""

    __slots__ = ()

    @contextmanager
    def span(
        self,
        name: str,
        *,
        parent: Any = None,
        trace_id: Optional[str] = None,
        **input_fields: Any,
    ) -> Iterator[Span]:
        yield NULL_SPAN

    def flush(self) -> None:
        return None


def hash_user_id(user_id: Any) -> str:
    """Return a stable, salted 16-char sha256 prefix of `user_id`.

    User ids are NEVER recorded raw. The salt comes from TRACE_SALT (env); an unset
    salt still yields a stable hash, but setting one in production prevents trivial
    reversal of the (small) id space. Returns "" for a falsy id.
    """
    if not user_id:
        return ""
    salt = os.getenv(TRACE_USER_SALT_ENV) or os.getenv(TRACE_SALT_ENV, "")
    digest = hashlib.sha256(f"{salt}:{user_id}".encode("utf-8")).hexdigest()
    return digest[:16]
