"""
Public tracing API + the fail-open boundary.

Everything the rest of the app touches goes through here:

    from app.observability import span, hash_user_id, capture_text_enabled

    with span("rag.request", user=hash_user_id(uid)) as root:
        with span("retrieve", k=k) as sp:
            ...
            sp.update(candidate_count=n)

Guarantees (the point of Phase 9A):
  * FAIL-OPEN. Every call into a tracer/span implementation is individually
    wrapped in try/except that logs at DEBUG and continues. A tracer that raises
    on any method cannot change a request's outcome.
  * span() ALWAYS yields a usable Span. If span creation fails it yields a NullSpan
    proxy, so callers never write `if span is not None`.
  * Business exceptions raised inside a span block are recorded on the span and
    then RE-RAISED unchanged. Tracing never swallows, retries, or alters flow.
  * No blocking on the request path. flush() is the only blocking call, used at
    shutdown and at the end of each worker job.

Nothing outside this package may import a tracing vendor SDK. Ever.
"""
from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from app.observability import context, pricing
from app.observability.base import (
    NULL_SPAN,
    NullSpan,
    NullTracer,
    Span,
    Tracer,
    hash_user_id,
)
from app.observability.context import (
    get_prompt_version,
    new_trace_id,
    set_prompt_version,
)

logger = logging.getLogger("app.observability")

TRACING_ENABLED_ENV = "TRACING_ENABLED"
TRACE_CAPTURE_ENV = "TRACE_CAPTURE"

# Capture levels — see PRIVACY in the phase spec.
CAPTURE_NONE = "none"
CAPTURE_METADATA = "metadata"   # default: ids, counts, tokens, latencies, scores
CAPTURE_FULL = "full"           # adds prompt/completion text — DEV ONLY

_VALID_CAPTURE = (CAPTURE_NONE, CAPTURE_METADATA, CAPTURE_FULL)

# ── Tracer singleton (memoised) ───────────────────────────────────────────────

_tracer_lock = threading.Lock()
_tracer: Optional[Tracer] = None


def _tracing_enabled() -> bool:
    return os.getenv(TRACING_ENABLED_ENV, "").strip().lower() == "true"


def _build_tracer() -> Tracer:
    """Construct the active tracer when TRACING_ENABLED=true.

    Returns the Langfuse-backed tracer when keys are present, else NullTracer.
    Startup must NEVER fail because of tracing: a missing SDK, absent keys, or any
    initialisation error logs a WARNING and falls back to NullTracer.
    """
    try:
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY", "").strip()
        secret_key = os.getenv("LANGFUSE_SECRET_KEY", "").strip()
        host = os.getenv("LANGFUSE_HOST", "").strip() or "https://cloud.langfuse.com"
        if not public_key or not secret_key:
            logger.warning(
                "TRACING_ENABLED but LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY absent "
                "— tracing disabled (NullTracer)."
            )
            return NullTracer()
        # Lazy import — kept inside this try so a missing SDK degrades to NullTracer.
        from app.observability.langfuse_tracer import LangfuseTracer

        return LangfuseTracer(public_key=public_key, secret_key=secret_key, host=host)
    except Exception as exc:
        logger.warning("Langfuse tracer init failed (%s) — tracing disabled (NullTracer).", exc)
        return NullTracer()


def get_tracer() -> Tracer:
    """Return the memoised process-wide tracer.

    NullTracer whenever TRACING_ENABLED is not "true" (or until Phase 9B wires a
    real backend). Memoised so repeated calls on the hot path are free.
    """
    global _tracer
    if _tracer is not None:
        return _tracer
    with _tracer_lock:
        if _tracer is None:
            _tracer = _build_tracer() if _tracing_enabled() else NullTracer()
        return _tracer


def set_tracer(tracer: Tracer) -> None:
    """Install a specific tracer (tests inject a recording in-memory tracer)."""
    global _tracer
    with _tracer_lock:
        _tracer = tracer


def reset_tracer() -> None:
    """Drop the memoised tracer so the next get_tracer() rebuilds from env."""
    global _tracer
    with _tracer_lock:
        _tracer = None


def flush() -> None:
    """Flush buffered trace data. BLOCKING — shutdown / end-of-job only, never on
    the request path. Fail-open like everything else."""
    try:
        get_tracer().flush()
    except Exception:
        logger.debug("tracing: flush failed", exc_info=True)


# ── Capture level (privacy) ───────────────────────────────────────────────────

def capture_level() -> str:
    val = os.getenv(TRACE_CAPTURE_ENV, CAPTURE_METADATA).strip().lower()
    return val if val in _VALID_CAPTURE else CAPTURE_METADATA


def capture_text_enabled() -> bool:
    """True only at capture level `full`: the sole level that records prompt /
    completion / document text. Instrumentation must gate all raw text on this."""
    return capture_level() == CAPTURE_FULL


# ── Fail-open Span proxy ──────────────────────────────────────────────────────

class _SafeSpan:
    """Wraps a tracer Span so EVERY call into it is individually guarded and the
    capture level is enforced centrally. This is the only Span type callers ever
    hold, so a raising tracer is contained here and privacy is applied in one place.
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: Span):
        self._inner = inner

    def update(self, **fields: Any) -> None:
        if capture_level() == CAPTURE_NONE:
            return
        try:
            self._inner.update(**fields)
        except Exception:
            logger.debug("tracing: span.update failed", exc_info=True)

    def score(self, name: str, value: float, comment: Optional[str] = None) -> None:
        if capture_level() == CAPTURE_NONE:
            return
        try:
            self._inner.score(name, value, comment)
        except Exception:
            logger.debug("tracing: span.score failed", exc_info=True)


def _record_exception(sp: _SafeSpan, exc: BaseException) -> None:
    """Record a business exception on the span (never raises). The wrapper then
    re-raises the original exception unchanged."""
    try:
        sp.update(status="error", error_type=type(exc).__name__, error=str(exc)[:500])
    except Exception:
        logger.debug("tracing: failed to record exception", exc_info=True)


class _SpanCtx:
    """The context manager returned by span(). Written as an explicit class (not a
    @contextmanager generator) so the streaming endpoint can drive one manually
    across an async generator boundary via enter()/exit()."""

    __slots__ = ("_name", "_trace_id_arg", "_fields", "_cm", "_trace_token", "_span_token", "_safe")

    def __init__(self, name: str, trace_id: Optional[str], fields: dict):
        self._name = name
        self._trace_id_arg = trace_id
        self._fields = fields
        self._cm = None
        self._trace_token = None
        self._span_token = None
        self._safe: Optional[_SafeSpan] = None

    def __enter__(self) -> _SafeSpan:
        tracer = get_tracer()
        parent = context.current_span()

        # A root span (no parent) seeds the trace id — either the one handed in
        # (cross-process propagation) or a freshly minted one. Children inherit it.
        self._trace_token = None
        if parent is None:
            tid = self._trace_id_arg or context.get_trace_id() or context.new_trace_id()
            self._trace_token = context.set_trace_id(tid)
        else:
            tid = context.get_trace_id()

        # The parent handed to the tracer is the concrete inner span, not our proxy.
        parent_inner = getattr(parent, "_inner", parent)

        inner: Optional[Span] = None
        if capture_level() != CAPTURE_NONE:
            try:
                self._cm = tracer.span(self._name, parent=parent_inner, trace_id=tid, **self._fields)
                inner = self._cm.__enter__()
            except Exception:
                logger.debug("tracing: failed to create span %s", self._name, exc_info=True)
                self._cm = None
                inner = None
        if inner is None:
            inner = NULL_SPAN
            self._cm = None

        self._safe = _SafeSpan(inner)
        self._span_token = context.push_span(self._safe)
        return self._safe

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Record a propagating business exception, then let it re-raise unchanged.
        if exc is not None and self._safe is not None:
            _record_exception(self._safe, exc)

        if self._span_token is not None:
            try:
                context.pop_span(self._span_token)
            except Exception:
                logger.debug("tracing: pop_span failed", exc_info=True)
            self._span_token = None
        if self._trace_token is not None:
            try:
                context.reset_trace_id(self._trace_token)
            except Exception:
                logger.debug("tracing: reset_trace_id failed", exc_info=True)
            self._trace_token = None
        if self._cm is not None:
            try:
                self._cm.__exit__(exc_type, exc, tb)
            except Exception:
                logger.debug("tracing: span exit failed for %s", self._name, exc_info=True)
            self._cm = None
        return False  # never suppress — tracing does not alter control flow


def span(name: str, *, trace_id: Optional[str] = None, **input_fields: Any) -> _SpanCtx:
    """Open a span. Use as `with span(name, **fields) as sp:`.

    ALWAYS yields a usable Span (a NullSpan proxy if the tracer is a no-op or span
    creation fails). Nesting is derived from the contextvar stack; `trace_id` is
    only for the root of a cross-process trace (the worker seeds it from the job).
    """
    return _SpanCtx(name, trace_id, input_fields)


def current_trace_id() -> Optional[str]:
    """The active trace id, or None outside any trace."""
    return context.get_trace_id()


def record_generation(
    model: Optional[str],
    usage: Optional[dict],
    prompt_version: Optional[str] = None,
) -> None:
    """Attach provider token usage + computed USD/INR cost to the CURRENT span.

    Called from the generation seam (make_llm_fn / make_verify_fn) while the
    generate/verify span is active — NOT from a 9A instrumentation call site. Cost
    is computed from pricing.py (config), never fabricated: an unpriced model or
    absent usage simply records fewer fields. Fully fail-open and a no-op under the
    NullTracer (NullSpan.update does nothing).
    """
    sp = context.current_span()
    if sp is None:
        return
    fields: dict = {}
    if model:
        fields["model"] = model
    input_tokens = usage.get("input_tokens") if usage else None
    output_tokens = usage.get("output_tokens") if usage else None
    if input_tokens is not None:
        fields["input_tokens"] = input_tokens
    if output_tokens is not None:
        fields["output_tokens"] = output_tokens
    usd = pricing.cost_usd(model, input_tokens, output_tokens)
    if usd is not None:
        fields["cost_usd"] = usd
        fields["cost_inr"] = pricing.to_inr(usd)
    if prompt_version:
        fields["prompt_version"] = prompt_version
    if not fields:
        return
    try:
        sp.update(**fields)
    except Exception:
        logger.debug("tracing: record_generation update failed", exc_info=True)


@contextmanager
def adopt(parent_span: Any, trace_id: Optional[str] = None) -> Iterator[None]:
    """Re-establish an existing span as the current parent (with its trace id) in a
    context that no longer has it on the stack — e.g. inside a StreamingResponse
    generator that runs after the endpoint's own `with span(...)` block has closed.

    Fail-open: any bookkeeping error is swallowed; the body still runs.
    """
    span_token = None
    trace_token = None
    try:
        if trace_id is not None:
            trace_token = context.set_trace_id(trace_id)
        if parent_span is not None:
            span_token = context.push_span(parent_span)
    except Exception:
        logger.debug("tracing: adopt setup failed", exc_info=True)
    try:
        yield
    finally:
        if span_token is not None:
            try:
                context.pop_span(span_token)
            except Exception:
                logger.debug("tracing: adopt pop failed", exc_info=True)
        if trace_token is not None:
            try:
                context.reset_trace_id(trace_token)
            except Exception:
                logger.debug("tracing: adopt reset failed", exc_info=True)


__all__ = [
    "span",
    "adopt",
    "current_trace_id",
    "new_trace_id",
    "record_generation",
    "set_prompt_version",
    "get_prompt_version",
    "get_tracer",
    "set_tracer",
    "reset_tracer",
    "flush",
    "hash_user_id",
    "capture_level",
    "capture_text_enabled",
    "CAPTURE_NONE",
    "CAPTURE_METADATA",
    "CAPTURE_FULL",
    "Span",
    "Tracer",
    "NullSpan",
    "NullTracer",
]
