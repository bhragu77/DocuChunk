"""
LangfuseTracer — the real backend behind the Phase 9A tracing seam (Phase 9B).

It implements the 9A `Tracer` / `Span` protocols, so NOTHING at the instrumentation
call sites changes: they still do `with span(...) as sp: sp.update(...)`.

Fail-open is non-negotiable. Every Langfuse SDK call is individually wrapped in
try/except that logs at DEBUG and continues. `update()`/`score()` only BUFFER into
plain Python dicts on the request path (no network, non-blocking); the single SDK
write per span happens at span end, and the client's background thread does the
actual HTTP flushing. A dead/slow/unreachable Langfuse therefore cannot change a
request's outcome or latency.

Version isolation: every version-specific SDK call lives in exactly two methods,
`_start_span` and `_end_span`. An SDK upgrade touches only those.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from app.observability import pricing

logger = logging.getLogger("app.observability.langfuse")

# Field keys the tracer maps to first-class Langfuse concepts rather than metadata.
_TEXT_INPUT_KEYS = ("query", "prompt")
_SPECIAL_KEYS = frozenset(
    {
        "query", "prompt", "model", "user",
        "input_tokens", "output_tokens", "cost_usd", "cost_inr",
        "status", "error", "error_type",
    }
)
# Scores that are ALSO mirrored to trace level so they chart over time cleanly.
_TRACE_LEVEL_SCORES = frozenset({"groundedness", "citation_validity", "confidence"})


class LangfuseSpan:
    """A buffered span. update()/score() never touch the SDK — they fill dicts the
    tracer drains once, at _end_span. `handle` is the SDK stateful client (or None
    if creation failed — then this span is an inert but valid no-op)."""

    __slots__ = ("handle", "kind", "name", "trace_id", "fields", "scores")

    def __init__(self, handle: Any, kind: str, name: str, trace_id: Optional[str], fields: dict):
        self.handle = handle
        self.kind = kind              # "trace" | "span" | "generation"
        self.name = name
        self.trace_id = trace_id
        self.fields: dict = dict(fields)
        self.scores: list[tuple[str, float, Optional[str]]] = []

    def update(self, **fields: Any) -> None:
        self.fields.update(fields)

    def score(self, name: str, value: float, comment: Optional[str] = None) -> None:
        self.scores.append((name, value, comment))


class LangfuseTracer:
    def __init__(self, public_key: str, secret_key: str, host: str, client=None):
        # `client` is injectable for tests (a fake that records SDK calls); prod
        # passes None and the real SDK client is built.
        if client is not None:
            self._client = client
            logger.info("LangfuseTracer initialised [injected client]")
            return
        # Lazy import stays inside the caller's try (see _build_tracer); a missing
        # SDK raises here and is turned into a NullTracer, never a startup failure.
        from langfuse import Langfuse

        self._client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            timeout=5,            # per-HTTP-request cap on the background flusher
            flush_at=50,          # batch size
            flush_interval=1.0,   # seconds between background flushes
        )
        logger.info("LangfuseTracer initialised [host=%s]", host)

    # ── Tracer protocol ───────────────────────────────────────────────────────

    @contextmanager
    def span(
        self,
        name: str,
        *,
        parent: Any = None,
        trace_id: Optional[str] = None,
        **input_fields: Any,
    ) -> Iterator[LangfuseSpan]:
        sp = self._start_span(name, parent, trace_id, input_fields)
        try:
            yield sp
        finally:
            self._end_span(sp)

    def flush(self) -> None:
        try:
            self._client.flush()
        except Exception:
            logger.debug("tracing: langfuse flush failed", exc_info=True)

    # ── Helpers (SDK-agnostic) ────────────────────────────────────────────────

    @staticmethod
    def _split_fields(fields: dict) -> tuple[Optional[str], dict]:
        """(input_text, metadata) — text keys become the observation input, the
        rest (counts/ids/versions/latencies) become metadata."""
        text = None
        for k in _TEXT_INPUT_KEYS:
            if fields.get(k):
                text = fields[k]
                break
        meta = {k: v for k, v in fields.items() if k not in _SPECIAL_KEYS and v is not None}
        return text, meta

    @staticmethod
    def _level(fields: dict) -> tuple[Optional[str], Optional[str]]:
        if fields.get("status") == "error":
            return "ERROR", (fields.get("error") or fields.get("error_type"))
        return None, None

    # ══════════════════════════════════════════════════════════════════════════
    # THE ONLY VERSION-SPECIFIC SDK CODE LIVES BELOW. An SDK upgrade touches only
    # _start_span and _end_span.
    # ══════════════════════════════════════════════════════════════════════════

    def _start_span(self, name: str, parent: Any, trace_id: Optional[str], fields: dict) -> LangfuseSpan:
        text, meta = self._split_fields(fields)
        # Root → a Langfuse trace; child → a span, or a generation for "generate".
        if parent is None:
            handle = None
            tid = trace_id
            try:
                handle = self._client.trace(
                    id=trace_id or None,
                    name=name,
                    user_id=fields.get("user") or None,
                    input=text,
                    metadata=meta or None,
                )
                tid = getattr(handle, "id", None) or trace_id
            except Exception:
                logger.debug("tracing: trace() failed for %s", name, exc_info=True)
            return LangfuseSpan(handle, "trace", name, tid, fields)

        parent_handle = getattr(parent, "handle", None)
        tid = getattr(parent, "trace_id", None) or trace_id
        if parent_handle is None:
            return LangfuseSpan(None, "span", name, tid, fields)

        if name == "generate":
            handle = None
            try:
                handle = parent_handle.generation(
                    name=name, model=fields.get("model"), input=text, metadata=meta or None,
                )
            except Exception:
                logger.debug("tracing: generation() failed", exc_info=True)
            return LangfuseSpan(handle, "generation", name, tid, fields)

        handle = None
        try:
            handle = parent_handle.span(name=name, input=text, metadata=meta or None)
        except Exception:
            logger.debug("tracing: span() failed for %s", name, exc_info=True)
        return LangfuseSpan(handle, "span", name, tid, fields)

    def _end_span(self, sp: LangfuseSpan) -> None:
        fields = sp.fields
        _, meta = self._split_fields(fields)
        level, status_message = self._level(fields)

        it = fields.get("input_tokens")
        ot = fields.get("output_tokens")
        cost = fields.get("cost_usd")
        if fields.get("cost_inr") is not None:
            meta["cost_inr"] = fields["cost_inr"]

        if sp.handle is not None:
            try:
                if sp.kind == "trace":
                    self._client.trace(id=sp.trace_id, metadata=meta or None)
                elif sp.kind == "generation":
                    kwargs: dict[str, Any] = {"metadata": meta or None}
                    if level:
                        kwargs["level"] = level
                        kwargs["status_message"] = status_message
                    if it is not None and ot is not None:
                        kwargs["usage"] = {"input": it, "output": ot, "unit": "TOKENS"}
                    if cost is not None:
                        kwargs["cost_details"] = {"total": cost}
                    if fields.get("model"):
                        kwargs["model"] = fields["model"]
                    sp.handle.end(**kwargs)
                else:  # plain span
                    kwargs = {"metadata": meta or None}
                    if level:
                        kwargs["level"] = level
                        kwargs["status_message"] = status_message
                    sp.handle.end(**kwargs)
            except Exception:
                logger.debug("tracing: end failed for %s", sp.name, exc_info=True)

        # Scores: attach to the observation AND mirror the chartable ones to the
        # trace so they trend over time regardless of which span carried them.
        for score_name, value, comment in sp.scores:
            if sp.handle is not None:
                try:
                    sp.handle.score(name=score_name, value=value, comment=comment)
                except Exception:
                    logger.debug("tracing: obs score failed", exc_info=True)
            if score_name in _TRACE_LEVEL_SCORES and sp.trace_id:
                try:
                    self._client.score(trace_id=sp.trace_id, name=score_name, value=value, comment=comment)
                except Exception:
                    logger.debug("tracing: trace score failed", exc_info=True)

        # Abstention is a boolean FIELD (9A sets it on both the verify span and the
        # root). Surface it as a TRACE score so it charts alongside groundedness
        # (acceptance c). Emit it from the non-root span that carries it (the verify
        # span) — NOT the root: on the streaming path the root is finalised before
        # its abstained field is set, and emitting from both would double-count.
        if sp.kind != "trace" and "abstained" in fields and sp.trace_id:
            try:
                self._client.score(
                    trace_id=sp.trace_id, name="abstained",
                    value=1.0 if fields.get("abstained") else 0.0,
                )
            except Exception:
                logger.debug("tracing: abstained score failed", exc_info=True)
