"""
Story 6 — Server-Sent Events (SSE) helpers + the streaming generation seam.

Two small, provider-agnostic pieces used by POST /generate/answer when
stream=true:

  * sse_event(event, data) — format one SSE frame ("event: <name>\\n
    data: <json>\\n\\n"). This is the wire format the frontend's EventSource
    parses.

  * stream_generation(prompt, stream_fn, llm_fn) — yield the answer as text
    chunks, with a graceful fallback: if no streaming callable is wired, or the
    provider's generate_stream RAISES on the very first chunk, fall back to the
    non-streaming generate() and yield the whole answer as ONE chunk. A provider
    without real streaming (or a transient streaming failure) still produces a
    valid SSE response — just one big token event instead of many.

The two-phase design (stream tokens → THEN verify) lives in the endpoint; this
module only concerns Phase 1's token production and its serialization.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)


def sse_event(event: str, data: dict[str, Any]) -> str:
    """Serialize one SSE frame: a named event carrying a JSON data payload."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def stream_generation(
    prompt: str,
    stream_fn: Callable[[str], Iterator[str]] | None,
    llm_fn: Callable[[str], str],
) -> Iterator[str]:
    """
    Yield the answer for `prompt` as text chunks (Phase 1 of the SSE flow).

    Fallback policy (graceful degradation, never a crash):
      * stream_fn is None            → fall back to llm_fn(), one chunk.
      * stream_fn RAISES on the first chunk → fall back to llm_fn(), one chunk.
      * stream_fn raises MID-stream (after ≥1 chunk) → stop and keep the partial
        answer; the endpoint still runs verification on what was produced. We do
        NOT restart from generate() there — that would re-emit already-sent text.
      * stream_fn yields nothing (empty stream) → a refusal; yield nothing.

    llm_fn is the same non-streaming callable the verification phase uses, so the
    fallback answer is exactly what the non-streaming path would have produced.
    """
    if stream_fn is not None:
        try:
            iterator = iter(stream_fn(prompt))
            first = next(iterator)
        except StopIteration:
            return  # empty stream → refusal, nothing to yield
        except Exception as exc:  # first-chunk failure → fall through to generate()
            logger.warning(
                "generate_stream failed on first chunk (%s); falling back to generate()", exc
            )
        else:
            yield first
            try:
                yield from iterator
            except Exception as exc:  # mid-stream failure → keep the partial answer
                logger.warning("generate_stream failed mid-stream (%s); using partial answer", exc)
            return

    # Fallback: non-streaming generate as a single chunk.
    text = (llm_fn(prompt) or "").strip()
    if text:
        yield text
