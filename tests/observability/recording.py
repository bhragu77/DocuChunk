"""
In-memory tracer implementations for observability tests.

RecordingTracer captures the full span tree (names, parent/child links, trace ids,
fields, scores) so tests can assert the exact structure the pipeline emits.

BrokenTracer raises on EVERY method — the fail-open contract must make a request
succeed byte-identically regardless.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Optional


class RecordingSpan:
    def __init__(self, name: str, parent: "Optional[RecordingSpan]", trace_id: Optional[str], fields: dict):
        self.name = name
        self.parent = parent
        self.trace_id = trace_id
        self.fields = dict(fields)
        self.scores: dict[str, tuple[float, Optional[str]]] = {}
        self.children: list["RecordingSpan"] = []
        self.exited = False

    def update(self, **fields: Any) -> None:
        self.fields.update(fields)

    def score(self, name: str, value: float, comment: Optional[str] = None) -> None:
        self.scores[name] = (value, comment)

    def child_names(self) -> list[str]:
        return [c.name for c in self.children]

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"RecordingSpan({self.name!r}, children={self.child_names()})"


class RecordingTracer:
    def __init__(self) -> None:
        self.spans: list[RecordingSpan] = []
        self.flush_count = 0

    @contextmanager
    def span(self, name: str, *, parent: Any = None, trace_id: Optional[str] = None, **input_fields: Any):
        sp = RecordingSpan(name, parent, trace_id, input_fields)
        self.spans.append(sp)
        if isinstance(parent, RecordingSpan):
            parent.children.append(sp)
        yield sp
        sp.exited = True

    def flush(self) -> None:
        self.flush_count += 1

    # ── Query helpers ─────────────────────────────────────────────────────────
    def roots(self) -> list[RecordingSpan]:
        return [s for s in self.spans if s.parent is None]

    def by_name(self, name: str) -> list[RecordingSpan]:
        return [s for s in self.spans if s.name == name]

    def only(self, name: str) -> RecordingSpan:
        matches = self.by_name(name)
        assert len(matches) == 1, f"expected exactly one {name!r} span, got {len(matches)}"
        return matches[0]


class BrokenTracer:
    """Every entry point raises — exercises the fail-open boundary."""

    def span(self, *args: Any, **kwargs: Any):
        raise RuntimeError("BrokenTracer.span boom")

    def flush(self) -> None:
        raise RuntimeError("BrokenTracer.flush boom")
