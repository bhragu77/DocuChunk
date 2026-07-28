"""
A fake Langfuse SDK client that records every call the LangfuseTracer makes, so
tests can assert the emitted trace waterfall, token usage, cost, and scores WITHOUT
a live Langfuse server. Injected via LangfuseTracer(..., client=FakeLangfuse()).

Shape mirrors the subset of the langfuse v2 stateful API the tracer uses:
  client.trace(...)          -> FakeObs(kind="trace")
  client.score(trace_id=...) -> trace-level score
  client.flush()
  obs.span(...) / obs.generation(...) -> child FakeObs
  obs.end(**kwargs) / obs.score(...)
"""
from __future__ import annotations

from typing import Any, Optional


class FakeObs:
    def __init__(self, root: "FakeLangfuse", kind: str, name: str, obs_id: str, trace_id: str, **fields: Any):
        self._root = root
        self.kind = kind
        self.name = name
        self.id = obs_id
        self.trace_id = trace_id
        self.fields = dict(fields)
        self.end_kwargs: Optional[dict] = None
        self.scores: list[tuple] = []
        self.children: list["FakeObs"] = []
        root.all_obs.append(self)

    def span(self, name: str, input=None, metadata=None) -> "FakeObs":
        child = FakeObs(
            self._root, "span", name, self._root._next_id(), self.trace_id,
            input=input, metadata=metadata,
        )
        self.children.append(child)
        return child

    def generation(self, name: str, model=None, input=None, metadata=None) -> "FakeObs":
        child = FakeObs(
            self._root, "generation", name, self._root._next_id(), self.trace_id,
            model=model, input=input, metadata=metadata,
        )
        self.children.append(child)
        return child

    def end(self, **kwargs: Any) -> None:
        self.end_kwargs = kwargs

    def score(self, name=None, value=None, comment=None, **kwargs: Any) -> None:
        self.scores.append((name, value, comment))

    def child_names(self) -> list[str]:
        return [c.name for c in self.children]


class FakeLangfuse:
    def __init__(self) -> None:
        self.all_obs: list[FakeObs] = []
        self.traces: list[FakeObs] = []
        self.trace_scores: list[tuple] = []   # (trace_id, name, value, comment)
        self.flush_count = 0
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"obs-{self._counter}"

    def trace(self, id=None, name=None, user_id=None, input=None, metadata=None) -> FakeObs:
        # The tracer calls trace() at start (create) and again at end (update by id).
        if id is not None:
            existing = next((t for t in self.traces if t.id == id), None)
            if existing is not None:
                if metadata is not None:
                    existing.fields["metadata"] = metadata
                return existing
        tid = id or self._next_id()
        t = FakeObs(self, "trace", name, tid, tid, user_id=user_id, input=input, metadata=metadata)
        self.traces.append(t)
        return t

    def score(self, trace_id=None, name=None, value=None, comment=None, observation_id=None, **kwargs) -> None:
        self.trace_scores.append((trace_id, name, value, comment))

    def flush(self) -> None:
        self.flush_count += 1

    # ── Query helpers ─────────────────────────────────────────────────────────
    def root(self) -> FakeObs:
        assert len(self.traces) == 1, f"expected one trace, got {len(self.traces)}"
        return self.traces[0]

    def find(self, name: str) -> FakeObs:
        matches = [o for o in self.all_obs if o.name == name and o.kind != "trace"]
        assert len(matches) == 1, f"expected one {name!r} observation, got {len(matches)}"
        return matches[0]


class ExplodingLangfuse(FakeLangfuse):
    """Every SDK entry point raises — proves the tracer's per-call fail-open."""

    def trace(self, *a, **k):
        raise RuntimeError("boom trace")

    def score(self, *a, **k):
        raise RuntimeError("boom score")

    def flush(self):
        raise RuntimeError("boom flush")
