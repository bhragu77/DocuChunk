"""
Agent state + step records — the shared object threaded through the ReAct loop.

DESIGN (retrieval-planning agent, not answer-writing agent)
===========================================================
The agent's job is to GATHER the right chunks, possibly over several retrieval
hops, and then hand them to the EXISTING grounded-generation path
(retrieval.generate_answer + build_grounded_prompt + citation/groundedness
machinery). It deliberately does NOT ask the small planning model to write the
final prose or invent [n] citations — that keeps citation integrity intact and
reuses every trust signal /generate/answer already produces.

So AgentState is an evidence accumulator:
  * collected  — the deduped union of every ScoredChunk any tool returned, in the
                 order first seen. This is what the final generate_answer() runs on.
  * history    — one StepRecord per executed step (for SSE 'step' events + traces).

Nothing here calls an LLM or touches Chroma; it is a plain data container so it
stays trivially testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.pipeline.retrieval import ScoredChunk


@dataclass
class StepRecord:
    """One executed agent step — a tool call and its outcome.

    `summary` is a single human-readable line (SSE 'step' event + trace label).
    `observation` is the compact evidence block fed back to the planner on the
    next step (may be several lines); it is NOT sent to the client verbatim.
    `data` is the normalized, model-facing result dict ({status, source, ...}).
    """
    n: int
    tool: str
    args: dict
    status: str                     # "ok" | "error"
    summary: str
    observation: str = ""
    data: dict = field(default_factory=dict)
    error: str | None = None

    def to_event(self) -> dict:
        """Client-facing payload for the SSE 'step' event (no bulky text)."""
        return {
            "n": self.n,
            "tool": self.tool,
            "args": self.args,
            "status": self.status,
            "summary": self.summary,
            "error": self.error,
        }


@dataclass
class AgentState:
    """Evidence + bookkeeping threaded through the loop. Mutated in place."""
    question: str
    user_id: str
    doc_id: str | None = None
    top_k: int = 8
    # The user's available documents [{doc_id, name}], shown to the planner so it
    # calls fetch_document with a REAL id instead of inventing one. Empty = unknown.
    available_docs: list[dict] = field(default_factory=list)
    steps: int = 0
    history: list[StepRecord] = field(default_factory=list)
    collected: list[ScoredChunk] = field(default_factory=list)
    final_answer: str | None = None
    _seen_chunk_ids: set[str] = field(default_factory=set)

    def add_chunks(self, chunks: list[ScoredChunk]) -> int:
        """Append chunks not already collected (dedup by chunk_id, first-seen order).

        Returns the number of NEW chunks added — 0 means the step surfaced only
        material already gathered (a signal the planner may be going in circles).
        """
        added = 0
        for c in chunks:
            if c.chunk_id in self._seen_chunk_ids:
                continue
            self._seen_chunk_ids.add(c.chunk_id)
            self.collected.append(c)
            added += 1
        return added
