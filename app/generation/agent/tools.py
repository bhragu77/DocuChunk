"""
Agent tool registry — thin, user-scoped wrappers over the EXISTING retrieval
functions. The agent never gets new capabilities here; it gets a small, well
described, model-choosable surface over what the RAG pipeline already does.

Two design rules from the plan:

  1. REUSE, don't reinvent. `retrieve_docs` = hybrid_search + rerank (exactly the
     LOCAL path of /generate/answer). `fetch_document` = fetch_whole_document (the
     GLOBAL path). No retrieval logic is duplicated.

  2. NORMALIZE the output. Every tool returns a ToolResult whose `.data` is a
     structured {"status", "source", ...} dict (stable for a future native
     function-calling provider and for the SSE 'step' payload), `.observation` is
     the compact evidence text the planner reads next step, and `.chunks` is the
     raw ScoredChunks the endpoint accumulates for real citations.

USER SCOPING: build_tools() closes each tool over (user_id, chroma, bm25,
embed_fn). Every call is therefore pinned to one tenant's corpus exactly like
hybrid_search(user_id=...) — the loop can never construct a cross-user query.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

from app.observability import span
from app.pipeline.retrieval import (
    ScoredChunk,
    fetch_whole_document,
    hybrid_search,
    rerank,
)

logger = logging.getLogger(__name__)

# How much of each chunk the planner sees in an observation. Enough to judge
# relevance/coverage without flooding the planner prompt with full chunk text.
_SNIPPET_CHARS = 180
# Cap the planner-visible snippet list per retrieval so a large top_k / whole-doc
# fetch can't blow up the planner prompt. The full set still reaches `chunks`.
_MAX_SNIPPETS = 8
# Upper bound on a tool-requested top_k (the planner is not trusted to size pools).
_MAX_TOP_K = 20


@dataclass
class ToolResult:
    """A tool's outcome. `data` is model-facing; `chunks` feed real citations."""
    status: str                                  # "success" | "no_results"
    source: str                                  # provenance tag, e.g. "hybrid_search"
    summary: str                                 # one-line trace/SSE label
    observation: str                             # compact evidence for the planner
    chunks: list[ScoredChunk] = field(default_factory=list)
    data: dict = field(default_factory=dict)     # normalized structured payload


@dataclass
class Tool:
    """A model-choosable tool: name + description + JSON-schema + callable."""
    name: str
    description: str
    parameters: dict                             # JSON schema for `args`
    func: Callable[..., ToolResult]

    def to_schema(self) -> dict:
        """OpenAI-style function schema (also drives the ReAct planner prompt)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


def _snippet(chunk: ScoredChunk) -> str:
    text = (chunk.text or "").strip().replace("\n", " ")
    if len(text) > _SNIPPET_CHARS:
        text = text[:_SNIPPET_CHARS].rstrip() + "…"
    loc = chunk.source or chunk.doc_id or "?"
    page = f" p.{chunk.page_number}" if chunk.page_number else ""
    return f"[{loc}{page}] {text}"


def _render_observation(header: str, chunks: list[ScoredChunk]) -> str:
    """Compact numbered evidence block the planner reads on its next step."""
    if not chunks:
        return f"{header}\n  (no chunks found)"
    lines = [f"  - {_snippet(c)}" for c in chunks[:_MAX_SNIPPETS]]
    if len(chunks) > _MAX_SNIPPETS:
        lines.append(f"  … and {len(chunks) - _MAX_SNIPPETS} more")
    return header + "\n" + "\n".join(lines)


def build_tools(
    user_id: str,
    chroma,
    bm25,
    embed_fn: Callable[[str], list[float]],
    *,
    doc_id: str | None = None,
    available_docs: list[dict] | None = None,
    reranker=None,
) -> dict[str, Tool]:
    """Construct the per-request tool registry, each tool pinned to `user_id`.

    `doc_id` (optional) scopes retrieve_docs to a single document, mirroring the
    /generate/answer `doc_id` filter; None searches the whole corpus.

    `available_docs` (optional) is the user's real documents [{doc_id, name}]. It
    lets fetch_document RESOLVE a valid target when the planner supplies an unknown
    or invented id (the fix for empty whole-doc fetches → wasteful RAG fallback).

    `reranker` (optional) is forwarded to retrieval.rerank — None (production) uses
    the real cross-encoder; the eval harness injects a lexical stand-in so an
    offline comparison never loads torch (mirrors gen_harness's LexicalReranker).
    """
    scoped_doc_id = doc_id
    known = list(available_docs or [])
    known_ids = {d.get("doc_id") for d in known}

    def retrieve_docs(query: str, top_k: int = 5) -> ToolResult:
        """hybrid_search + rerank — the LOCAL top-k path, as a tool."""
        k = max(1, min(int(top_k or 5), _MAX_TOP_K))
        with span("tool.retrieve_docs", k=k) as sp:
            candidates = hybrid_search(
                query=query,
                user_id=user_id,
                chroma_client=chroma,
                bm25_index=bm25,
                embed_fn=embed_fn,
                doc_id=doc_id,
                top_n=min(k * 6, 50),
            )
            ranked = (
                rerank(query, candidates, top_k=k, reranker=reranker)
                if candidates else []
            )
            sp.update(candidate_count=len(candidates), returned=len(ranked))
        status = "success" if ranked else "no_results"
        data = {
            "status": status,
            "source": "hybrid_search",
            "query": query,
            "count": len(ranked),
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "source": c.source,
                    "page_number": c.page_number,
                    "score": c.reranker_score,
                }
                for c in ranked
            ],
        }
        return ToolResult(
            status=status,
            source="hybrid_search",
            summary=f'retrieve_docs("{query}") → {len(ranked)} chunk(s)',
            observation=_render_observation(
                f'retrieve_docs("{query}") → {len(ranked)} chunk(s):', ranked
            ),
            chunks=ranked,
            data=data,
        )

    def _resolve_doc_id(requested: str | None) -> tuple[str | None, str | None]:
        """Map the planner's requested doc_id to a REAL one. Returns (target, note).

        The planner often invents a placeholder id (e.g. "document_0") because it
        can't know real ids. So: a known id is used as-is; otherwise fall back to
        the scoped doc_id, else the sole document when there's exactly one, else
        None with a note listing the valid ids so the planner can retry."""
        if requested and requested in known_ids:
            return requested, None
        if scoped_doc_id:
            return scoped_doc_id, None
        if len(known) == 1:
            return known[0].get("doc_id"), None
        if known_ids:
            valid = ", ".join(str(i) for i in list(known_ids)[:10])
            return (requested or None), f"unknown doc_id — valid ids: {valid}"
        return requested, None

    def fetch_document(doc_id: str | None = None) -> ToolResult:  # noqa: shadow ok
        """fetch_whole_document — the GLOBAL whole-document path, as a tool.

        Resolves the requested id to a real document (see _resolve_doc_id) so an
        invented/placeholder id no longer silently returns 0 chunks."""
        target, note = _resolve_doc_id(doc_id)
        with span("tool.fetch_document") as sp:
            chunks = fetch_whole_document(chroma, user_id, target) if target else []
            sp.update(returned=len(chunks), resolved=bool(target and target != doc_id))
        status = "success" if chunks else "no_results"
        label = target or doc_id or "?"
        header = f"fetch_document({label}) → {len(chunks)} chunk(s)"
        if note and not chunks:
            header += f" — {note}"
        data = {
            "status": status,
            "source": "whole_document",
            "doc_id": target,
            "requested_doc_id": doc_id,
            "count": len(chunks),
        }
        return ToolResult(
            status=status,
            source="whole_document",
            summary=header,
            observation=_render_observation(header + ":", chunks),
            chunks=chunks,
            data=data,
        )

    tools: dict[str, Tool] = {
        "retrieve_docs": Tool(
            name="retrieve_docs",
            description=(
                "Search the user's document corpus for passages relevant to a "
                "natural-language query, e.g. 'SolarMax 5000 warranty duration'. "
                "Use one focused query per distinct fact you need; call again with "
                "a different query to gather another part of a multi-part question."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A clear, specific search query.",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "How many passages to return (1-20).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
            func=retrieve_docs,
        ),
        "fetch_document": Tool(
            name="fetch_document",
            description=(
                "Load the FULL text of one document in reading order. Use for "
                "whole-document tasks (summarize, list every item, reconstruct "
                "structure) where top-k search would miss chunks. Pass an EXACT "
                "doc_id from AVAILABLE DOCUMENTS; if there is only one document you "
                "may omit doc_id and it will be used automatically."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "An exact doc_id from AVAILABLE DOCUMENTS (optional when only one document exists).",
                    },
                },
                "required": [],
            },
            func=fetch_document,
        ),
    }
    return tools
