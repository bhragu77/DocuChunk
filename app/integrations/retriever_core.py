"""
The retrieval call shared by every framework adapter.

WHY A SHARED CORE
=================
The point of the framework comparison is to isolate SYNTHESIS. If the LlamaIndex
adapter and the LangChain adapter each assembled retrieval slightly differently —
a different candidate pool, a different rerank cutoff — then any quality delta in
the final answer could be retrieval noise rather than a property of the framework.

Both adapters therefore call `retrieve()` below, and so does the benchmark's native
path. Retrieval is byte-identical across all three; only what happens to the chunks
afterwards differs. That is what makes the comparison mean anything.

No framework is imported here. This module is plain Python and is safe to import
with none of the integration extras installed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from app.pipeline.retrieval import ScoredChunk, hybrid_search, rerank

# Same ratio the search router uses: retrieve a wider pool than needed so the
# cross-encoder has room to reorder, capped so latency stays bounded.
CANDIDATE_MULTIPLIER = 6
CANDIDATE_CAP = 50


@dataclass
class RetrievalDeps:
    """Everything retrieval needs, gathered once.

    Bundled rather than passed as five loose arguments because the adapters are
    constructed by framework code that will not know about our dependency-injection
    conventions — this keeps the surface a single object.
    """
    user_id: str
    vector_client: object          # Chroma-compatible client (chroma/pgvector/pinecone)
    bm25_index: object
    embed_fn: Callable[[str], list[float]]
    doc_id: str | None = None


def retrieve(query: str, deps: RetrievalDeps, top_k: int = 5) -> list[ScoredChunk]:
    """Hybrid dense+BM25 retrieval, RRF-fused, cross-encoder reranked to top_k."""
    candidate_n = min(top_k * CANDIDATE_MULTIPLIER, CANDIDATE_CAP)
    candidates = hybrid_search(
        query=query,
        user_id=deps.user_id,
        chroma_client=deps.vector_client,
        bm25_index=deps.bm25_index,
        embed_fn=deps.embed_fn,
        doc_id=deps.doc_id,
        top_n=candidate_n,
    )
    if not candidates:
        return []
    return rerank(query, candidates, top_k=top_k)


def chunk_to_payload(c: ScoredChunk) -> tuple[str, dict]:
    """Flatten a ScoredChunk into (text, metadata) for a framework's document type.

    The per-stage scores are carried through deliberately. A framework that discards
    them still works, but keeping them means a debugging session inside LlamaIndex or
    LangChain can still see WHY a chunk ranked where it did — which is the main thing
    lost when retrieval disappears behind a framework abstraction.
    """
    meta = {
        "chunk_id": c.chunk_id,
        "doc_id": c.doc_id,
        "source": c.source,
        "page_number": c.page_number,
        "fused_score": c.fused_score,
        "dense_rank": c.dense_rank,
        "bm25_rank": c.bm25_rank,
        "reranker_score": c.reranker_score,
    }
    meta.update(c.metadata or {})
    return c.text, meta


def relevance_score(c: ScoredChunk) -> float:
    """Single score for frameworks that expect one number per result.

    Prefers the cross-encoder score, which is the last and most accurate signal.
    Falls back to the RRF fused score when reranking was skipped.
    """
    return float(c.reranker_score if c.reranker_score is not None else c.fused_score)
