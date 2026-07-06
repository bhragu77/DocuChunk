"""
Phase 8 — Search and RAG endpoints.

POST /search/semantic  — hybrid retrieval + reranking.  NO LLM call.
POST /generate/answer  — retrieval + generation + groundedness check.

Both endpoints return 503 until Phase 6 (embed_fn) and Phase 7 (bm25) are
wired into app.state in main.py's lifespan.
"""
import logging
from typing import Callable

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.core.dependencies import get_current_user, get_chroma
from app.models.user import User
from app.pipeline.retrieval import (
    ScoredChunk,
    generate_answer,
    groundedness_check,
    hybrid_search,
    rerank,
)

logger = logging.getLogger(__name__)

search_router = APIRouter(prefix="/search", tags=["search"])
generate_router = APIRouter(prefix="/generate", tags=["generate"])


# ── FastAPI state dependencies ────────────────────────────────────────────────

def _get_bm25(request: Request):
    """Phase 7 dependency — raises 503 until vector_store.py is built."""
    bm25 = getattr(request.app.state, "bm25", None)
    if bm25 is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="BM25 index not available — complete Phase 7 (vector_store.py) first.",
        )
    return bm25


def _get_embed_fn(request: Request) -> Callable[[str], list[float]]:
    """Phase 6 dependency — raises 503 until embedder.py is built."""
    fn = getattr(request.app.state, "embed_fn", None)
    if fn is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding function not available — complete Phase 6 (embedder.py) first.",
        )
    return fn


# ── Request / response schemas ────────────────────────────────────────────────

class SemanticSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=512)
    doc_id: str | None = None
    top_k: int = Field(default=8, ge=1, le=50)


class ChunkResult(BaseModel):
    chunk_id: str
    text: str
    doc_id: str
    source: str
    page_number: int
    dense_rank: int | None
    bm25_rank: int | None
    fused_score: float
    reranker_score: float | None


class SemanticSearchResponse(BaseModel):
    query: str
    results: list[ChunkResult]
    total: int


class GenerateAnswerRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=512)
    doc_id: str | None = None
    top_k: int = Field(default=8, ge=1, le=50)


class Citation(BaseModel):
    chunk_id: str
    source: str
    page_number: int
    text_excerpt: str


class GenerateAnswerResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    grounded: bool
    confidence: float
    unsupported_claims: list[str]


# ── POST /search/semantic ─────────────────────────────────────────────────────

@search_router.post("/semantic", response_model=SemanticSearchResponse)
def semantic_search(
    payload: SemanticSearchRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    chroma=Depends(get_chroma),
    bm25=Depends(_get_bm25),
    embed_fn: Callable = Depends(_get_embed_fn),
):
    """
    Hybrid semantic + keyword search.  NO LLM call — pure retrieval.

    Retrieve 6× more candidates than needed, then rerank to top_k.
    Returns reranked chunks with per-stage scores and source citations.
    Fast and cheap: no generation step.
    """
    # Fetch a larger candidate pool so the reranker has room to work
    candidate_n = min(payload.top_k * 6, 50)

    candidates = hybrid_search(
        query=payload.query,
        user_id=current_user.id,
        chroma_client=chroma,
        bm25_index=bm25,
        embed_fn=embed_fn,
        doc_id=payload.doc_id,
        top_n=candidate_n,
    )

    if not candidates:
        return SemanticSearchResponse(query=payload.query, results=[], total=0)

    ranked = rerank(payload.query, candidates, top_k=payload.top_k)

    return SemanticSearchResponse(
        query=payload.query,
        results=[
            ChunkResult(
                chunk_id=c.chunk_id,
                text=c.text,
                doc_id=c.doc_id,
                source=c.source,
                page_number=c.page_number,
                dense_rank=c.dense_rank,
                bm25_rank=c.bm25_rank,
                fused_score=c.fused_score,
                reranker_score=c.reranker_score,
            )
            for c in ranked
        ],
        total=len(ranked),
    )


# ── POST /generate/answer ─────────────────────────────────────────────────────

@generate_router.post("/answer", response_model=GenerateAnswerResponse)
def generate_answer_endpoint(
    payload: GenerateAnswerRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    chroma=Depends(get_chroma),
    bm25=Depends(_get_bm25),
    embed_fn: Callable = Depends(_get_embed_fn),
):
    """
    RAG endpoint: retrieval → LLM generation → groundedness check.

    The groundedness check is a second, independent LLM call that verifies
    every claim in the draft answer against the source chunks.  If claims
    cannot be verified, grounded=False and the specific unsupported claims
    are returned — the caller decides whether to surface a warning to the user.
    """
    candidates = hybrid_search(
        query=payload.query,
        user_id=current_user.id,
        chroma_client=chroma,
        bm25_index=bm25,
        embed_fn=embed_fn,
        doc_id=payload.doc_id,
        top_n=50,
    )

    if not candidates:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No relevant content found. Upload documents and ensure they are processed.",
        )

    top_chunks = rerank(payload.query, candidates, top_k=payload.top_k)

    answer_draft = generate_answer(payload.query, top_chunks)

    groundedness = groundedness_check(payload.query, answer_draft, top_chunks)

    return GenerateAnswerResponse(
        query=payload.query,
        answer=answer_draft,
        citations=[
            Citation(
                chunk_id=c.chunk_id,
                source=c.source,
                page_number=c.page_number,
                text_excerpt=c.text[:200],
            )
            for c in top_chunks
        ],
        grounded=groundedness["grounded"],
        confidence=groundedness["confidence"],
        unsupported_claims=groundedness["unsupported_claims"],
    )
