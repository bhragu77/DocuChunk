"""
Phase 8 — Search and RAG endpoints.

POST /search/semantic  — hybrid retrieval + reranking.  NO LLM call.
POST /generate/answer  — retrieval + generation + citation validation +
                         groundedness. Story 6: streams tokens over SSE
                         (?stream=true, default) then sends a verification event;
                         ?stream=false serves the classic synchronous JSON body.
                         Story 7: an answer cache short-circuits the whole chain
                         for a repeated (user, doc, doc_version, query); a cheaper
                         VERIFY_* model tier runs validation + groundedness.

Both endpoints return 503 until Phase 6 (embed_fn) and Phase 7 (bm25) are
wired into app.state in main.py's lifespan.
"""
import logging
import math
import re
import time
from typing import Callable

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.dependencies import get_current_user, get_chroma, get_read_chroma
from app.database import get_db
from app.generation.base import GenerationError
from app.generation.cache import compute_cache_key, normalize_query
from app.generation.citation_parser import is_abstention, parse_citations
from app.generation.citation_validator import validate_citations
from app.generation.context_expander import fetch_doc_chunks
from app.generation.prompt_builder import build_grounded_prompt
from app.generation.query_classifier import (
    AnswerTask,
    QueryType,
    RetrievalScope,
    classify_query,
    classify_scope,
    classify_task,
)
from app.generation.quality_guard import check_verbatim
from app.generation.streaming import sse_event, stream_generation
from app.models.document import Document
from app.models.user import User
from app.observability import (
    adopt,
    capture_text_enabled,
    current_trace_id,
    hash_user_id,
    span,
)
from app.pipeline.retrieval import (
    ScoredChunk,
    fetch_whole_document,
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
    # Chat model picker: "gemini" (accurate cloud) | "offline" (local Ollama LLM).
    # None → the server's default generation seam (backward compatible). An unknown
    # or unconfigured value also falls back to the default rather than erroring.
    model: str | None = Field(default=None)


class Citation(BaseModel):
    chunk_id: str
    source: str
    page_number: int
    text_excerpt: str


class ValidatedCitationDetail(BaseModel):
    """Story 3 — per-citation validation verdict (debugging / admin surface)."""
    index: int
    claim_text: str
    chunk_id: str | None
    verdict: str  # supported | contradicted | not_mentioned | unverified | invalid
    confidence: float


class GenerateAnswerResponse(BaseModel):
    query: str
    # answer is null when generation is not configured (retrieval + citations still real).
    answer: str | None = None
    citations: list[Citation]
    # Story 2/3 — the cited-vs-validated-vs-provided distinction:
    #   cited_sources = chunks the model cited [n] AND that PASSED validation
    #                   (verdict=supported). Story 2 filtered to "cited"; Story 3
    #                   further filters to "cited AND actually supported".
    #   dropped_sources = chunks the model cited but that FAILED validation
    #                   (contradicted / not_mentioned). The [n] markers stay in the
    #                   answer text (see below) so the frontend can flag them.
    #   all_sources   = every chunk fed to the model (transparency / debugging).
    # NOTE: the answer TEXT is returned UNCHANGED — markers for dropped citations
    # are deliberately left in place. The frontend renders them with a warning
    # rather than silently deleting a [n] the user can see.
    cited_sources: list[Citation] = []
    dropped_sources: list[Citation] = []
    validation_details: list[ValidatedCitationDetail] = []
    all_sources: list[Citation] = []
    # abstained=True → the model said the answer isn't in the documents (fuzzy match).
    # The frontend renders this instead of an empty/hallucinated answer.
    abstained: bool = False
    grounded: bool
    confidence: float
    unsupported_claims: list[str]
    # verified=True → the groundedness check actually ran; False → it didn't (fail-closed).
    verified: bool = True
    # The trust-metric BREAKDOWN behind `confidence`, surfaced for the UI's
    # "power-user" expander. Empty {} on the pre-generation error shapes.
    #   retrieval         — how relevant the retrieved evidence is (0-1, from the
    #                        reranker's top score).
    #   citation_coverage — fraction of the model's [n] markers that survived
    #                        validation (cited / (cited+dropped)).
    #   groundedness      — the groundedness verifier's confidence (same source as
    #                        top-level `confidence`).
    #   verified          — mirrors `verified` (did verification actually run).
    #   capped            — True when groundedness confidence outran the weaker
    #                        retrieval/coverage evidence (a "confident but thin" flag).
    confidence_signals: dict = {}
    # Set only on a failure state, e.g. "generation_not_configured" / "verification_unavailable".
    error: str | None = None
    # ── Generation-quality fix (Part 6) — additive fields, no existing field changes ──
    #   query_type      — the heuristic query classification (navigational /
    #                     definitional / analytical / informational) that shaped
    #                     the prompt's response-style hint (Part 3).
    #   verbatim_ratio  — 0-1, fraction of answer sentences that closely copy a
    #                     source (anti-verbatim guard, Part 5). Monitoring only.
    #   quality_warning — "high_verbatim" when verbatim_ratio exceeds the warning
    #                     threshold, else null; lets the frontend optionally flag
    #                     an answer that hugs the source text. null on error shapes.
    query_type: str = QueryType.INFORMATIONAL.value
    verbatim_ratio: float = 0.0
    quality_warning: str | None = None
    # ── Scope-Aware Retrieval Routing — additive, no existing field changes ──
    #   retrieval_scope — "local" (top-k) or "global" (whole document was fed).
    #   answer_task     — the detected output shape (answer / enumerate / table /
    #                     classify / summarize / critique / infer) that shaped the
    #                     TASK INSTRUCTIONS block and the groundedness policy.
    retrieval_scope: str = RetrievalScope.LOCAL.value
    answer_task: str = AnswerTask.ANSWER.value


# ── POST /search/semantic ─────────────────────────────────────────────────────

@search_router.post("/semantic", response_model=SemanticSearchResponse)
def semantic_search(
    payload: SemanticSearchRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    chroma=Depends(get_read_chroma),
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

def _resolve_doc_version(db: Session, user_id: str, doc_id: str | None) -> int | None:
    """Story 7 — look up Document.version for the cache key. None when doc_id is
    absent (a multi-doc query isn't version-scoped, so it isn't cached) or the
    document doesn't belong to this user / doesn't exist."""
    if not doc_id:
        return None
    doc = (
        db.query(Document)
        .filter(Document.id == doc_id, Document.user_id == user_id)
        .first()
    )
    return doc.version if doc is not None else None


def _retrieval_signal(top_chunks: list[ScoredChunk]) -> float:
    """A 0-1 relevance score for the retrieved evidence.

    Prefer the cross-encoder reranker's top score (an unbounded logit → squashed
    with a sigmoid). Fall back to the RRF fused score when no reranker ran. This
    is the "did we even find good source material?" signal, independent of what
    the model then did with it.
    """
    if not top_chunks:
        return 0.0
    rerank_scores = [c.reranker_score for c in top_chunks if c.reranker_score is not None]
    if rerank_scores:
        return round(1.0 / (1.0 + math.exp(-max(rerank_scores))), 3)
    top_fused = max((c.fused_score for c in top_chunks), default=0.0)
    return round(min(1.0, top_fused * 30.0), 3)  # RRF scores are small positives


def _compute_confidence_signals(
    top_chunks: list[ScoredChunk],
    valid_indices,
    dropped_indices,
    groundedness_conf: float,
    verified: bool,
) -> dict:
    """The metric breakdown behind `confidence` — see GenerateAnswerResponse."""
    retrieval = _retrieval_signal(top_chunks)
    total_cited = len(valid_indices) + len(
        [i for i in dropped_indices if 1 <= i <= len(top_chunks)]
    )
    citation_coverage = round(len(valid_indices) / total_cited, 3) if total_cited else 0.0
    evidence_floor = min(retrieval, citation_coverage)
    return {
        "retrieval": retrieval,
        "citation_coverage": citation_coverage,
        "groundedness": round(float(groundedness_conf), 3),
        "verified": bool(verified),
        # "confident but thin": the verifier is more sure than the evidence warrants.
        "capped": bool(verified and groundedness_conf > evidence_floor + 1e-9),
    }


def _build_answer_response(
    query: str,
    top_chunks: list[ScoredChunk],
    citations: list[Citation],
    answer_draft: str,
    verify_fn: Callable[[str], str],
    query_type: str = QueryType.INFORMATIONAL.value,
    retrieval_scope: str = RetrievalScope.LOCAL.value,
    answer_task: AnswerTask = AnswerTask.ANSWER,
) -> GenerateAnswerResponse:
    """
    Stories 3-4 verification over a COMPLETE answer → the final response payload.

    Shared by BOTH request paths so they are schema-identical:
      * stream=false → this object is returned as the JSON body.
      * stream=true  → this object is emitted (model_dump) as the SSE
        'verification' event after the token stream finishes.

    Verification (citation validation + groundedness) requires the whole answer,
    which is exactly why it runs here — after generation, not during it. Story 7:
    it runs through verify_fn (the cheaper VERIFICATION tier), not the generator.

    Generation-quality fix (Part 5/6): after citation validation and BEFORE
    confidence, the anti-verbatim guard measures how much of the answer copies
    the sources and stamps query_type / verbatim_ratio / quality_warning onto the
    response. Monitoring only — the answer text is never modified.
    """
    abstained = is_abstention(answer_draft)

    # Story 3: validate the model's [n] markers. Parse the RAW indices (NO
    # num_chunks filter) so an out-of-range marker like [99] reaches the validator
    # and is recorded as invalid/dropped instead of being silently swallowed here.
    _, raw_cited_indices = parse_citations(answer_draft)
    validation = validate_citations(
        answer_draft,
        raw_cited_indices,
        top_chunks,
        llm_fn=verify_fn,
        use_llm=get_settings().validate_use_llm,
    )

    # Part 5 — anti-verbatim guard, after validation, before confidence. No LLM.
    verbatim = check_verbatim(answer_draft, top_chunks)

    # cited_sources = only the citations that PASSED validation (verdict=supported).
    # dropped_sources = the in-range chunks that were cited but FAILED. (Invalid
    # out-of-range indices have no chunk, so they appear only in validation_details.)
    cited_sources = [citations[i - 1] for i in validation.valid_indices]
    dropped_sources = [
        citations[i - 1]
        for i in validation.dropped_indices
        if 1 <= i <= len(top_chunks)
    ]
    validation_details = [
        ValidatedCitationDetail(
            index=v.index,
            claim_text=v.claim_text,
            chunk_id=v.chunk_id,
            verdict=v.verdict,
            confidence=v.confidence,
        )
        for v in validation.validated_citations
    ]

    if abstained:
        # The model said the answer isn't in the documents. Fact-checking a refusal
        # is meaningless — skip the groundedness LLM call and report zero confidence
        # so the frontend renders the abstention, not a confident-looking answer.
        return GenerateAnswerResponse(
            query=query,
            answer=answer_draft,
            citations=citations,
            cited_sources=cited_sources,
            dropped_sources=dropped_sources,
            validation_details=validation_details,
            all_sources=citations,
            abstained=True,
            grounded=False,
            confidence=0.0,
            unsupported_claims=[],
            verified=True,
            confidence_signals=_compute_confidence_signals(
                top_chunks, validation.valid_indices, validation.dropped_indices,
                groundedness_conf=0.0, verified=True,
            ),
            query_type=query_type,
            verbatim_ratio=verbatim.verbatim_ratio,
            quality_warning=verbatim.quality_warning,
            retrieval_scope=retrieval_scope,
            answer_task=answer_task.value,
        )

    groundedness = groundedness_check(
        query, answer_draft, top_chunks, llm_fn=verify_fn, answer_task=answer_task
    )

    return GenerateAnswerResponse(
        query=query,
        # Answer text is returned UNCHANGED — dropped [n] markers stay in place.
        answer=answer_draft,
        citations=citations,
        cited_sources=cited_sources,
        dropped_sources=dropped_sources,
        validation_details=validation_details,
        all_sources=citations,
        abstained=False,
        grounded=groundedness["grounded"],
        confidence=groundedness["confidence"],
        unsupported_claims=groundedness["unsupported_claims"],
        verified=groundedness.get("verified", True),
        confidence_signals=_compute_confidence_signals(
            top_chunks, validation.valid_indices, validation.dropped_indices,
            groundedness_conf=groundedness["confidence"],
            verified=groundedness.get("verified", True),
        ),
        error=groundedness.get("error"),
        query_type=query_type,
        verbatim_ratio=verbatim.verbatim_ratio,
        quality_warning=verbatim.quality_warning,
        retrieval_scope=retrieval_scope,
        answer_task=answer_task.value,
    )


def _generation_error_response(
    query: str,
    citations: list[Citation],
    error: str,
    query_type: str = QueryType.INFORMATIONAL.value,
) -> GenerateAnswerResponse:
    """Honest error shape for a RUNTIME generation failure (GenerationError:
    timeout, rate limit/429, transport). Per the GenerationProvider contract the
    endpoint returns HTTP 200 with error=... and REAL citations — retrieval
    succeeded, only the LLM call failed — never a 500."""
    return GenerateAnswerResponse(
        query=query,
        answer=None,
        citations=citations,
        cited_sources=[],
        all_sources=citations,
        abstained=False,
        grounded=False,
        confidence=0.0,
        unsupported_claims=[],
        verified=False,
        error=error,
        query_type=query_type,
    )


# ── Observability helpers (Phase 9A) ──────────────────────────────────────────

def _attach_verify_scores(vsp, response: GenerateAnswerResponse) -> None:
    """Attach the verification scores + booleans to the "verify" span, all derived
    from the already-built response (no extra work, no logic change)."""
    signals = response.confidence_signals or {}
    vsp.score("groundedness", float(signals.get("groundedness", 0.0)))
    vsp.score("citation_validity", float(signals.get("citation_coverage", 0.0)))
    vsp.score("confidence", float(response.confidence))
    vsp.update(
        grounded=bool(response.grounded),
        verified=bool(response.verified),
        abstained=bool(response.abstained),
    )


# ── Part 4a — thin-result retry helpers ───────────────────────────────────────

# A "code" token: carries BOTH a letter and a digit (GX-4200, 4000mAh). Dropping
# it is the "drop the most specific term" broadening move.
_CODE_TOKEN_RE = re.compile(r"^(?=[\w-]*[A-Za-z])(?=[\w-]*\d)[\w-]+$")
# Question/glue words stripped when reducing a query to its key noun phrase.
_BROADEN_STOPWORDS = frozenset({
    "what", "is", "are", "the", "a", "an", "how", "do", "does", "did", "i",
    "to", "of", "for", "define", "explain", "describe", "which", "can", "with",
    "on", "in", "and", "or", "about", "tell", "me",
})


def _broaden_query(query: str) -> str:
    """Produce a BROADER version of `query` for the thin-result retry.

    First strip question/glue words down to the key noun phrase. If that phrase
    still holds MORE than one content term, also drop the most specific (code-like)
    token to widen further; a single-term phrase is kept as-is (it's already
    broader than the full question with its framing removed). Falls back to the
    original query if broadening would leave nothing.
    """
    tokens = re.findall(r"[\w-]+", query)
    if not tokens:
        return query

    content = [t for t in tokens if t.lower() not in _BROADEN_STOPWORDS]
    if not content:
        content = tokens  # query was all glue words — nothing to strip

    # With several content terms, drop the most specific (code-like) one to broaden.
    if len(content) > 1:
        code_idx = next(
            (i for i, t in enumerate(content) if _CODE_TOKEN_RE.match(t)), None
        )
        if code_idx is not None:
            content = content[:code_idx] + content[code_idx + 1:]

    broadened = " ".join(content).strip()
    return broadened or query.strip()


def _cached_sse_stream(cached: dict):
    """Story 7 — the cache-hit SSE fast path: one token event carrying the whole
    answer, then the cached verification. No generation, no verification calls."""
    answer = cached.get("answer") or ""

    async def _gen():
        yield sse_event("token", {"text": answer, "done": True, "full_answer": answer})
        yield sse_event("verification", cached)

    return StreamingResponse(_gen(), media_type="text/event-stream")


@generate_router.get("/models")
def list_generation_models(request: Request):
    """Chat model picker discovery: which generation tiers are wired right now.

    The UI calls this to build the picker and to HIDE tiers that aren't configured
    (e.g. "offline" when Ollama is unreachable) rather than letting the user pick a
    model that would silently fall back. `default` is the tier the UI should
    pre-select (gemini preferred, else offline, else none → server default seam)."""
    registry = getattr(request.app.state, "gen_registry", None) or {}
    # Labels name the TRADE-OFF, so the picker reads as a choice: the cloud tier is
    # the accurate one, the Ollama tier is the free one (no API cost, runs locally).
    _labels = {"gemini": "Gemini (accurate)", "offline": "Ollama (free)"}
    models = [
        {"key": key, "model_name": entry["model_name"], "label": _labels.get(key, key)}
        for key, entry in registry.items()
    ]
    default = "gemini" if "gemini" in registry else ("offline" if "offline" in registry else None)
    return {"models": models, "default": default}


@generate_router.post("/answer", response_model=GenerateAnswerResponse)
async def generate_answer_endpoint(
    payload: GenerateAnswerRequest,
    request: Request,
    stream: bool = True,
    current_user: User = Depends(get_current_user),
    chroma=Depends(get_read_chroma),
    bm25=Depends(_get_bm25),
    embed_fn: Callable = Depends(_get_embed_fn),
    db: Session = Depends(get_db),
):
    """
    RAG endpoint: retrieval → LLM generation → citation validation + groundedness.

    Story 6 — two response modes over the SAME logic:
      * ?stream=true (default): a text/event-stream. Phase 1 streams generation
        tokens as 'token' events (status "generating", confidence not yet known);
        Phase 2, AFTER the stream completes, runs validation + groundedness and
        sends a single 'verification' event carrying the full final payload. This
        is a deliberate "show-then-verify": the user reads immediately and gets
        the trust signal a moment later. Verification MUST see the whole answer,
        so it cannot run mid-stream.
      * ?stream=false: the classic synchronous JSON response (retrieval →
        generate → verify → return everything at once). This is the
        backward-compatible path used by unit tests.

    Story 7 — cost/latency levers over that same logic:
      * ANSWER CACHE: a repeated (user, doc, doc_version, query) short-circuits
        the ENTIRE chain (checked before any work), returning the stored payload
        in <10ms. The key embeds doc_version, so a re-upload busts it automatically.
      * MODEL TIERING: generation runs on the strong llm_fn; validation +
        groundedness run on the cheaper verify_fn (VERIFY_* tier). verify_fn
        falls back to llm_fn when no separate tier is configured.

    The global STREAM_ENABLED kill switch forces non-streaming for every request
    regardless of ?stream. Retrieval always runs fully BEFORE any streaming
    starts (it's fast); only generation is streamed; verification runs after.
    """
    settings = get_settings()
    # STREAM_ENABLED is the global kill switch: false → serve JSON to everyone,
    # even ?stream=true. Otherwise honor the per-request ?stream flag.
    use_stream = stream and settings.stream_enabled

    # Part 3 — classify the query up front (pure heuristic, <1ms). Drives the
    # prompt's response-style hint and is echoed back in the response.
    query_type = classify_query(payload.query)

    # Scope-Aware Retrieval Routing — a second, orthogonal axis (pure heuristic).
    # answer_task = the required OUTPUT SHAPE; scope = LOCAL (top-k) vs GLOBAL
    # (whole document). A GLOBAL query with a single doc_id feeds the FULL document
    # instead of the top-k slice, so "list all / table / classify / summarize /
    # critique / infer" questions see every chunk. Gated by SCOPE_ROUTING_ENABLED;
    # off → every query stays LOCAL (pure pre-existing top-k behavior).
    answer_task = classify_task(payload.query)
    scope = (
        classify_scope(payload.query, answer_task)
        if getattr(settings, "scope_routing_enabled", True)
        else RetrievalScope.LOCAL
    )

    # Story 1/6 seams; Story 7 adds verify_fn (cheap tier) and the answer cache.
    llm_fn: Callable[[str], str] | None = getattr(request.app.state, "llm_fn", None)
    # verify_fn falls back to llm_fn when no separate verification tier is wired
    # (VERIFY_PROVIDER unset) — backward compatible single-model behavior.
    verify_fn: Callable[[str], str] | None = getattr(request.app.state, "verify_fn", None) or llm_fn
    cache = getattr(request.app.state, "answer_cache", None)

    # ── Chat model picker ───────────────────────────────────────────────────────
    # The per-request `model` ("gemini" | "offline") selects the GENERATION backend
    # from the startup registry. Retrieval, reranking, prompt assembly and
    # verification are UNCHANGED — only the answer-generating model swaps. An unknown
    # or unconfigured selection falls back to the default seam (llm_fn / llm_stream_fn
    # from app.state). selected_model is threaded into the cache key so the two tiers
    # never serve each other's cached answers.
    registry = getattr(request.app.state, "gen_registry", None) or {}
    selected_model = payload.model if payload.model in registry else None
    if selected_model:
        _entry = registry[selected_model]
        llm_fn = _entry["llm_fn"]
        active_stream_fn = _entry["llm_stream_fn"]
        active_gen_model_name = _entry["model_name"]
    else:
        active_stream_fn = getattr(request.app.state, "llm_stream_fn", None)
        active_gen_model_name = getattr(request.app.state, "gen_model_name", None)

    # ── Story 7 CACHE CHECK — before any retrieval / generation / verification ──
    # Only single-document, version-resolvable queries are cached (multi-doc
    # queries have no single doc_version to key on / bust with).
    cache_key = None
    if cache is not None:
        doc_version = _resolve_doc_version(db, current_user.id, payload.doc_id)
        if payload.doc_id and doc_version is not None:
            cache_key = compute_cache_key(
                current_user.id, payload.doc_id, doc_version, payload.query,
                model=selected_model,
            )
            cached = cache.get(cache_key)
            if cached is not None:
                logger.info(
                    "cache_hit query=%s doc=%s",
                    normalize_query(payload.query)[:50], payload.doc_id,
                )
                if use_stream:
                    return _cached_sse_stream(cached)
                return GenerateAnswerResponse.model_validate(cached)

    # ── Observability root span (Phase 9A) ─────────────────────────────────────
    # Everything from retrieval through verification hangs off "rag.request". The
    # root is driven manually (not a `with`) because the streaming path returns a
    # StreamingResponse whose generator runs AFTER this coroutine returns — the
    # generator closes the root in its own finally. Fail-open: get_tracer() is a
    # no-op unless tracing is enabled, so this is byte-identical to before then.
    root_ctx = span(
        "rag.request",
        user=hash_user_id(current_user.id),
        doc_id=payload.doc_id,
        top_k=payload.top_k,
        stream=use_stream,
        query_type=query_type.value,
        retrieval_scope=scope.value,
        answer_task=answer_task.value,
        **({"query": payload.query} if capture_text_enabled() else {}),
    )
    root = root_ctx.__enter__()
    # Captured while the root scope is active, for the streaming generator (which
    # runs in a copied context and re-establishes the root as parent via adopt()).
    root_trace_id = current_trace_id()

    # ── GLOBAL scope — feed the WHOLE document, bypass top-k ────────────────────
    # For a whole-document question with a single doc in scope, top-k retrieval
    # drops the chunks a complete answer needs. Fetch every chunk in reading order
    # and skip hybrid/rerank/thin-retry entirely. If the fetch is empty (doc not
    # found / Chroma error) fall through to the normal LOCAL path.
    top_chunks: list[ScoredChunk] = []
    used_whole_doc = False
    if scope == RetrievalScope.GLOBAL and payload.doc_id:
        with span("retrieve", k=payload.top_k, scope="global") as rsp:
            whole = fetch_whole_document(chroma, current_user.id, payload.doc_id)
            max_chunks = getattr(settings, "scope_whole_doc_max_chunks", 0)
            if max_chunks and len(whole) > max_chunks:
                whole = whole[:max_chunks]  # safety rail for very large docs
            if whole:
                top_chunks = whole
                used_whole_doc = True
            rsp.update(
                candidate_count=len(whole),
                returned_chunk_ids=[c.chunk_id for c in top_chunks][:200],
            )

    if not used_whole_doc:
        # ── LOCAL scope — the existing top-k hybrid + rerank path (unchanged) ────
        with span("retrieve", k=payload.top_k, scope="local") as rsp:
            candidates = hybrid_search(
                query=payload.query,
                user_id=current_user.id,
                chroma_client=chroma,
                bm25_index=bm25,
                embed_fn=embed_fn,
                doc_id=payload.doc_id,
                top_n=50,
            )
            rsp.update(
                candidate_count=len(candidates),
                returned_chunk_ids=[c.chunk_id for c in candidates][:200],
            )

        if not candidates:
            root_ctx.__exit__(None, None, None)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No relevant content found. Upload documents and ensure they are processed.",
            )

        top_chunks = rerank(payload.query, candidates, top_k=payload.top_k)

        # ── Part 4a — thin-result retry ─────────────────────────────────────────
        # When rerank returns very few chunks for a DEFINITIONAL/INFORMATIONAL query,
        # the model has little to synthesize from and either parrots or abstains.
        # Re-run retrieval ONCE with a broadened query and append the deduped hits.
        # Bounded to a single extra retrieval — never a loop. Gated by THIN_RESULT_RETRY.
        thin_retry = getattr(settings, "thin_result_retry", True)
        thin_threshold = getattr(settings, "thin_result_threshold", 3)
        if (
            thin_retry
            and len(top_chunks) < thin_threshold
            and query_type in (QueryType.DEFINITIONAL, QueryType.INFORMATIONAL)
        ):
            broadened = _broaden_query(payload.query)
            if broadened and broadened.lower() != payload.query.strip().lower():
                extra = hybrid_search(
                    query=broadened,
                    user_id=current_user.id,
                    chroma_client=chroma,
                    bm25_index=bm25,
                    embed_fn=embed_fn,
                    doc_id=payload.doc_id,
                    top_n=50,
                )
                if extra:
                    extra_ranked = rerank(broadened, extra, top_k=payload.top_k)
                    seen = {c.chunk_id for c in top_chunks}
                    for c in extra_ranked:
                        if c.chunk_id not in seen:
                            top_chunks.append(c)
                            seen.add(c.chunk_id)

    # Part 4b — if STILL sparse (LOCAL only; whole-doc is never "thin"), tell the
    # prompt so the model answers from thin-but-present evidence rather than abstaining.
    thin_context = (not used_whole_doc) and len(top_chunks) < 2

    # Citations come from RETRIEVAL, so they are real even when generation is down.
    citations = [
        Citation(
            chunk_id=c.chunk_id,
            source=c.source,
            page_number=c.page_number,
            text_excerpt=c.text[:200],
        )
        for c in top_chunks
    ]

    # Generation backend is wired onto app.state.llm_fn by Story 1's provider seam.
    # Until then (or on any misconfiguration) it is None: return an HONEST error with
    # HTTP 200 — retrieval succeeded and the citations are real; only generation failed.
    # (No 500: the request itself is fine.) The old code silently returned answer="".
    # This is returned as JSON even for stream=true: there is nothing to stream, and
    # it is NEVER cached (an error is not a valid cached answer).
    if llm_fn is None:
        root.update(error="generation_not_configured")
        root_ctx.__exit__(None, None, None)
        return GenerateAnswerResponse(
            query=payload.query,
            answer=None,
            citations=citations,
            # No answer means no model-chosen citations; all_sources stays real.
            cited_sources=[],
            all_sources=citations,
            abstained=False,
            grounded=False,
            confidence=0.0,
            unsupported_claims=[],
            verified=False,
            error="generation_not_configured",
            query_type=query_type.value,
            retrieval_scope=scope.value,
            answer_task=answer_task.value,
        )

    # Story 5: expansion fetches every chunk of a cited document from Chroma so the
    # prompt builder can widen each chunk to its neighbors. Bound to this user's
    # collection; the builder caches per doc_id so it's one query per document.
    # NOTE: whole-doc GLOBAL retrieval already holds every chunk, so neighbor
    # expansion would re-fetch and duplicate text — disable it in that case.
    doc_chunks_fetcher = (
        None if used_whole_doc
        else (lambda doc_id: fetch_doc_chunks(chroma, current_user.id, doc_id))
    )

    def _maybe_cache(response: GenerateAnswerResponse) -> None:
        """Store the completed response under the version-aware key (Story 7)."""
        if cache_key is not None:
            cache.set(cache_key, response.model_dump(), ttl=settings.cache_ttl)
            logger.info(
                "cache_miss query=%s doc=%s ttl=%d",
                normalize_query(payload.query)[:50], payload.doc_id, settings.cache_ttl,
            )

    if not use_stream:
        # ── Synchronous path (backward compatible) ──────────────────────────
        # generate_answer emits the prompt_build + generate child spans internally
        # (so they sit as siblings under rag.request); we pass the model name in for
        # the generate span's label. NOTE: token counts / ttft / finish_reason must
        # come from the provider usage field, which the str-returning llm_fn seam
        # does not expose — Phase 9B records them; a tiktoken estimate is NOT used.
        try:
            answer_draft = generate_answer(
                payload.query, top_chunks, llm_fn=llm_fn,
                doc_chunks_fetcher=doc_chunks_fetcher,
                query_type=query_type, thin_context=thin_context,
                answer_task=answer_task,
                model_name=active_gen_model_name,
            )
        except GenerationError as exc:
            # Runtime LLM failure (e.g. Gemini 429 / timeout) → honest error, not 500.
            logger.warning("Generation failed: %s", exc)
            root.update(error="generation_failed")
            root_ctx.__exit__(None, None, None)
            return _generation_error_response(
                payload.query, citations, "generation_failed", query_type.value
            )
        with span("verify") as vsp:
            response = _build_answer_response(
                payload.query, top_chunks, citations, answer_draft, verify_fn,
                query_type.value, scope.value, answer_task,
            )
            _attach_verify_scores(vsp, response)
        root.update(abstained=bool(response.abstained))
        _maybe_cache(response)
        root_ctx.__exit__(None, None, None)
        return response

    # ── Streaming path (SSE, two-phase) ─────────────────────────────────────
    # Build the prompt once, up front — retrieval + prompt assembly finish before
    # a single token is streamed. The streaming seam falls back to llm_fn if the
    # provider has no real streaming or fails on the first chunk.
    with span("prompt_build") as psp:
        prompt = build_grounded_prompt(
            payload.query, top_chunks, doc_chunks_fetcher=doc_chunks_fetcher,
            query_type=query_type, thin_context=thin_context, answer_task=answer_task,
        )
        psp.update(source_count=len(top_chunks))
        if capture_text_enabled():
            psp.update(prompt=prompt)
    stream_fn: Callable[[str], object] | None = active_stream_fn
    gen_model_name = active_gen_model_name

    # Close the root's contextvar scope HERE (this coroutine's context), so its
    # tokens reset cleanly. The generator runs in a copied context and re-adopts the
    # root span object as parent for generate/verify. The root span stays a valid
    # target for update()/child links after this — closing it only ends its scope.
    root_ctx.__exit__(None, None, None)

    async def event_generator():
        with adopt(root, root_trace_id):
            # PHASE 1 — stream the generation tokens.
            full_answer = ""
            try:
                started = time.perf_counter()
                first = True
                with span("generate", model=gen_model_name) as gsp:
                    for token in stream_generation(prompt, stream_fn, llm_fn):
                        if first:
                            first = False
                            gsp.update(ttft_ms=round((time.perf_counter() - started) * 1000, 1))
                        full_answer += token
                        yield sse_event("token", {"text": token, "done": False})
            except GenerationError as exc:
                # Runtime LLM failure (e.g. Gemini 429) → close the stream honestly:
                # a terminal token then an error verification event (never a 500).
                logger.warning("Streaming generation failed: %s", exc)
                root.update(error="generation_failed")
                yield sse_event("token", {"text": "", "done": True, "full_answer": ""})
                err = _generation_error_response(
                    payload.query, citations, "generation_failed", query_type.value
                )
                yield sse_event("verification", err.model_dump())
                return
            # The terminal token carries the authoritative assembled answer so the
            # client never has to trust its own concatenation.
            full_answer = full_answer.strip()
            yield sse_event("token", {"text": "", "done": True, "full_answer": full_answer})

            # PHASE 2 — verify AFTER the stream (validation + groundedness need the
            # WHOLE answer). Emitted as one 'verification' event whose shape is
            # identical to the stream=false JSON body, then cached.
            with span("verify") as vsp:
                response = _build_answer_response(
                    payload.query, top_chunks, citations, full_answer, verify_fn,
                    query_type.value, scope.value, answer_task,
                )
                _attach_verify_scores(vsp, response)
            root.update(abstained=bool(response.abstained))
            _maybe_cache(response)
            yield sse_event("verification", response.model_dump())

    return StreamingResponse(event_generator(), media_type="text/event-stream")
