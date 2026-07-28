"""
Phase 8 — Hybrid Retrieval Pipeline

Solves entity ambiguity (e.g. "Apple" fruit vs Apple Inc.) through three
stacked mechanisms — NOT through chunking tweaks:

  1. Hybrid search  — dense (ChromaDB cosine) + lexical (BM25), fused via
                      Reciprocal Rank Fusion.  BM25 rewards exact entity-name
                      matches that dense embeddings blur; RRF lifts chunks
                      that score well in both lists.

  2. Cross-encoder reranking  — joint (query, chunk) scoring catches
                      topically-similar-but-wrong-entity false positives that
                      survive hybrid retrieval.

  3. Groundedness check  — after LLM generation, a SEPARATE LLM call verifies
                      every claim in the draft answer is supported by the
                      retrieved source chunks.  Unsupported claims are surfaced
                      in the API response; no silent hallucination pass-through.

Dependencies (set on app.state by Phase 6/7 startup code in main.py):
  app.state.embed_fn  — Callable[[str], list[float]]  (Phase 6)
  app.state.bm25      — BM25IndexProtocol             (Phase 7)

Phase 4 note:
  tag_entities() is exported here.  The Phase 4 chunker SHOULD call it and
  store the result in chunk.metadata["entities"] so retrieval can use entity
  signals.  Retrieval still works without it; entity tags are advisory.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Callable

from app.generation.prompt_builder import build_grounded_prompt
from app.observability import capture_text_enabled, span

logger = logging.getLogger(__name__)

# Standard RRF constant.  k=60 is the value from the original RRF paper
# (Cormack et al. 2009) and remains the community default.
RRF_K: int = 60

_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_SPACY_ENTITY_LABELS = frozenset({"ORG", "PERSON", "GPE", "PRODUCT", "NORP"})
_GROUNDEDNESS_SUPPORTED = frozenset(
    {"none", "all supported", "all_supported", "no unsupported claims", "fully supported"}
)


# ── Shared data type ──────────────────────────────────────────────────────────

@dataclass
class ScoredChunk:
    """A retrieved chunk annotated with scores from each pipeline stage."""
    chunk_id: str
    text: str
    doc_id: str
    source: str
    page_number: int
    fused_score: float                    # RRF score (higher = better)
    dense_rank: int | None = None         # rank in dense list  (1-indexed, lower = better)
    bm25_rank: int | None = None          # rank in BM25 list
    reranker_score: float | None = None   # cross-encoder score (higher = better)
    metadata: dict = field(default_factory=dict)


# ── Entity tagging ────────────────────────────────────────────────────────────
# Exported for Phase 4 chunker:  from app.pipeline.retrieval import tag_entities
# Call: chunk.metadata["entities"] = tag_entities(chunk.text)

_spacy_nlp = None   # False = attempted and unavailable; None = not yet tried

_CORP_RE = re.compile(
    r'\b[A-Z][A-Za-z&]+(?: [A-Z][A-Za-z&]+)*'
    r'(?:[ ](?:Inc|Corp|Ltd|LLC|Co|Group|Holdings|Technologies|Solutions))\.?\b'
)
_TICKER_RE = re.compile(r'\b[A-Z]{2,5}\b')


def _load_spacy():
    global _spacy_nlp
    if _spacy_nlp is not None:
        return _spacy_nlp
    try:
        import spacy
        _spacy_nlp = spacy.load("en_core_web_sm")
        logger.info("spaCy en_core_web_sm loaded for entity tagging")
    except Exception as exc:
        logger.debug("spaCy unavailable (%s) — using regex entity tagger", exc)
        _spacy_nlp = False
    return _spacy_nlp


def _regex_tag(text: str) -> list[str]:
    corp = _CORP_RE.findall(text)
    tickers = _TICKER_RE.findall(text[:600])
    return list({*corp, *tickers})


def tag_entities(text: str) -> list[str]:
    """
    Best-effort entity extraction for chunk metadata.
    Uses spaCy en_core_web_sm when available, regex fallback otherwise.
    Returns a deduplicated list of entity strings.
    """
    nlp = _load_spacy()
    if nlp:
        doc = nlp(text[:1500])
        return list({ent.text for ent in doc.ents if ent.label_ in _SPACY_ENTITY_LABELS})
    return _regex_tag(text)


# ── Hybrid search ─────────────────────────────────────────────────────────────

def hybrid_search(
    query: str,
    user_id: str,
    chroma_client,                          # chromadb.ClientAPI (Phase 7)
    bm25_index,                             # BM25IndexProtocol  (Phase 7)
    embed_fn: Callable[[str], list[float]], # Phase 6 embedder
    doc_id: str | None = None,
    top_n: int = 50,
) -> list[ScoredChunk]:
    """
    Run dense + BM25 retrieval and fuse results with Reciprocal Rank Fusion.

    RRF score for chunk c = sum over each list L of: 1 / (RRF_K + rank_of_c_in_L)
    Chunks that appear in both lists naturally score higher.  A chunk ranked 1st
    in BM25 but 20th in dense will outscore a chunk that only appears in dense.

    Returns up to top_n ScoredChunks sorted by fused_score descending.
    """
    collection_name = f"user_{user_id}"
    chunk_data: dict[str, dict] = {}   # chunk_id -> {text, meta, dense_rank?, bm25_rank?}
    dense_count = 0

    # ── 1. Dense retrieval ────────────────────────────────────────────────────
    try:
        with span("embed_query"):
            query_vec = embed_fn(query)
        with span("vector_search", n_results=min(top_n, 100)) as vs:
            coll = chroma_client.get_collection(collection_name)
            where = {"doc_id": doc_id} if doc_id else None
            raw = coll.query(
                query_embeddings=[query_vec],
                n_results=min(top_n, 100),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            for rank, (cid, text, meta) in enumerate(
                zip(raw["ids"][0], raw["documents"][0], raw["metadatas"][0]), 1
            ):
                chunk_data[cid] = {"text": text, "meta": meta or {}, "dense_rank": rank}
                dense_count += 1
            vs.update(returned=dense_count)
    except Exception as exc:
        logger.warning("Dense retrieval failed for user=%s: %s", user_id, exc)

    # ── 2. BM25 retrieval ─────────────────────────────────────────────────────
    bm25_results: list[tuple[str, float]] = []
    try:
        # user_id scopes BM25 to the owner's corpus, mirroring the per-user Chroma
        # collection — the lexical index must never score across other users' chunks.
        with span("bm25_search") as bs:
            bm25_results = bm25_index.query(query, user_id=user_id, doc_id=doc_id, top_n=top_n)
            bs.update(returned=len(bm25_results))
    except Exception as exc:
        logger.warning("BM25 retrieval failed: %s", exc)

    # Fetch text + metadata from ChromaDB for BM25 hits not already in chunk_data
    missing = [cid for cid, _ in bm25_results if cid not in chunk_data]
    if missing:
        try:
            coll = chroma_client.get_collection(collection_name)
            fetched = coll.get(ids=missing, include=["documents", "metadatas"])
            for cid, text, meta in zip(
                fetched["ids"], fetched["documents"], fetched["metadatas"]
            ):
                chunk_data[cid] = {"text": text, "meta": meta or {}}
        except Exception as exc:
            logger.warning("Failed to fetch BM25-only chunks from Chroma: %s", exc)

    for rank, (cid, _) in enumerate(bm25_results, 1):
        if cid in chunk_data:
            chunk_data[cid]["bm25_rank"] = rank

    # ── 3. RRF fusion ─────────────────────────────────────────────────────────
    with span("rrf_fuse", rrf_k=RRF_K) as fs:
        rrf: dict[str, float] = {}
        for cid, data in chunk_data.items():
            score = 0.0
            if "dense_rank" in data:
                score += 1.0 / (RRF_K + data["dense_rank"])
            if "bm25_rank" in data:
                score += 1.0 / (RRF_K + data["bm25_rank"])
            rrf[cid] = score

        top_ids = sorted(rrf, key=lambda x: rrf[x], reverse=True)[:top_n]
        fs.update(fused=len(top_ids))

    results: list[ScoredChunk] = []
    for cid in top_ids:
        d = chunk_data[cid]
        m = d["meta"]
        results.append(ScoredChunk(
            chunk_id=cid,
            text=d["text"],
            doc_id=m.get("doc_id", ""),
            source=m.get("source", ""),
            page_number=int(m.get("page_number", 0)),
            dense_rank=d.get("dense_rank"),
            bm25_rank=d.get("bm25_rank"),
            fused_score=rrf[cid],
            metadata=m,
        ))

    logger.info(
        "hybrid_search user=%s dense=%d bm25=%d fused=%d query=%r",
        user_id, dense_count, len(bm25_results), len(results), query[:60],
    )
    return results


# ── Whole-document retrieval (Scope-Aware Routing, GLOBAL scope) ──────────────

def fetch_whole_document(chroma_client, user_id: str, doc_id: str) -> list[ScoredChunk]:
    """Fetch EVERY chunk of one document as ScoredChunks, in reading order.

    The GLOBAL-scope counterpart to hybrid_search: for questions that operate over
    the whole document ("list all X", "summarize", "build a table"), top-k
    retrieval DROPS the very chunks a complete answer needs. This bypasses
    dense/BM25/RRF/rerank entirely and returns the full document ordered by
    chunk_index, so the model sees the complete set.

    Unlike context_expander.fetch_doc_chunks (which returns ChunkMeta for the merge
    algorithm), this returns ScoredChunk with source/page_number preserved, so the
    prompt builder's [n] labels and the endpoint's citations work unchanged. Any
    Chroma error yields [] and the caller falls back to normal top-k retrieval.
    """
    collection_name = f"user_{user_id}"
    try:
        coll = chroma_client.get_collection(collection_name)
        fetched = coll.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
    except Exception as exc:
        logger.warning("fetch_whole_document failed for doc_id=%s: %s", doc_id, exc)
        return []

    ids = fetched.get("ids") or []
    docs = fetched.get("documents") or []
    metas = fetched.get("metadatas") or []

    items: list[tuple[int, str, str, dict]] = []
    for cid, text, meta in zip(ids, docs, metas):
        meta = meta or {}
        try:
            ci = int(meta.get("chunk_index", 0))
        except (TypeError, ValueError):
            ci = 0
        items.append((ci, cid, text or "", meta))

    items.sort(key=lambda t: t[0])  # reading order — chunk_index ascending

    out: list[ScoredChunk] = []
    for ci, cid, text, meta in items:
        try:
            page = int(meta.get("page_number", 0) or 0)
        except (TypeError, ValueError):
            page = 0
        out.append(ScoredChunk(
            chunk_id=cid,
            text=text,
            doc_id=meta.get("doc_id", doc_id),
            source=meta.get("source", ""),
            page_number=page,
            # Not rank-scored — the whole doc is included; a uniform positive keeps
            # any fused_score consumer (e.g. the retrieval signal) well-behaved.
            fused_score=1.0,
            dense_rank=None,
            bm25_rank=None,
            metadata=meta,
        ))
    logger.info("fetch_whole_document user=%s doc=%s chunks=%d", user_id, doc_id, len(out))
    return out


# ── Cross-encoder reranking ───────────────────────────────────────────────────

_cross_encoder = None  # module-level singleton, lazy-loaded on first rerank call


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers.cross_encoder import CrossEncoder  # noqa
        _cross_encoder = CrossEncoder(_RERANKER_MODEL)
        logger.info("Cross-encoder loaded: %s", _RERANKER_MODEL)
    return _cross_encoder


def rerank(
    query: str,
    candidates: list[ScoredChunk],
    top_k: int = 8,
    reranker=None,  # injectable: any object with .predict(pairs) -> list[float]
) -> list[ScoredChunk]:
    """
    Cross-encoder reranking of hybrid search candidates.

    Unlike bi-encoder embeddings (which encode query and chunk separately),
    the cross-encoder sees both texts concatenated.  This joint scoring makes
    it far better at distinguishing "Apple Inc. revenue" from "apple orchard
    revenue" — the bi-encoder may map both close to a "revenue" query vector,
    but the cross-encoder can read the full context and score accordingly.

    Returns top_k chunks sorted by reranker_score descending.
    Mutates reranker_score on each ScoredChunk in-place.
    """
    if not candidates:
        return []

    with span("rerank", model=_RERANKER_MODEL, candidates_in=len(candidates)) as rsp:
        model = reranker if reranker is not None else _get_cross_encoder()
        pairs = [(query, c.text) for c in candidates]

        try:
            raw_scores = model.predict(pairs)
        except Exception as exc:
            logger.error("Cross-encoder prediction failed (%s) — returning hybrid order", exc)
            rsp.update(kept_out=min(top_k, len(candidates)), reranked=False)
            return candidates[:top_k]

        for chunk, score in zip(candidates, raw_scores):
            chunk.reranker_score = float(score)

        ranked = sorted(candidates, key=lambda c: c.reranker_score, reverse=True)  # type: ignore[arg-type]
        logger.info("rerank: %d → top %d", len(candidates), min(top_k, len(ranked)))
        rsp.update(kept_out=min(top_k, len(ranked)))
        return ranked[:top_k]


# ── LLM helpers (shared by generate + groundedness) ──────────────────────────

# The old direct HuggingFace inference call was deleted (Story 0/1): that endpoint
# (api-inference.huggingface.co) is dead — confirmed in embedding_providers.py and by
# live testing — and it silently returned "" on every call. Generation now flows
# through the swappable GenerationProvider seam (app/generation/*, wired onto
# app.state.llm_fn in main.py's lifespan). generate_answer/groundedness_check require
# an explicit llm_fn passed by the /generate/answer endpoint; there is NO internal
# fallback. When no provider is configured, llm_fn is None and the endpoint returns
# the honest "generation_not_configured" response.


def generate_answer(
    query: str,
    source_chunks: list[ScoredChunk],
    llm_fn: Callable[[str], str] | None = None,
    doc_chunks_fetcher: Callable[[str], list] | None = None,
    query_type=None,
    thin_context: bool = False,
    answer_task=None,
    model_name: str | None = None,
) -> str:
    """
    Generate an answer grounded in source_chunks.  Used by /generate/answer.

    Requires an explicit llm_fn (the dead default HF path was removed — see the note
    above). Callers that have no generation backend configured must not call this;
    the /generate/answer endpoint guards on llm_fn is None and returns an honest
    "generation_not_configured" response instead.

    Story 5: doc_chunks_fetcher (doc_id -> list[ChunkMeta]) is passed through to
    the prompt builder to widen each chunk with its neighbors. None (the default)
    leaves the pre-Story-5 behavior — the raw chunk text — untouched.

    Generation-quality fix: query_type (Part 3) and thin_context (Part 4b) are
    forwarded to the prompt builder to append the response-style hint / limited-
    material note. Both default to off, preserving the pre-fix prompt.
    """
    if llm_fn is None:
        raise RuntimeError("generation_not_configured: generate_answer requires an llm_fn")
    # Story 2: the grounded prompt labels each source [n] with its filename+page,
    # specifies the [n] citation format, and gives the model an abstention clause.
    # The old bare "Context:\n...---..." blob (no labels, no format, no escape hatch)
    # is gone — parsing/abstention now happen on the response (see citation_parser).
    #
    # Phase 9A: prompt_build and generate are emitted as sibling spans (children of
    # whatever span is active — rag.request when called from /generate/answer). The
    # split is instrumentation only; the generation behaviour is unchanged.
    with span("prompt_build") as psp:
        prompt = build_grounded_prompt(
            query, source_chunks, doc_chunks_fetcher=doc_chunks_fetcher,
            query_type=query_type, thin_context=thin_context, answer_task=answer_task,
        )
        psp.update(source_count=len(source_chunks))
        if capture_text_enabled():
            psp.update(prompt=prompt)
    with span("generate", model=model_name):
        # Token usage / ttft / finish_reason come from the provider usage field in
        # Phase 9B — the str-returning llm_fn seam does not surface them here, and a
        # tiktoken estimate is deliberately NOT substituted.
        return llm_fn(prompt).strip()


# ── Groundedness check ────────────────────────────────────────────────────────

def groundedness_check(
    query: str,
    answer_draft: str,
    source_chunks: list[ScoredChunk],
    llm_fn: Callable[[str], str] | None = None,
    answer_task=None,
) -> dict:
    """
    Verify that every claim in answer_draft is supported by source_chunks.

    Makes a SEPARATE LLM call distinct from the one that generated the answer.
    This mirrors the "gauntlet" QA pattern: generate → verify → surface gaps.

    Args:
        query:         The original user question.
        answer_draft:  The LLM-generated answer to verify.
        source_chunks: The chunks used to generate the answer.
        llm_fn:        Injectable LLM callable (for tests / alternative models).

    Returns:
        {
            "grounded":           bool,
            "confidence":         float,   # 0.0–1.0
            "unsupported_claims": list[str],
            "verified":           bool,    # True = verification ran; False = it didn't
            "error":              str | None,  # set only when verification failed
        }

    Failure policy: FAIL CLOSED. If the verification LLM call fails or is not
    configured, return grounded=False, confidence=0.0, verified=False. A total
    outage of the verifier must NOT look like perfect confidence to the user — the
    frontend shows "unverified" instead of either "confident" or "wrong".
    """
    if llm_fn is None:
        # No verifier configured (the dead HF default was removed — Story 1 wires a
        # real provider). Fail closed rather than pretend the answer was verified.
        return {
            "grounded": False,
            "confidence": 0.0,
            "unsupported_claims": [],
            "verified": False,
            "error": "verification_unavailable",
        }
    call = llm_fn

    # Scope-Aware Routing (Phase 3): for INFER (hypothetical / counterfactual)
    # answers the "must be DIRECTLY supported" test is wrong — a good inference is
    # deliberately NOT literal in the sources. The strict prompt marks such answers
    # unsupported and collapses confidence. For INFER we check the real failure mode
    # instead: fabricated facts / entities that aren't in the sources, allowing
    # reasonable inference. Every OTHER task keeps the original strict prompt
    # verbatim (default None → unchanged behavior).
    is_infer = getattr(answer_task, "value", answer_task) == "infer"
    # NUMBER the excerpts exactly as the generation prompt numbered them ("[n]").
    # The answer cites its sources as [1], [2], … — if the verifier is shown an
    # unnumbered blob it has no referent for those markers and reports the CITATION
    # as an unsupported claim ("the citation [1, 2] is inaccurate because the source
    # excerpts do not use a numbering system"). That turned correct, fully grounded
    # one-line answers into confidence=0.0. Measured on the 45-query fixture: it was
    # the dominant cause of a 33% false "hallucination" rate.
    context = "\n---\n".join(
        f"[{n}] {c.text}" for n, c in enumerate(source_chunks[:8], 1)
    )
    _citation_note = (
        "The answer cites sources with bracketed markers like [1] or [1, 2]; those "
        "markers refer to the numbered excerpts above. Judge only the factual claims "
        "— never treat a citation marker itself as a claim.\n"
    )
    if is_infer:
        prompt = (
            "You are a fact-checking assistant evaluating a REASONING answer.\n\n"
            "Source excerpts (the only established facts):\n"
            f"{context}\n\n"
            f"Query: {query}\n\n"
            f"Answer to verify:\n{answer_draft}\n\n"
            f"{_citation_note}"
            "The answer may draw REASONABLE INFERENCES from the sources — those are "
            "acceptable. List only claims that introduce NEW FACTS or ENTITIES that "
            "are neither stated in nor reasonably inferable from the sources. If there "
            "are none, write exactly: none\n"
            "Unsupported claims:"
        )
    else:
        prompt = (
            "You are a fact-checking assistant.\n\n"
            "Source excerpts (the ONLY information allowed to support the answer):\n"
            f"{context}\n\n"
            f"Query: {query}\n\n"
            f"Answer to verify:\n{answer_draft}\n\n"
            f"{_citation_note}"
            "List every claim in the answer that is NOT directly supported by the source "
            "excerpts above.  If all claims are supported, write exactly: none\n"
            "Unsupported claims:"
        )

    try:
        response = call(prompt).strip()
    except Exception as exc:
        # Any error from the verifier call is a verification failure, not a pass.
        logger.warning("Groundedness verification call failed: %s", exc)
        response = ""

    if not response:
        # Verifier unavailable (empty output or raised) — FAIL CLOSED.
        return {
            "grounded": False,
            "confidence": 0.0,
            "unsupported_claims": [],
            "verified": False,
            "error": "verification_unavailable",
        }

    first_line = response.lower().split("\n")[0].strip().rstrip(".:")
    # Models rarely reply with the single literal token we ask for. Treat any clear
    # "nothing is unsupported" phrasing as fully grounded — otherwise a perfect
    # answer whose verifier says "No unsupported claims." gets that very sentence
    # miscounted as an unsupported claim, collapsing confidence to 0.
    _GROUNDED_PHRASES = (
        "no unsupported", "all claims are supported", "all of the claims are supported",
        "everything is supported", "all supported", "fully supported", "none are unsupported",
    )
    if (
        first_line in _GROUNDEDNESS_SUPPORTED
        or first_line.startswith("none")
        or first_line in {"no", "n/a", "na"}
        or any(p in first_line for p in _GROUNDED_PHRASES)
    ):
        return {"grounded": True, "confidence": 1.0, "unsupported_claims": [], "verified": True}

    # Collect the listed claims, skipping meta/negative lines (headers, a restated
    # "no unsupported claims", etc.) that are not themselves claims.
    _META = ("no unsupported", "all supported", "fully supported", "all claims are supported",
             "everything is supported", "unsupported claims", "here are", "the following",
             "no claims")
    # A verifier asked for a LIST often answers in prose, and its bullets frequently
    # AFFIRM support ("The claim that X is supported by the source, but …"). Counting
    # such a line as an unsupported claim inverts its meaning and collapses confidence
    # on a correct answer. A line that says a claim IS supported — and does not also
    # negate it — is not a finding against the answer.
    _AFFIRMS = ("is supported", "are supported", "is directly supported",
                "is fully supported", "supported by the source")
    _NEGATIONS = ("not supported", "not directly supported", "isn't supported",
                  "is not", "no support", "unsupported")

    def _affirms_support(text: str) -> bool:
        return any(a in text for a in _AFFIRMS) and not any(n in text for n in _NEGATIONS)

    unsupported: list[str] = []
    for line in response.split("\n"):
        clean = line.strip().lstrip("-•*0123456789.) ").strip()
        cl = clean.lower()
        if not clean or len(clean) <= 8:
            continue
        if cl == "none" or any(m in cl for m in _META):
            continue
        if _affirms_support(cl):
            continue
        unsupported.append(clean)

    if not unsupported:
        return {"grounded": True, "confidence": 0.85, "unsupported_claims": [], "verified": True}

    # Confidence = fraction of answer sentences that ARE supported
    try:
        import nltk
        sentences = nltk.sent_tokenize(answer_draft)
    except Exception:
        sentences = [s.strip() for s in answer_draft.split(".") if s.strip()]

    total = max(len(sentences), 1)
    confidence = round(max(0.0, 1.0 - len(unsupported) / total), 3)

    logger.warning(
        "Groundedness: %d unsupported claims / ~%d sentences  confidence=%.2f",
        len(unsupported), total, confidence,
    )
    return {
        "grounded": False,
        "confidence": confidence,
        "unsupported_claims": unsupported,
        "verified": True,
    }
