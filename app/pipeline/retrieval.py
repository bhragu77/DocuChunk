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

import httpx

from app.config import get_settings

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
        query_vec = embed_fn(query)
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
    except Exception as exc:
        logger.warning("Dense retrieval failed for user=%s: %s", user_id, exc)

    # ── 2. BM25 retrieval ─────────────────────────────────────────────────────
    bm25_results: list[tuple[str, float]] = []
    try:
        # user_id scopes BM25 to the owner's corpus, mirroring the per-user Chroma
        # collection — the lexical index must never score across other users' chunks.
        bm25_results = bm25_index.query(query, user_id=user_id, doc_id=doc_id, top_n=top_n)
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
    rrf: dict[str, float] = {}
    for cid, data in chunk_data.items():
        score = 0.0
        if "dense_rank" in data:
            score += 1.0 / (RRF_K + data["dense_rank"])
        if "bm25_rank" in data:
            score += 1.0 / (RRF_K + data["bm25_rank"])
        rrf[cid] = score

    top_ids = sorted(rrf, key=lambda x: rrf[x], reverse=True)[:top_n]

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

    model = reranker if reranker is not None else _get_cross_encoder()
    pairs = [(query, c.text) for c in candidates]

    try:
        raw_scores = model.predict(pairs)
    except Exception as exc:
        logger.error("Cross-encoder prediction failed (%s) — returning hybrid order", exc)
        return candidates[:top_k]

    for chunk, score in zip(candidates, raw_scores):
        chunk.reranker_score = float(score)

    ranked = sorted(candidates, key=lambda c: c.reranker_score, reverse=True)  # type: ignore[arg-type]
    logger.info("rerank: %d → top %d", len(candidates), min(top_k, len(ranked)))
    return ranked[:top_k]


# ── LLM helpers (shared by generate + groundedness) ──────────────────────────

def _call_hf_api(prompt: str) -> str:
    """POST to HuggingFace Inference API.  Returns generated text or empty string on failure."""
    settings = get_settings()
    if not settings.hf_api_token:
        logger.warning("HF_API_TOKEN not configured — LLM call skipped")
        return ""
    url = f"https://api-inference.huggingface.co/models/{settings.hf_gen_model}"
    headers = {"Authorization": f"Bearer {settings.hf_api_token}"}
    try:
        resp = httpx.post(
            url,
            json={"inputs": prompt, "parameters": {"max_new_tokens": 300, "temperature": 0.1}},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            return data[0].get("generated_text", "")
        return str(data)
    except Exception as exc:
        logger.error("HF API call failed: %s", exc)
        return ""


def generate_answer(
    query: str,
    source_chunks: list[ScoredChunk],
    llm_fn: Callable[[str], str] | None = None,
) -> str:
    """Generate an answer grounded in source_chunks.  Used by /generate/answer."""
    call = llm_fn if llm_fn is not None else _call_hf_api
    context = "\n---\n".join(c.text for c in source_chunks)
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer based ONLY on the context above. Be concise and cite the source:"
    )
    return call(prompt).strip()


# ── Groundedness check ────────────────────────────────────────────────────────

def groundedness_check(
    query: str,
    answer_draft: str,
    source_chunks: list[ScoredChunk],
    llm_fn: Callable[[str], str] | None = None,
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
            "unsupported_claims": list[str]
        }

    Failure policy: if the LLM call fails, returns grounded=True (fail open)
    so a connectivity issue never silently swallows the user's answer.
    """
    call = llm_fn if llm_fn is not None else _call_hf_api

    context = "\n---\n".join(c.text for c in source_chunks[:8])
    prompt = (
        "You are a fact-checking assistant.\n\n"
        "Source excerpts (the ONLY information allowed to support the answer):\n"
        f"{context}\n\n"
        f"Query: {query}\n\n"
        f"Answer to verify:\n{answer_draft}\n\n"
        "List every claim in the answer that is NOT directly supported by the source "
        "excerpts above.  If all claims are supported, write exactly: none\n"
        "Unsupported claims:"
    )

    response = call(prompt).strip()

    if not response:
        # LLM unavailable — fail open
        return {"grounded": True, "confidence": 1.0, "unsupported_claims": []}

    first_line = response.lower().split("\n")[0].strip().rstrip(".")
    if first_line in _GROUNDEDNESS_SUPPORTED:
        return {"grounded": True, "confidence": 1.0, "unsupported_claims": []}

    unsupported: list[str] = []
    for line in response.split("\n"):
        clean = line.strip().lstrip("-•*123456789. ").strip()
        if clean and len(clean) > 8:
            unsupported.append(clean)

    if not unsupported:
        return {"grounded": True, "confidence": 0.85, "unsupported_claims": []}

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
    }
