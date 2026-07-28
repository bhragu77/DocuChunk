"""
Phase 8 tests — hybrid retrieval, reranking, and groundedness check.

All tests operate on pure Python functions with injected mocks.  No real
ChromaDB, no real BM25, no real LLM, no real cross-encoder is required.
This keeps tests fast and deterministic regardless of API availability.

Key regression: the Apple-ambiguity fixture (apple fruit vs Apple Inc.)
must be present and must pass.  If tests/eval_set.json drifts out of sync
with retrieval.py behaviour, these tests will catch it.
"""
import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.pipeline.retrieval import (
    RRF_K,
    ScoredChunk,
    generate_answer,
    groundedness_check,
    hybrid_search,
    rerank,
    tag_entities,
)

EVAL_SET = json.loads((Path(__file__).parent / "eval_set.json").read_text())
APPLE_CASE = EVAL_SET["cases"][0]


# ── Helpers / fixtures ────────────────────────────────────────────────────────

def _scored_chunk(chunk_id: str, text: str, doc_id: str = "doc1", page: int = 1) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id,
        text=text,
        doc_id=doc_id,
        source="test.pdf",
        page_number=page,
        fused_score=0.0,
    )


def _apple_chunks() -> dict[str, ScoredChunk]:
    """Build ScoredChunk objects from the eval_set fixture."""
    chunks = {}
    for doc in APPLE_CASE["documents"]:
        for c in doc["chunks"]:
            chunks[c["chunk_id"]] = ScoredChunk(
                chunk_id=c["chunk_id"],
                text=c["text"],
                doc_id=c["metadata"]["doc_id"],
                source=c["metadata"]["source"],
                page_number=c["metadata"]["page_number"],
                fused_score=0.0,
                metadata=c["metadata"],
            )
    return chunks


def _make_chroma_mock(dense_order: list[str], chunk_lookup: dict[str, ScoredChunk]):
    """
    Return a mock ChromaDB client whose collection.query() returns chunks in
    the given order (dense_order[0] = rank 1, etc.).
    collection.get() returns any requested chunk_ids from chunk_lookup.
    """
    ids = dense_order
    texts = [chunk_lookup[cid].text for cid in ids]
    metas = [chunk_lookup[cid].metadata for cid in ids]

    mock_coll = MagicMock()
    mock_coll.query.return_value = {
        "ids": [ids],
        "documents": [texts],
        "metadatas": [metas],
        "distances": [[0.1 * i for i in range(len(ids))]],
    }

    def _get(ids, include=None):
        return {
            "ids": ids,
            "documents": [chunk_lookup[cid].text for cid in ids if cid in chunk_lookup],
            "metadatas": [chunk_lookup[cid].metadata for cid in ids if cid in chunk_lookup],
        }

    mock_coll.get.side_effect = _get

    mock_client = MagicMock()
    mock_client.get_collection.return_value = mock_coll
    return mock_client


class _MockBM25:
    """Deterministic BM25 mock: returns a fixed ranked list."""
    def __init__(self, results: list[tuple[str, float]]):
        self._results = results

    def query(self, text: str, user_id=None, doc_id=None, top_n: int = 50):
        if doc_id:
            return [(cid, s) for cid, s in self._results if cid.startswith(f"{doc_id}_") or True]
        return self._results[:top_n]

    def upsert(self, *args, **kwargs): pass
    def upsert_many(self, *args, **kwargs): pass
    def delete(self, *args, **kwargs): pass
    def delete_chunks(self, *args, **kwargs): pass


class _MockCrossEncoder:
    """
    Deterministic cross-encoder mock.
    Score = 2.0 if chunk contains '$394 billion', -1.0 if contains 'orchard', else 0.0.
    """
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = []
        for _, doc in pairs:
            if "$394 billion" in doc or "Apple Inc." in doc and "revenue" in doc.lower():
                scores.append(2.0)
            elif "orchard" in doc.lower():
                scores.append(-1.0)
            else:
                scores.append(0.0)
        return scores


# ── 1. RRF fusion test: entity ambiguity ──────────────────────────────────────

def test_rrf_promotes_correct_entity_chunk():
    """
    Core regression for entity ambiguity.

    Dense retrieval ranks the orchard chunk FIRST (both docs share "revenue"
    vocabulary, so the embedding is confused).
    BM25 ranks the company chunk FIRST (exact match on "Apple Inc." and "billion").

    After RRF fusion the COMPANY chunk must rank above the orchard chunk.

    RRF math (k=60):
      orchard: 1/(60+1) + 1/(60+10) = 0.01639 + 0.01429 = 0.03068
      company: 1/(60+2) + 1/(60+1)  = 0.01613 + 0.01639 = 0.03252
    """
    chunks = _apple_chunks()
    company_id = "chunk_company_001"
    orchard_id = "chunk_orchard_001"

    # Dense: orchard first (false positive), company second
    dense_order = [orchard_id, company_id]
    # BM25: company first, then a gap, then orchard at rank 10
    bm25_results = [(company_id, 18.5)] + [
        (f"other_{i}", 5.0 - i * 0.3) for i in range(8)
    ] + [(orchard_id, 1.2)]

    # Add placeholder chunks for the "other_*" ids that BM25 returns
    for i in range(8):
        cid = f"other_{i}"
        chunks[cid] = ScoredChunk(
            chunk_id=cid, text=f"Other content {i}", doc_id="doc_other",
            source="other.pdf", page_number=1, fused_score=0.0,
        )

    mock_chroma = _make_chroma_mock(dense_order, chunks)
    mock_bm25 = _MockBM25(bm25_results)
    embed_fn = lambda q: [0.1] * 384  # noqa: E731  — fixed vector, doesn't affect test

    results = hybrid_search(
        query=APPLE_CASE["query"],
        user_id="user_test",
        chroma_client=mock_chroma,
        bm25_index=mock_bm25,
        embed_fn=embed_fn,
        top_n=20,
    )

    assert len(results) > 0, "hybrid_search returned empty list"
    top_ids = [r.chunk_id for r in results]
    assert company_id in top_ids, f"Company chunk missing from results: {top_ids}"
    assert orchard_id in top_ids, "Orchard chunk missing from results (needed to test ordering)"

    company_pos = top_ids.index(company_id)
    orchard_pos = top_ids.index(orchard_id)
    assert company_pos < orchard_pos, (
        f"RRF failed: orchard_chunk ranked #{orchard_pos + 1} "
        f"but company_chunk ranked #{company_pos + 1} — expected company first.\n"
        f"company fused_score={results[company_pos].fused_score:.6f}  "
        f"orchard fused_score={results[orchard_pos].fused_score:.6f}"
    )


def test_rrf_score_formula():
    """Unit-test the RRF score arithmetic directly, independent of mocking."""
    # With dense_rank=1, bm25_rank=10:
    expected_orchard = 1 / (RRF_K + 1) + 1 / (RRF_K + 10)
    assert math.isclose(expected_orchard, 1 / 61 + 1 / 70, rel_tol=1e-9)

    # With dense_rank=2, bm25_rank=1:
    expected_company = 1 / (RRF_K + 2) + 1 / (RRF_K + 1)
    assert math.isclose(expected_company, 1 / 62 + 1 / 61, rel_tol=1e-9)

    assert expected_company > expected_orchard, (
        "Company chunk (dense rank 2, BM25 rank 1) should score higher than "
        "orchard chunk (dense rank 1, BM25 rank 10)"
    )


def test_hybrid_search_dense_only_fallback():
    """If BM25 fails, hybrid_search falls back gracefully to dense-only results."""
    chunks = _apple_chunks()

    class FailingBM25:
        def query(self, *args, **kwargs): raise RuntimeError("index not available")
        def upsert(self, *args, **kwargs): pass
        def delete(self, *args, **kwargs): pass

    mock_chroma = _make_chroma_mock(["chunk_company_001"], chunks)
    results = hybrid_search(
        query="Apple revenue",
        user_id="u1",
        chroma_client=mock_chroma,
        bm25_index=FailingBM25(),
        embed_fn=lambda q: [0.0] * 384,
        top_n=10,
    )
    assert len(results) == 1
    assert results[0].chunk_id == "chunk_company_001"


def test_hybrid_search_both_fail_returns_empty():
    """If both dense and BM25 fail, return empty list (no exception propagated)."""
    class FailChroma:
        def get_collection(self, name): raise RuntimeError("no collection")

    class FailBM25:
        def query(self, *a, **k): raise RuntimeError("no index")
        def upsert(self, *a, **k): pass
        def delete(self, *a, **k): pass

    results = hybrid_search(
        query="test", user_id="u1",
        chroma_client=FailChroma(), bm25_index=FailBM25(),
        embed_fn=lambda q: [0.0] * 384, top_n=10,
    )
    assert results == []


# ── 2. Cross-encoder reranking tests ─────────────────────────────────────────

def test_rerank_demotes_false_positive():
    """
    After hybrid search, a false-positive orchard chunk may appear above the
    company chunk.  The cross-encoder must demote it.
    """
    company_chunk = _scored_chunk(
        "chunk_company_001",
        "Apple Inc. reported annual revenue of $394 billion in fiscal 2023.",
        doc_id="doc_apple_company",
    )
    orchard_chunk = _scored_chunk(
        "chunk_orchard_001",
        "Apple orchards generate revenue of $10,000 per acre in Washington State.",
        doc_id="doc_apple_orchard",
    )

    # Hybrid search returned orchard FIRST (false positive), company SECOND
    candidates = [orchard_chunk, company_chunk]

    ranked = rerank(
        query="What is Apple Inc.'s annual revenue?",
        candidates=candidates,
        top_k=2,
        reranker=_MockCrossEncoder(),
    )

    assert ranked[0].chunk_id == "chunk_company_001", (
        "Reranker failed to promote the correct company chunk above the false-positive orchard chunk"
    )
    assert ranked[0].reranker_score > ranked[1].reranker_score


def test_rerank_preserves_top_k_limit():
    candidates = [
        _scored_chunk(f"c{i}", f"Chunk content number {i}") for i in range(20)
    ]
    ranked = rerank("test query", candidates, top_k=5, reranker=_MockCrossEncoder())
    assert len(ranked) <= 5


def test_rerank_sets_reranker_score_on_chunks():
    c1 = _scored_chunk("c1", "Apple Inc. revenue $394 billion")
    c2 = _scored_chunk("c2", "Some unrelated orchard content")
    ranked = rerank("Apple revenue", [c1, c2], top_k=2, reranker=_MockCrossEncoder())
    assert all(c.reranker_score is not None for c in ranked)


def test_rerank_empty_candidates_returns_empty():
    result = rerank("anything", [], top_k=5, reranker=_MockCrossEncoder())
    assert result == []


def test_rerank_falls_back_on_model_error():
    """If the cross-encoder raises, rerank returns candidates in hybrid order."""
    class BrokenEncoder:
        def predict(self, pairs): raise RuntimeError("GPU OOM")

    candidates = [
        _scored_chunk("c1", "Content A"),
        _scored_chunk("c2", "Content B"),
    ]
    result = rerank("query", candidates, top_k=2, reranker=BrokenEncoder())
    assert [r.chunk_id for r in result] == ["c1", "c2"]  # original order preserved


# ── 3. Groundedness check tests ───────────────────────────────────────────────

def _source_chunk(text: str) -> ScoredChunk:
    return _scored_chunk("src1", text)


def test_groundedness_flags_unsupported_claim():
    """
    An answer that includes a fabricated claim must be detected as ungrounded.
    Uses injected llm_fn to deterministically simulate the LLM response.
    """
    source = _source_chunk(
        "Apple Inc. reported annual revenue of $394 billion in fiscal year 2023."
    )
    answer = (
        "Apple Inc. reported annual revenue of $394 billion in fiscal 2023. "
        "They also acquired Netflix in 2023 to boost their services segment."
    )

    def mock_llm(prompt: str) -> str:
        return "- They also acquired Netflix in 2023 to boost their services segment."

    result = groundedness_check(
        query="What is Apple's revenue?",
        answer_draft=answer,
        source_chunks=[source],
        llm_fn=mock_llm,
    )

    assert result["grounded"] is False
    assert result["confidence"] < 1.0
    assert len(result["unsupported_claims"]) >= 1
    assert any("Netflix" in claim for claim in result["unsupported_claims"]), (
        f"Expected Netflix acquisition claim to be flagged, got: {result['unsupported_claims']}"
    )


def test_groundedness_passes_fully_supported_answer():
    """An answer where all claims trace directly to source chunks must pass."""
    source = _source_chunk(
        "Apple Inc. reported annual revenue of $394 billion in fiscal year 2023, "
        "with iPhone sales at 52 percent of total revenue."
    )
    answer = "Apple Inc.'s annual revenue was $394 billion, with iPhone at 52 percent."

    def mock_llm(prompt: str) -> str:
        return "none"

    result = groundedness_check(
        query="What is Apple's revenue?",
        answer_draft=answer,
        source_chunks=[source],
        llm_fn=mock_llm,
    )

    assert result["grounded"] is True
    assert result["confidence"] == 1.0
    assert result["unsupported_claims"] == []


def test_groundedness_fails_closed_when_llm_unavailable():
    """Story 0 BUG 2: if the verifier returns an empty string (API failure), FAIL
    CLOSED — grounded=False, confidence=0.0, verified=False. A verifier outage must
    never look like perfect confidence."""
    source = _source_chunk("Some content.")
    result = groundedness_check(
        query="test", answer_draft="Some answer.", source_chunks=[source],
        llm_fn=lambda p: "",
    )
    assert result["grounded"] is False
    assert result["confidence"] == 0.0
    assert result["verified"] is False
    assert result["unsupported_claims"] == []
    assert result.get("error") == "verification_unavailable"


def test_groundedness_fails_closed_when_llm_raises():
    """A verifier that RAISES (not just empty) is also a failure → fail closed."""
    def boom(prompt: str) -> str:
        raise RuntimeError("model backend exploded")

    source = _source_chunk("Some content.")
    result = groundedness_check(
        query="test", answer_draft="Some answer.", source_chunks=[source],
        llm_fn=boom,
    )
    assert result["grounded"] is False
    assert result["confidence"] == 0.0
    assert result["verified"] is False


def test_groundedness_not_configured_fails_closed():
    """With no llm_fn at all, groundedness fails closed (dead HF default removed)."""
    source = _source_chunk("Some content.")
    result = groundedness_check(
        query="test", answer_draft="Some answer.", source_chunks=[source],
    )
    assert result["grounded"] is False
    assert result["confidence"] == 0.0
    assert result["verified"] is False


def test_groundedness_happy_path_sets_verified_true():
    """The fix only changes the ERROR path: a successful verifier still returns its
    normal result, now annotated verified=True."""
    source = _source_chunk("The sky is blue.")

    supported = groundedness_check(
        query="q", answer_draft="The sky is blue.", source_chunks=[source],
        llm_fn=lambda p: "none",
    )
    assert supported["grounded"] is True
    assert supported["confidence"] == 1.0
    assert supported["verified"] is True

    unsupported = groundedness_check(
        query="q", answer_draft="The sky is blue. Cats can fly.", source_chunks=[source],
        llm_fn=lambda p: "- Cats can fly.",
    )
    assert unsupported["grounded"] is False
    assert unsupported["verified"] is True
    assert any("Cats" in c for c in unsupported["unsupported_claims"])


def test_groundedness_confidence_decreases_with_more_unsupported_claims():
    source = _source_chunk("One single fact: the sky is blue.")
    answer = "The sky is blue. Cats can fly. Water is dry. Fire is cold."

    def mock_llm(prompt: str) -> str:
        return "- Cats can fly.\n- Water is dry.\n- Fire is cold."

    result = groundedness_check(
        query="Tell me facts", answer_draft=answer,
        source_chunks=[source], llm_fn=mock_llm,
    )

    assert result["grounded"] is False
    assert result["confidence"] < 0.5, (
        f"Expected low confidence with 3 unsupported claims, got {result['confidence']}"
    )


# ── 4. /search/semantic never calls LLM ──────────────────────────────────────

def test_semantic_search_never_invokes_llm():
    """
    POST /search/semantic is a pure-retrieval endpoint.
    generate_answer() and groundedness_check() must never be called.
    """
    chunks = _apple_chunks()
    mock_chroma = _make_chroma_mock(["chunk_company_001"], chunks)
    mock_bm25 = _MockBM25([("chunk_company_001", 10.0)])
    embed_fn = lambda q: [0.0] * 384  # noqa: E731

    generate_called = []
    groundedness_called = []

    candidates = hybrid_search(
        query="Apple revenue",
        user_id="u1",
        chroma_client=mock_chroma,
        bm25_index=mock_bm25,
        embed_fn=embed_fn,
        top_n=10,
    )
    ranked = rerank("Apple revenue", candidates, top_k=5, reranker=_MockCrossEncoder())

    # The semantic search path ends HERE — no LLM calls below
    # generate_answer and groundedness_check are NOT called in /search/semantic
    assert len(generate_called) == 0
    assert len(groundedness_called) == 0
    assert len(ranked) > 0

    # Verify by patching the generation functions and asserting retrieval never calls them.
    with patch("app.pipeline.retrieval.generate_answer") as mock_gen, \
         patch("app.pipeline.retrieval.groundedness_check") as mock_ground:
        _ = hybrid_search(
            query="Apple revenue", user_id="u1",
            chroma_client=mock_chroma, bm25_index=mock_bm25,
            embed_fn=embed_fn, top_n=10,
        )
        _ = rerank("Apple revenue", candidates, top_k=5, reranker=_MockCrossEncoder())
        mock_gen.assert_not_called()
        mock_ground.assert_not_called()


# ── 5. /generate/answer returns grounded=False with fabricated claim ──────────

def test_generate_answer_returns_grounded_false_for_hallucination():
    """
    When the LLM generates an answer containing a claim not in the source
    chunks, the endpoint must return grounded=False and surface the claim.

    Uses injected llm_fn calls to control both generation and groundedness.
    """
    source_chunk = _source_chunk(
        "Apple Inc. reported annual revenue of $394 billion in fiscal 2023."
    )

    hallucinated_answer = (
        "Apple Inc. reported annual revenue of $394 billion. "
        "The CEO Tim Cook stated they plan to acquire Netflix by 2025."
    )
    groundedness_llm_response = "- The CEO Tim Cook stated they plan to acquire Netflix by 2025."

    # Both generation and groundedness use injected llm_fn arguments
    answer = generate_answer(
        query="What is Apple's revenue?",
        source_chunks=[source_chunk],
        llm_fn=lambda p: hallucinated_answer,
    )
    assert answer == hallucinated_answer

    result = groundedness_check(
        query="What is Apple's revenue?",
        answer_draft=answer,
        source_chunks=[source_chunk],
        llm_fn=lambda p: groundedness_llm_response,
    )

    assert result["grounded"] is False, "Expected grounded=False for hallucinated claim"
    assert any("Netflix" in c for c in result["unsupported_claims"]), (
        f"Expected Netflix claim flagged, got: {result['unsupported_claims']}"
    )


def test_generate_answer_returns_grounded_true_for_clean_answer():
    source_chunk = _source_chunk(
        "Apple Inc. reported annual revenue of $394 billion in fiscal 2023."
    )
    clean_answer = "Apple Inc.'s revenue was $394 billion in fiscal 2023."

    answer = generate_answer(
        query="What is Apple's revenue?",
        source_chunks=[source_chunk],
        llm_fn=lambda p: clean_answer,
    )

    result = groundedness_check(
        query="What is Apple's revenue?",
        answer_draft=answer,
        source_chunks=[source_chunk],
        llm_fn=lambda p: "none",
    )

    assert result["grounded"] is True
    assert result["unsupported_claims"] == []


# ── 6. Eval set fixture integrity ─────────────────────────────────────────────

def test_eval_set_apple_ambiguity():
    """
    Smoke-test: load the eval_set.json fixture and verify that hybrid_search
    with appropriate mocks returns the expected chunk first.

    This is the canonical regression test for entity ambiguity.
    """
    case = APPLE_CASE
    chunks = _apple_chunks()

    # Simulate: dense gets confused and ranks orchard first
    # BM25 correctly identifies the company chunk via "Apple Inc." + "billion"
    mock_chroma = _make_chroma_mock(
        ["chunk_orchard_001", "chunk_company_001"],
        chunks,
    )
    mock_bm25 = _MockBM25([
        ("chunk_company_001", 22.0),
        ("chunk_company_002", 8.0),
        ("chunk_orchard_002", 3.0),
        ("chunk_orchard_001", 1.5),
    ])

    results = hybrid_search(
        query=case["query"],
        user_id="eval_user",
        chroma_client=mock_chroma,
        bm25_index=mock_bm25,
        embed_fn=lambda q: [0.1] * 384,
        top_n=20,
    )

    top = results[0]
    assert top.chunk_id == case["expected_top_chunk_id"], (
        f"Eval case '{case['id']}' failed: "
        f"expected {case['expected_top_chunk_id']} at rank 1, got {top.chunk_id}"
    )
    assert top.doc_id == case["expected_doc_id"]


# ── 7. tag_entities ───────────────────────────────────────────────────────────

def test_tag_entities_extracts_corporate_suffix():
    entities = tag_entities("Apple Inc. reported strong earnings this quarter.")
    # With spaCy OR regex, "Apple Inc." should be detected
    assert any("Apple" in e for e in entities), f"Expected Apple-related entity, got {entities}"


def test_tag_entities_does_not_tag_plain_fruit():
    entities = tag_entities("Fresh apple orchards produce fruit in autumn seasons.")
    # "apple" (lowercase) and no corporate suffix — regex should NOT tag it
    # spaCy might tag "autumn" as TIME but not "apple" as ORG
    org_entities = [e for e in entities if "apple" in e.lower() and len(e) <= 6]
    assert not org_entities, f"Bare 'apple' should not be tagged as a corporate entity: {entities}"


def test_tag_entities_returns_list():
    result = tag_entities("Google LLC and Microsoft Corp dominate cloud computing.")
    assert isinstance(result, list)


# ── 8. Endpoint tests (TestClient) ────────────────────────────────────────────
# These exercise the router wiring (app.state deps, auth, response models) with the
# heavy retrieval internals patched, so no cross-encoder/model/LLM is loaded in CI.

import tempfile

from fastapi.testclient import TestClient

from app.main import app
from app.core.dependencies import get_current_user
from app.pipeline.bm25_index import BM25Index


class _FakeUser:
    id = "u_endpoint"
    email = "endpoint@example.com"


def _cand(cid, text, doc_id="docA", source="m.pdf", page=1):
    return ScoredChunk(
        chunk_id=cid, text=text, doc_id=doc_id, source=source, page_number=page,
        fused_score=0.5, dense_rank=1, bm25_rank=1, reranker_score=1.0,
    )


@pytest.fixture
def endpoint_client():
    app.dependency_overrides[get_current_user] = lambda: _FakeUser()
    prev_bm25 = getattr(app.state, "bm25", None)
    prev_embed = getattr(app.state, "embed_fn", None)
    prev_llm = getattr(app.state, "llm_fn", None)
    app.state.bm25 = BM25Index(persist_dir=tempfile.mkdtemp(prefix="bm25_ep_"))
    app.state.embed_fn = lambda text: [0.1] * 384
    # Default: NO generation backend wired (Story 1 wires the real provider). Tests
    # that exercise the generation happy path set app.state.llm_fn explicitly.
    app.state.llm_fn = None
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.state.bm25 = prev_bm25
        app.state.embed_fn = prev_embed
        app.state.llm_fn = prev_llm


def test_semantic_search_endpoint_never_calls_llm(endpoint_client, monkeypatch):
    """POST /search/semantic returns ranked citations and must never touch the LLM
    or load a cross-encoder (rerank is patched to a passthrough)."""
    cands = [_cand("c1", "Apple Inc. revenue was $394 billion.")]
    monkeypatch.setattr("app.routers.search.hybrid_search", lambda **kw: cands)
    monkeypatch.setattr("app.routers.search.rerank", lambda q, c, top_k: c[:top_k])
    # Wire a generation backend so we can prove /search/semantic never touches it.
    llm = MagicMock()
    app.state.llm_fn = llm

    resp = endpoint_client.post("/search/semantic", json={"query": "Apple revenue", "top_k": 5})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] == 1
    assert body["results"][0]["chunk_id"] == "c1"
    assert body["results"][0]["reranker_score"] is not None
    llm.assert_not_called()   # pure retrieval — no generation, no groundedness


def test_semantic_search_503_without_bm25(monkeypatch):
    """The endpoint reports 503 (not 500) when Phase 8 deps are absent."""
    app.dependency_overrides[get_current_user] = lambda: _FakeUser()
    old = getattr(app.state, "bm25", None)
    app.state.bm25 = None
    app.state.embed_fn = lambda t: [0.1] * 384
    try:
        resp = TestClient(app).post("/search/semantic", json={"query": "x", "top_k": 3})
        assert resp.status_code == 503
    finally:
        app.state.bm25 = old
        app.dependency_overrides.pop(get_current_user, None)


def test_generate_answer_endpoint_flags_hallucination(endpoint_client, monkeypatch):
    """POST /generate/answer with a forced-hallucination LLM stub returns
    grounded=false and surfaces the unsupported claim."""
    cands = [_cand("c1", "Apple Inc. reported annual revenue of $394 billion in fiscal 2023.")]
    monkeypatch.setattr("app.routers.search.hybrid_search", lambda **kw: cands)
    monkeypatch.setattr("app.routers.search.rerank", lambda q, c, top_k: c[:top_k])

    hallucinated = (
        "Apple Inc. reported annual revenue of $394 billion. "
        "They also acquired Netflix in 2023."
    )

    def fake_llm(prompt: str) -> str:
        # The groundedness prompt is the fact-checking one; return the unsupported claim.
        if "fact-checking" in prompt.lower():
            return "- They also acquired Netflix in 2023."
        return hallucinated

    # Story 1's provider seam wires app.state.llm_fn; here we inject a fake to exercise
    # the generation happy path through the endpoint.
    app.state.llm_fn = fake_llm

    resp = endpoint_client.post("/generate/answer?stream=false", json={"query": "What is Apple's revenue?"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["answer"] == hallucinated
    assert body["grounded"] is False
    assert body["verified"] is True
    assert any("Netflix" in c for c in body["unsupported_claims"])
    assert body["citations"] and body["citations"][0]["chunk_id"] == "c1"


def test_generate_answer_endpoint_honest_error_when_generation_not_configured(endpoint_client, monkeypatch):
    """Story 0 BUG 1: with no generation backend wired (app.state.llm_fn is None), the
    endpoint returns an HONEST error — HTTP 200, answer=null, error=generation_not_configured
    — with REAL citations from retrieval, instead of silently returning an empty answer."""
    cands = [_cand("c1", "Apple Inc. reported annual revenue of $394 billion in fiscal 2023.")]
    monkeypatch.setattr("app.routers.search.hybrid_search", lambda **kw: cands)
    monkeypatch.setattr("app.routers.search.rerank", lambda q, c, top_k: c[:top_k])

    # Prove no generation is attempted even if generate_answer were reachable.
    gen_spy = MagicMock(side_effect=AssertionError("generate_answer must not be called"))
    monkeypatch.setattr("app.routers.search.generate_answer", gen_spy)

    # endpoint_client fixture already sets app.state.llm_fn = None
    resp = endpoint_client.post("/generate/answer?stream=false", json={"query": "What is Apple's revenue?"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["answer"] is None
    assert body["error"] == "generation_not_configured"
    assert body["grounded"] is False
    assert body["confidence"] == 0.0
    assert body["verified"] is False
    # Citations are real — retrieval succeeded.
    assert body["citations"] and body["citations"][0]["chunk_id"] == "c1"
    gen_spy.assert_not_called()


def test_generate_answer_endpoint_with_stub_provider(endpoint_client, monkeypatch):
    """Story 1: with the StubProvider wired onto app.state.llm_fn, /generate/answer
    returns a real (non-empty) answer plus citations — the retrieval half runs normally
    and generation uses the stub (no live API)."""
    from app.generation.stub import StubProvider

    cands = [_cand("c1", "Apple Inc. reported annual revenue of $394 billion in fiscal 2023.")]
    monkeypatch.setattr("app.routers.search.hybrid_search", lambda **kw: cands)
    monkeypatch.setattr("app.routers.search.rerank", lambda q, c, top_k: c[:top_k])

    stub = StubProvider()
    stub.set_response("Apple Inc.'s revenue was $394 billion in fiscal 2023.")
    app.state.llm_fn = lambda prompt: stub.generate(prompt)

    resp = endpoint_client.post("/generate/answer?stream=false", json={"query": "What is Apple's revenue?"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["answer"] == "Apple Inc.'s revenue was $394 billion in fiscal 2023."
    assert body["answer"]  # non-empty
    assert body["error"] is None
    assert body["verified"] is True  # groundedness actually ran (stub, not fail-closed)
    assert body["citations"] and body["citations"][0]["chunk_id"] == "c1"
    # the stub was invoked for BOTH generation and groundedness verification
    assert len(stub.calls) == 2


def test_generate_answer_endpoint_passes_llm_fn_through_to_both(endpoint_client, monkeypatch):
    """The endpoint must pass app.state.llm_fn INTO generate_answer AND
    groundedness_check — there is no hardcoded internal fallback."""
    cands = [_cand("c1", "Apple Inc. revenue was $394 billion.")]
    monkeypatch.setattr("app.routers.search.hybrid_search", lambda **kw: cands)
    monkeypatch.setattr("app.routers.search.rerank", lambda q, c, top_k: c[:top_k])

    sentinel = lambda prompt: "stub answer"  # noqa: E731 — identity is what we assert
    app.state.llm_fn = sentinel

    seen = {}

    def gen_spy(query, chunks, llm_fn=None, doc_chunks_fetcher=None, **kwargs):
        seen["generate"] = llm_fn
        return "stub answer"

    def ground_spy(query, draft, chunks, llm_fn=None, answer_task=None):
        seen["groundedness"] = llm_fn
        return {"grounded": True, "confidence": 1.0, "unsupported_claims": [], "verified": True}

    monkeypatch.setattr("app.routers.search.generate_answer", gen_spy)
    monkeypatch.setattr("app.routers.search.groundedness_check", ground_spy)

    resp = endpoint_client.post("/generate/answer?stream=false", json={"query": "What is Apple's revenue?"})
    assert resp.status_code == 200, resp.text
    # exact object identity: the endpoint forwarded app.state.llm_fn to both, not a fallback
    assert seen["generate"] is sentinel
    assert seen["groundedness"] is sentinel


# ── Verifier false-positive regressions (found by the neural eval run) ────────

def test_groundedness_ignores_verifier_lines_that_affirm_support():
    """A verifier asked for a LIST often answers in prose whose bullets AFFIRM support
    ("The claim that X is supported by the source, but ..."). Counting such a line as
    an unsupported claim inverts its meaning and collapsed a correct one-line answer
    to confidence 0.0. This is the verbatim response captured from the live run."""
    raw = (
        'Unsupported claims:\n'
        '- The claim that the GX-4200 is an "outdoor sensor unit" is supported by the '
        'source, but the citation "[1, 2]" is inaccurate because the source excerpts '
        'do not use a numbering system.\n'
        '- The claim that the firmware version is supported'
    )
    source = _source_chunk("The recommended firmware for the GX-4200 is version 4.0.2.")
    result = groundedness_check(
        query="what firmware version is recommended for the GX-4200",
        answer_draft="The recommended firmware version for the GX-4200 is 4.0.2 [1, 2].",
        source_chunks=[source],
        llm_fn=lambda p: raw,
    )
    assert result["grounded"] is True
    assert result["confidence"] > 0.0
    assert result["unsupported_claims"] == []


def test_groundedness_still_catches_a_real_unsupported_claim():
    """The affirm-skip must not swallow genuine findings."""
    source = _source_chunk("The recommended firmware for the GX-4200 is version 4.0.2.")
    result = groundedness_check(
        query="what does the GX-4200 cost",
        answer_draft="The GX-4200 costs 500 dollars.",
        source_chunks=[source],
        llm_fn=lambda p: (
            "Unsupported claims:\n- The claim that the GX-4200 costs 500 dollars is "
            "not supported by the sources."
        ),
    )
    assert result["grounded"] is False
    assert result["unsupported_claims"]


def test_verification_prompt_numbers_sources_like_the_answer_cites_them():
    """The answer cites [1], [2]; if the verifier sees an unnumbered blob it reports
    the CITATION as an unsupported claim. The excerpts must carry the same numbering."""
    seen = {}
    chunks = [_source_chunk("First source text."), _source_chunk("Second source text.")]
    groundedness_check(
        query="q", answer_draft="An answer [1].", source_chunks=chunks,
        llm_fn=lambda p: seen.setdefault("prompt", p) and "none",
    )
    prompt = seen["prompt"]
    assert "[1] First source text." in prompt
    assert "[2] Second source text." in prompt
