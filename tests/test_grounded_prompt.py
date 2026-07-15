"""
Story 2 — The Grounded Prompt.

Hermetic tests (StubProvider / injected llm_fn only, no live API):
  * build_grounded_prompt labels sources [1][2][3] with real filenames + pages,
    specifies the [n] format, gives an abstention clause, and puts the query
    after QUESTION:.
  * parse_citations handles adjacent / comma / range / duplicate / out-of-range.
  * is_abstention fuzzy-matches the refusal phrase.
  * /generate/answer returns cited_sources (only the [n] the model used) vs
    all_sources (everything fed), plus abstained.
  * the old bare "cite the source:" prompt is gone.
  * /search/semantic is unaffected (no prompt building).
"""
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from app.core.dependencies import get_current_user
from app.generation.citation_parser import is_abstention, parse_citations
from app.generation.prompt_builder import (
    ABSTENTION_PHRASE,
    build_grounded_prompt,
)
from app.main import app
from app.pipeline.bm25_index import BM25Index
from app.pipeline.retrieval import ScoredChunk


def _chunk(cid, text, source="test.pdf", page=1, use_metadata=True):
    meta = {"source": source, "page_number": page} if use_metadata else {}
    return ScoredChunk(
        chunk_id=cid, text=text, doc_id="doc1", source=source, page_number=page,
        fused_score=0.5, dense_rank=1, bm25_rank=1, reranker_score=1.0, metadata=meta,
    )


# ── build_grounded_prompt ─────────────────────────────────────────────────────

def test_build_grounded_prompt_labels_sources_1_based():
    chunks = [
        _chunk("c1", "First chunk text.", source="a.pdf", page=1),
        _chunk("c2", "Second chunk text.", source="b.docx", page=3),
        _chunk("c3", "Third chunk text.", source="c.pdf", page=45),
    ]
    prompt = build_grounded_prompt("What is the revenue?", chunks)

    # 1-based labels with correct filename + page from metadata
    assert "[1] (a.pdf, p.1)" in prompt
    assert "[2] (b.docx, p.3)" in prompt
    assert "[3] (c.pdf, p.45)" in prompt
    # chunk text is present under its label
    assert "First chunk text." in prompt
    assert "Second chunk text." in prompt
    # no 0-based labeling leaked through
    assert "[0]" not in prompt
    # the query appears after the QUESTION: marker
    q_idx = prompt.index("QUESTION:")
    assert "What is the revenue?" in prompt[q_idx:]
    # the grounding contract is present (wording rewritten in the generation-quality
    # fix, but the [n]-marker citation format + abstention clause are preserved)
    assert "[n] markers" in prompt
    assert ABSTENTION_PHRASE in prompt


def test_build_grounded_prompt_exact_label_format():
    """A chunk with source='report.pdf', page_number=7 → '[1] (report.pdf, p.7)'."""
    prompt = build_grounded_prompt("q", [_chunk("c", "body", source="report.pdf", page=7)])
    assert "[1] (report.pdf, p.7)" in prompt


def test_build_grounded_prompt_falls_back_to_scoredchunk_fields():
    """When metadata is empty, the label uses the ScoredChunk.source/page_number
    fields (which hybrid_search populates from that same metadata)."""
    prompt = build_grounded_prompt(
        "q", [_chunk("c", "body", source="fromfield.pdf", page=12, use_metadata=False)]
    )
    assert "[1] (fromfield.pdf, p.12)" in prompt


# ── parse_citations ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected", [
    ("The answer is X [1] and Y [2].", [1, 2]),
    ("Adjacent [1][2] markers.", [1, 2]),
    ("Comma group [1, 2].", [1, 2]),
    ("A range [1-3] here.", [1, 2, 3]),
    ("Duplicates [1] then [1] then [2].", [1, 2]),
    ("No markers at all.", []),
    ("", []),
])
def test_parse_citations_extracts_indices(text, expected):
    clean, indices = parse_citations(text)
    assert indices == expected
    assert clean == text  # markers preserved for display


def test_parse_citations_flags_out_of_range():
    """[99] against a 3-source set is invalid — dropped, never raised."""
    clean, indices = parse_citations("Claim [99] and [2].", num_chunks=3)
    assert 99 not in indices
    assert indices == [2]


def test_parse_citations_mixed_shapes():
    clean, indices = parse_citations("[1][2] and [3, 4] plus [5-7] and dup [1].")
    assert indices == [1, 2, 3, 4, 5, 6, 7]


# ── is_abstention ─────────────────────────────────────────────────────────────

def test_is_abstention_matches_exact_and_rephrased():
    assert is_abstention(ABSTENTION_PHRASE) is True
    assert is_abstention("Sorry, I don't have enough information about that.") is True
    assert is_abstention("That is not contained in the provided documents.") is True
    # curly apostrophe variant
    assert is_abstention("I don’t have enough information here.") is True


def test_is_abstention_false_for_real_answer():
    assert is_abstention("Apple's revenue was $394 billion. [1]") is False
    assert is_abstention("") is False


# ── Dead-code hygiene ─────────────────────────────────────────────────────────

def test_old_bare_prompt_is_gone():
    """The old unlabeled 'cite the source:' prompt must be fully removed from app/."""
    app_dir = Path(__file__).resolve().parent.parent / "app"
    offenders = [
        str(py) for py in app_dir.rglob("*.py")
        if "cite the source:" in py.read_text(encoding="utf-8")
    ]
    assert not offenders, f"old bare prompt still present in: {offenders}"


# ── Endpoint integration ──────────────────────────────────────────────────────

class _FakeUser:
    id = "u_story2"
    email = "story2@example.com"


def _cand(cid, text, source="m.pdf", page=1):
    return ScoredChunk(
        chunk_id=cid, text=text, doc_id="docA", source=source, page_number=page,
        fused_score=0.5, dense_rank=1, bm25_rank=1, reranker_score=1.0,
    )


@pytest.fixture
def endpoint_client():
    app.dependency_overrides[get_current_user] = lambda: _FakeUser()
    prev_bm25 = getattr(app.state, "bm25", None)
    prev_embed = getattr(app.state, "embed_fn", None)
    prev_llm = getattr(app.state, "llm_fn", None)
    app.state.bm25 = BM25Index(persist_dir=tempfile.mkdtemp(prefix="bm25_s2_"))
    app.state.embed_fn = lambda text: [0.1] * 384
    app.state.llm_fn = None
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.state.bm25 = prev_bm25
        app.state.embed_fn = prev_embed
        app.state.llm_fn = prev_llm


def test_generate_answer_reports_cited_vs_all_sources(endpoint_client, monkeypatch):
    """StubProvider cites [1] and [2] over a 3-chunk set with claims that RESTATE
    the chunks → both pass Story 3 Tier-1 validation → cited_sources is exactly
    chunks 1 and 2; all_sources has all three; abstained=false.

    (The claims mirror the chunk text so validation clears them lexically without
    an LLM judge — this test targets the cited-vs-all split, not validation drops;
    those are exercised in test_citation_validation.py.)"""
    from app.generation.stub import StubProvider

    cands = [
        _cand("c1", "Apple Inc. revenue was $394 billion.", source="a.pdf", page=1),
        _cand("c2", "iPhone was 52 percent of revenue.", source="b.pdf", page=2),
        _cand("c3", "Unrelated orchard content.", source="c.pdf", page=3),
    ]
    monkeypatch.setattr("app.routers.search.hybrid_search", lambda **kw: cands)
    monkeypatch.setattr("app.routers.search.rerank", lambda q, c, top_k: c[:top_k])

    stub = StubProvider()
    stub.set_response("Apple Inc. revenue was $394 billion [1]. iPhone was 52 percent of revenue [2].")
    app.state.llm_fn = lambda prompt: stub.generate(prompt)

    resp = endpoint_client.post("/generate/answer?stream=false", json={"query": "What is Apple's revenue?"})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["answer"] == "Apple Inc. revenue was $394 billion [1]. iPhone was 52 percent of revenue [2]."
    assert body["abstained"] is False
    cited_ids = [c["chunk_id"] for c in body["cited_sources"]]
    all_ids = [c["chunk_id"] for c in body["all_sources"]]
    assert cited_ids == ["c1", "c2"]        # cited AND validated
    assert all_ids == ["c1", "c2", "c3"]    # everything fed to the model


def test_generate_answer_detects_abstention(endpoint_client, monkeypatch):
    """StubProvider returns the abstention phrase → abstained=true, cited_sources empty."""
    from app.generation.stub import StubProvider

    cands = [_cand("c1", "Some content."), _cand("c2", "Other content.")]
    monkeypatch.setattr("app.routers.search.hybrid_search", lambda **kw: cands)
    monkeypatch.setattr("app.routers.search.rerank", lambda q, c, top_k: c[:top_k])

    stub = StubProvider()
    stub.set_response(ABSTENTION_PHRASE)
    app.state.llm_fn = lambda prompt: stub.generate(prompt)

    resp = endpoint_client.post("/generate/answer?stream=false", json={"query": "unanswerable?"})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["abstained"] is True
    assert body["cited_sources"] == []
    assert len(body["all_sources"]) == 2  # sources are still surfaced for transparency


def test_semantic_search_unaffected_by_story2(endpoint_client, monkeypatch):
    """/search/semantic builds no prompt and never touches the LLM — no cited/abstained
    fields, pure retrieval."""
    cands = [_cand("c1", "Apple Inc. revenue was $394 billion.")]
    monkeypatch.setattr("app.routers.search.hybrid_search", lambda **kw: cands)
    monkeypatch.setattr("app.routers.search.rerank", lambda q, c, top_k: c[:top_k])
    llm = MagicMock()
    app.state.llm_fn = llm

    resp = endpoint_client.post("/search/semantic", json={"query": "Apple revenue", "top_k": 5})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] == 1
    assert body["results"][0]["chunk_id"] == "c1"
    assert "cited_sources" not in body
    assert "abstained" not in body
    llm.assert_not_called()
