"""
Story 3 — Citation Validation.

Hermetic: the Tier 2 LLM judge is a StubProvider / MagicMock / raising callable,
never a live API. The tests assert both the verdicts AND the cost control — a
clearly-supported citation must NOT reach the judge.

Covers:
  * SUPPORTED  — fact present in chunk → Tier 1 clears it, no LLM.
  * CONTRADICTED — asserted number absent from chunk → Tier 1 flags → judge drops.
  * NOT_MENTIONED — off-topic fact → flagged → judge → not_mentioned → dropped.
  * OUT_OF_RANGE — [99] over 3 chunks → invalid, dropped, no crash.
  * JUDGE FAILURE — Tier 2 raises → unverified (fail-open), not dropped.
  * ABSTENTION — nothing to validate → empty result.
  * Tier 1 cost control — clear support does not call the judge.
  * VALIDATE_USE_LLM=false — suspect citations become unverified, judge untouched.
  * endpoint integration — one valid + one invalid citation → response splits them,
    answer text preserved with both markers.
"""
import tempfile

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from app.core.dependencies import get_current_user
from app.generation.citation_validator import validate_citations
from app.generation.prompt_builder import ABSTENTION_PHRASE
from app.main import app
from app.pipeline.bm25_index import BM25Index
from app.pipeline.retrieval import ScoredChunk


def _chunk(cid, text, source="spec.pdf", page=1):
    return ScoredChunk(
        chunk_id=cid, text=text, doc_id="doc1", source=source, page_number=page,
        fused_score=0.5, dense_rank=1, bm25_rank=1, reranker_score=1.0,
        metadata={"source": source, "page_number": page},
    )


# The canonical spec-sheet chunk from the story's worked example.
_RATING = "The GX-4200 is rated for indoor use between 0°C and 45°C."


# ── SUPPORTED ─────────────────────────────────────────────────────────────────

def test_supported_claim_passes_tier1_without_llm():
    chunks = [_chunk("c1", _RATING)]
    answer = "The GX-4200 supports temperatures up to 45°C [1]."
    judge = MagicMock()

    result = validate_citations(answer, [1], chunks, llm_fn=judge)

    assert result.valid_indices == [1]
    assert result.dropped_indices == []
    (vc,) = result.validated_citations
    assert vc.verdict == "supported"
    assert vc.chunk_id == "c1"
    assert "45" in vc.claim_text
    judge.assert_not_called()  # every asserted fact present → no judge needed


# ── CONTRADICTED ──────────────────────────────────────────────────────────────

def test_contradicted_claim_is_flagged_by_tier1_then_dropped():
    chunks = [_chunk("c1", _RATING)]
    answer = "The GX-4200 supports temperatures up to 85°C [1]."
    # Tier 1 catches it: 85 is not in the chunk → SUSPECT → judge adjudicates.
    judge = MagicMock(return_value="contradicted")

    result = validate_citations(answer, [1], chunks, llm_fn=judge)

    assert result.dropped_indices == [1]
    assert result.valid_indices == []
    assert result.validated_citations[0].verdict == "contradicted"
    judge.assert_called_once()


# ── NOT_MENTIONED ─────────────────────────────────────────────────────────────

def test_not_mentioned_claim_is_dropped():
    chunks = [
        _chunk("c1", "The GX-4200 ships in a recyclable box."),
        _chunk("c2", _RATING),  # temperature only — nothing about weight
    ]
    answer = "The GX-4200 weighs 2.3kg [2]."
    judge = MagicMock(return_value="not_mentioned")

    result = validate_citations(answer, [2], chunks, llm_fn=judge)

    assert result.dropped_indices == [2]
    assert result.valid_indices == []
    assert result.validated_citations[0].verdict == "not_mentioned"
    judge.assert_called_once()


# ── OUT_OF_RANGE ──────────────────────────────────────────────────────────────

def test_out_of_range_citation_is_invalid_not_crashed():
    chunks = [_chunk("c1", "a"), _chunk("c2", "b"), _chunk("c3", "c")]
    answer = "A bold claim [99]."
    judge = MagicMock()

    result = validate_citations(answer, [99], chunks, llm_fn=judge)

    assert result.dropped_indices == [99]
    assert result.valid_indices == []
    assert result.unverified_indices == []
    (vc,) = result.validated_citations
    assert vc.verdict == "invalid"
    assert vc.chunk_id is None
    judge.assert_not_called()  # no chunk to check


# ── JUDGE FAILURE (fail-open) ─────────────────────────────────────────────────

def test_judge_failure_makes_citation_unverified_not_dropped():
    chunks = [_chunk("c1", _RATING)]
    answer = "The GX-4200 weighs 2.3kg [1]."  # suspect → routed to the judge

    def boom(_prompt):
        raise RuntimeError("judge backend down")

    result = validate_citations(answer, [1], chunks, llm_fn=boom)

    assert result.unverified_indices == [1]
    assert result.dropped_indices == []      # fail-open: NOT dropped
    assert result.valid_indices == []
    assert result.validated_citations[0].verdict == "unverified"


# ── ABSTENTION ────────────────────────────────────────────────────────────────

def test_abstention_skips_validation_entirely():
    chunks = [_chunk("c1", _RATING)]
    judge = MagicMock()

    result = validate_citations(ABSTENTION_PHRASE, [1], chunks, llm_fn=judge)

    assert result.validated_citations == []
    assert result.valid_indices == []
    assert result.dropped_indices == []
    assert result.unverified_indices == []
    judge.assert_not_called()


# ── Tier 1 cost control ───────────────────────────────────────────────────────

def test_tier1_prevents_unnecessary_llm_call():
    """A claim whose content words all appear in the chunk (no missing facts) is
    cleared by Tier 1 — the judge is never invoked."""
    chunks = [_chunk("c1", "The device is rated for indoor use between 0 and 45 degrees.")]
    answer = "The device is rated for indoor use [1]."
    judge = MagicMock()

    result = validate_citations(answer, [1], chunks, llm_fn=judge)

    assert result.valid_indices == [1]
    judge.assert_not_called()


# ── VALIDATE_USE_LLM=false ────────────────────────────────────────────────────

def test_use_llm_false_makes_suspect_citations_unverified():
    """With the judge disabled, a citation Tier 1 can't clear becomes unverified
    (not dropped) and no judge call is made."""
    chunks = [_chunk("c1", _RATING)]
    answer = "The GX-4200 supports temperatures up to 85°C [1]."  # suspect (85 absent)
    judge = MagicMock()

    result = validate_citations(answer, [1], chunks, llm_fn=judge, use_llm=False)

    assert result.unverified_indices == [1]
    assert result.dropped_indices == []
    assert result.valid_indices == []
    assert result.validated_citations[0].verdict == "unverified"
    judge.assert_not_called()


# ── Endpoint integration ──────────────────────────────────────────────────────

class _FakeUser:
    id = "u_story3"
    email = "story3@example.com"


def _cand(cid, text, source="m.pdf", page=1):
    return ScoredChunk(
        chunk_id=cid, text=text, doc_id="docA", source=source, page_number=page,
        fused_score=0.5, dense_rank=1, bm25_rank=1, reranker_score=1.0,
        metadata={"source": source, "page_number": page},
    )


@pytest.fixture
def endpoint_client():
    app.dependency_overrides[get_current_user] = lambda: _FakeUser()
    prev_bm25 = getattr(app.state, "bm25", None)
    prev_embed = getattr(app.state, "embed_fn", None)
    prev_llm = getattr(app.state, "llm_fn", None)
    app.state.bm25 = BM25Index(persist_dir=tempfile.mkdtemp(prefix="bm25_s3_"))
    app.state.embed_fn = lambda text: [0.1] * 384
    app.state.llm_fn = None
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.state.bm25 = prev_bm25
        app.state.embed_fn = prev_embed
        app.state.llm_fn = prev_llm


def test_endpoint_splits_valid_and_dropped_citations(endpoint_client, monkeypatch):
    """One valid + one invalid citation → cited_sources holds only the valid chunk,
    dropped_sources holds the invalid one, and the answer text keeps BOTH markers."""
    cands = [
        _cand("c1", "Cats are small mammals.", source="a.pdf", page=1),
        _cand("c2", "Birds have feathers.", source="b.pdf", page=2),
    ]
    monkeypatch.setattr("app.routers.search.hybrid_search", lambda **kw: cands)
    monkeypatch.setattr("app.routers.search.rerank", lambda q, c, top_k: c[:top_k])

    answer = "Cats are mammals [1]. Dogs can fly to the moon [2]."

    def fake_llm(prompt: str) -> str:
        # The Story 3 judge prompt is distinguishable from generation/groundedness.
        if "Does the source support this claim" in prompt:
            return "contradicted"
        return answer

    app.state.llm_fn = fake_llm

    resp = endpoint_client.post("/generate/answer?stream=false", json={"query": "tell me about cats"})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # [1] restates chunk 1 → validated; [2] is suspect → judge → contradicted → dropped.
    assert [c["chunk_id"] for c in body["cited_sources"]] == ["c1"]
    assert [c["chunk_id"] for c in body["dropped_sources"]] == ["c2"]
    # answer text is untouched — both markers survive for the frontend to flag.
    assert body["answer"] == answer
    assert "[1]" in body["answer"] and "[2]" in body["answer"]
    # validation_details records every citation, verdicts and all.
    verdicts = {d["index"]: d["verdict"] for d in body["validation_details"]}
    assert verdicts == {1: "supported", 2: "contradicted"}
