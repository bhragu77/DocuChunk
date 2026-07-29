"""
Tests for MMR diversification and multi-query expansion.

Both are opt-in retrieval enhancements, so the tests focus on the properties that
decide whether enabling them is safe: they must degrade to the existing behaviour on
failure, they must never fail a query, and MMR must actually trade relevance for
coverage rather than silently reducing to plain reranking.
"""
from __future__ import annotations

import pytest

from app.pipeline.mmr import mmr_rerank
from app.pipeline.query_rewriter import (
    _parse_rewrites,
    clear_cache,
    fuse_ranked_lists,
    rewrite_query,
)
from app.pipeline.retrieval import ScoredChunk


def _c(cid, text, rr=None, fused=0.5) -> ScoredChunk:
    return ScoredChunk(chunk_id=cid, text=text, doc_id="d1", source="f.pdf",
                       page_number=1, fused_score=fused, reranker_score=rr)


# Two clusters: a/b/c near-identical, x/y distinct. Cosine on these is unambiguous.
def _embed(text: str) -> list[float]:
    return {"a": [1.0, 0.0], "b": [0.99, 0.01], "c": [0.98, 0.02],
            "x": [0.0, 1.0], "y": [0.02, 0.99]}[text]


# ── MMR ───────────────────────────────────────────────────────────────────────

def test_mmr_breaks_up_a_block_of_near_duplicates():
    """The failure this exists to fix: five restatements of one fact filling the
    budget, so a multi-hop question never sees its second fact."""
    cands = [_c("1", "a", rr=9.0), _c("2", "b", rr=8.9), _c("3", "c", rr=8.8),
             _c("4", "x", rr=8.0), _c("5", "y", rr=7.9)]
    out = mmr_rerank(cands, _embed, top_k=3, lambda_mult=0.5)
    texts = [c.text for c in out]
    assert texts[0] == "a", "most relevant chunk must still lead"
    assert "x" in texts, "a distinct chunk must be pulled into the budget"


def test_lambda_one_is_a_no_op():
    cands = [_c("1", "a", rr=9.0), _c("2", "b", rr=8.9), _c("3", "x", rr=1.0)]
    assert [c.chunk_id for c in mmr_rerank(cands, _embed, top_k=2, lambda_mult=1.0)] \
        == ["1", "2"]


def test_low_lambda_favours_diversity_over_relevance():
    cands = [_c("1", "a", rr=9.0), _c("2", "b", rr=8.9), _c("3", "x", rr=0.1)]
    out = mmr_rerank(cands, _embed, top_k=2, lambda_mult=0.1)
    assert [c.chunk_id for c in out] == ["1", "3"]


def test_scores_are_never_mutated():
    """MMR changes order and selection only, so per-stage score reporting stays
    truthful about what produced them."""
    cands = [_c("1", "a", rr=9.0), _c("2", "x", rr=8.0)]
    out = mmr_rerank(cands, _embed, top_k=2, lambda_mult=0.5)
    assert out[0].reranker_score == 9.0
    assert all(c.fused_score == 0.5 for c in out)


def test_falls_back_to_input_order_when_embedding_fails():
    """Diversity is an enhancement; it must never fail the query."""
    def boom(_t):
        raise RuntimeError("embedder down")

    cands = [_c("1", "a", rr=9.0), _c("2", "b", rr=8.0), _c("3", "x", rr=7.0)]
    assert [c.chunk_id for c in mmr_rerank(cands, boom, top_k=2)] == ["1", "2"]


def test_empty_and_undersized_inputs():
    assert mmr_rerank([], _embed, top_k=5) == []
    cands = [_c("1", "a", rr=9.0)]
    assert len(mmr_rerank(cands, _embed, top_k=5)) == 1


def test_unbounded_reranker_scores_do_not_swamp_the_diversity_term():
    """Cross-encoder scores are unbounded logits; cosine is [-1,1]. Without
    normalisation the relevance term dominates at any lambda and MMR degenerates
    into plain reranking."""
    cands = [_c("1", "a", rr=500.0), _c("2", "b", rr=499.0), _c("3", "x", rr=498.0)]
    out = mmr_rerank(cands, _embed, top_k=2, lambda_mult=0.5)
    assert [c.chunk_id for c in out] == ["1", "3"]


# ── Query rewriting ───────────────────────────────────────────────────────────

def test_original_query_is_always_first_and_present():
    clear_cache()
    out = rewrite_query("what battery", lambda p: "which cell\nwhat power source", n=2)
    assert out[0] == "what battery"
    assert len(out) == 3


def test_llm_failure_degrades_to_the_original_query():
    clear_cache()

    def boom(_p):
        raise RuntimeError("provider down")

    assert rewrite_query("what battery", boom) == ["what battery"]


def test_list_markers_and_duplicates_are_stripped():
    clear_cache()
    raw = "1. which cell\n- which cell\n* what power source\n\n"
    out = rewrite_query("what battery", lambda p: raw, n=5)
    assert out == ["what battery", "which cell", "what power source"]


def test_rewrites_are_capped_at_n():
    clear_cache()
    raw = "\n".join(f"variant {i}" for i in range(20))
    assert len(rewrite_query("q", lambda p: raw, n=3)) == 4


def test_echoing_the_original_is_not_counted_twice():
    clear_cache()
    out = rewrite_query("what battery", lambda p: "What Battery\nwhich cell", n=3)
    assert out == ["what battery", "which cell"]


def test_result_is_cached_so_a_repeat_does_not_pay_twice():
    clear_cache()
    calls = {"n": 0}

    def llm(_p):
        calls["n"] += 1
        return "which cell"

    rewrite_query("q", llm, n=1)
    rewrite_query("q", llm, n=1)
    assert calls["n"] == 1


def test_parse_ignores_noise_lines():
    assert _parse_rewrites("\n\n  \nok phrasing\n", "orig", 3) == ["ok phrasing"]


# ── RRF fusion ────────────────────────────────────────────────────────────────

def test_fusion_rewards_agreement_across_phrasings():
    """An ID ranked well by several phrasings should beat one ranked first by a
    single phrasing — that agreement is the entire signal expansion adds."""
    fused = fuse_ranked_lists([["a", "b"], ["b", "a"], ["b", "c"]])
    assert fused[0] == "b"


def test_fusion_handles_disjoint_lists():
    assert set(fuse_ranked_lists([["a"], ["b"]])) == {"a", "b"}


def test_fusion_of_nothing_is_empty():
    assert fuse_ranked_lists([]) == []
    assert fuse_ranked_lists([[], []]) == []
