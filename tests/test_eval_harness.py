"""
Eval harness tests.

  1. metric math is correct on a tiny hand-verified fixture (recall@k / MRR / precision@k)
  2. the ambiguous-entity query is present and reported (pass or fail — both informative)
  3. the harness runs end to end against a REAL persistent store and returns a
     complete per-query + aggregate report
"""
import pytest

from app.eval.harness import (
    HashingBoWProvider,
    _aggregate,
    _first_hit_rank,
    evaluate_query,
    load_eval_set,
    run_eval,
)


# ── 1. Metric correctness (hand-verified) ─────────────────────────────────────

def _q(qid="q", category="general", doc="doc1"):
    return {"id": qid, "query": "?", "category": category, "expected_doc_id": doc,
            "expected_snippets": ["x"]}


def test_first_hit_rank():
    assert _first_hit_rank(["a", "b", "c"], {"b"}) == 2
    assert _first_hit_rank(["a", "b", "c"], {"a", "c"}) == 1   # first correct wins
    assert _first_hit_rank(["a", "b", "c"], {"z"}) is None


def test_metrics_hit_at_rank_1():
    r = evaluate_query(_q(), ["c1", "c2", "c3"], "doc1", expected={"c1"}, k=3)
    assert r.rank == 1
    assert r.recall_at_k == 1.0
    assert r.reciprocal_rank == 1.0
    assert r.precision_at_k == round(1 / 3, 4)   # 1 correct out of top 3
    assert r.passed is True


def test_metrics_hit_at_rank_2():
    r = evaluate_query(_q(), ["c1", "c2", "c3"], "doc1", expected={"c2"}, k=3)
    assert r.rank == 2
    assert r.recall_at_k == 1.0
    assert r.reciprocal_rank == 0.5
    assert r.precision_at_k == round(1 / 3, 4)


def test_metrics_miss():
    r = evaluate_query(_q(), ["c1", "c2", "c3"], "doc1", expected={"c9"}, k=3)
    assert r.rank is None
    assert r.recall_at_k == 0.0
    assert r.reciprocal_rank == 0.0
    assert r.precision_at_k == 0.0
    assert r.passed is False


def test_metrics_two_relevant_in_topk():
    r = evaluate_query(_q(), ["c1", "c2", "c3"], "doc1", expected={"c1", "c2"}, k=3)
    assert r.rank == 1
    assert r.precision_at_k == round(2 / 3, 4)   # 2 correct out of top 3


def test_recall_bounded_by_k():
    # correct chunk sits at rank 4 but k=3 → not counted as a hit
    r = evaluate_query(_q(), ["c1", "c2", "c3", "c4"], "doc1", expected={"c4"}, k=3)
    assert r.rank is None
    assert r.recall_at_k == 0.0


def test_aggregate_hand_verified():
    a = evaluate_query(_q("a"), ["c1", "c2", "c3"], "doc1", {"c1"}, k=3)   # rr=1
    b = evaluate_query(_q("b"), ["c1", "c2", "c3"], "doc1", {"c2"}, k=3)   # rr=0.5
    c = evaluate_query(_q("c"), ["c1", "c2", "c3"], "doc1", {"c9"}, k=3)   # miss
    agg = _aggregate([a, b, c])
    assert agg["recall_at_k"] == round(2 / 3, 4)      # 2 of 3 hit
    assert agg["mrr"] == round((1.0 + 0.5 + 0.0) / 3, 4)
    assert agg["precision_at_k"] == round((1 / 3 + 1 / 3 + 0) / 3, 4)


def test_flagged_categories_marked():
    amb = evaluate_query(_q(category="ambiguous_entity"), ["c1"], "doc1", {"c1"}, k=1)
    ident = evaluate_query(_q(category="identifier"), ["c1"], "doc1", {"c1"}, k=1)
    gen = evaluate_query(_q(category="general"), ["c1"], "doc1", {"c1"}, k=1)
    assert amb.flagged and ident.flagged and not gen.flagged


# ── Surrogate provider sanity ─────────────────────────────────────────────────

def test_surrogate_is_deterministic_and_unit_norm():
    p = HashingBoWProvider()
    import numpy as np
    v1 = p.embed(["apple revenue billion"])[0]
    v2 = p.embed(["apple revenue billion"])[0]
    assert v1 == v2                               # deterministic
    assert len(v1) == 384
    assert abs(np.linalg.norm(v1) - 1.0) < 1e-5   # L2-normalized
    assert p.semantic is False


# ── 2. Fixture contains the required flagged cases ────────────────────────────

def test_fixture_has_ambiguous_and_identifier_queries():
    es = load_eval_set()
    cats = {q["category"] for q in es["queries"]}
    assert "ambiguous_entity" in cats
    assert "identifier" in cats
    assert len(es["queries"]) >= 12
    assert len(es["documents"]) >= 2
    # The specific Apple Inc. vs apple fruit ambiguity is present.
    ids = {q["id"] for q in es["queries"]}
    assert "q-apple-inc-revenue" in ids


# ── 3. End-to-end against the real store ──────────────────────────────────────

@pytest.fixture(scope="module")
def report():
    return run_eval(k=5)


def test_end_to_end_report_is_complete(report):
    es = load_eval_set()
    assert report["harness"] == "dense_only"
    assert report["k"] == 5
    assert report["num_documents"] == len(es["documents"])
    assert report["num_queries"] == len(es["queries"])
    assert len(report["queries"]) == len(es["queries"])

    agg = report["aggregate"]
    for key in ("recall_at_k", "mrr", "precision_at_k"):
        assert 0.0 <= agg[key] <= 1.0

    # Every query resolved to at least one real stored chunk (ground truth by text).
    for q in report["queries"]:
        assert q["expected_chunk_ids"], f"{q['id']} resolved no ground-truth chunk"
        assert q["rank"] is None or q["rank"] >= 1
        assert 0.0 <= q["recall_at_k"] <= 1.0
        assert 0.0 <= q["precision_at_k"] <= 1.0

    # Per-category breakdown present, including the flagged ones.
    assert "ambiguous_entity" in report["by_category"]
    assert "identifier" in report["by_category"]
    assert report["by_category"]["ambiguous_entity"]["flagged"] is True


def test_end_to_end_flags_ambiguous_entity_query(report):
    ids = {q["id"] for q in report["queries"]}
    assert "q-apple-inc-revenue" in ids

    # The flagged summary reports the ambiguous/identifier cases explicitly.
    flagged_ids = {q["id"] for q in report["flagged_summary"]["queries"]}
    assert "q-apple-inc-revenue" in flagged_ids
    apple = next(q for q in report["queries"] if q["id"] == "q-apple-inc-revenue")
    assert apple["flagged"] is True
    assert "passed" in apple and "rank" in apple   # result is reported either way


def test_end_to_end_provider_labeled(report):
    prov = report["provider"]
    assert "semantic" in prov and isinstance(prov["semantic"], bool)
    assert prov["note"]   # always explains what the numbers mean


# ── 4. Phase 8 three-mode comparison ──────────────────────────────────────────

class _StubReranker:
    """Deterministic reranker for CI: score = # of shared lowercased words
    (no cross-encoder download). Enough to exercise the rerank code path."""
    def predict(self, pairs):
        out = []
        for q, doc in pairs:
            out.append(float(len(set(q.lower().split()) & set(doc.lower().split()))))
        return out


@pytest.fixture(scope="module")
def comparison():
    from app.eval.harness import run_comparison
    return run_comparison(k=5, reranker=_StubReranker())


def test_comparison_runs_all_three_modes(comparison):
    from app.eval.harness import format_comparison

    es = load_eval_set()
    assert comparison["harness"] == "phase8_comparison"
    assert comparison["num_documents"] == len(es["documents"])
    assert comparison["num_queries"] == len(es["queries"])
    assert set(comparison["modes"]) == {"dense_only", "hybrid", "hybrid_rerank"}

    for mode in comparison["modes"].values():
        agg = mode["aggregate"]
        for key in ("recall_at_k", "mrr", "precision_at_k"):
            assert 0.0 <= agg[key] <= 1.0
        assert len(mode["queries"]) == len(es["queries"])
        # flagged-only aggregate is present (the identifier/ambiguous cut)
        assert "flagged_aggregate" in mode

    # The comparison renders to a markdown table with all three column headers.
    md = format_comparison(comparison)
    assert "dense-only" in md and "hybrid" in md and "hybrid+rerank" in md
    assert "## Aggregate" in md


def test_comparison_hybrid_and_rerank_do_not_hurt_recall(comparison):
    """Adding BM25 + rerank must not drop recall below the dense-only baseline —
    the whole point is to lift ranking (MRR) without losing correct chunks."""
    dense = comparison["modes"]["dense_only"]["aggregate"]["recall_at_k"]
    hybrid = comparison["modes"]["hybrid"]["aggregate"]["recall_at_k"]
    rerank = comparison["modes"]["hybrid_rerank"]["aggregate"]["recall_at_k"]
    assert hybrid >= dense
    assert rerank >= dense
