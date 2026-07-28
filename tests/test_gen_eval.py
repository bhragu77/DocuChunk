"""
Phase 11 — generation-quality eval harness tests.

Fast, hermetic unit tests over the pure scoring pieces (metrics, judges, generator,
gate, formatting), plus ONE full offline run_gen_eval integration test that ingests
the fixture through the real pipeline and asserts the deterministic gate passes.
"""
from __future__ import annotations

import pytest

from app.eval import gen_harness as gh
from app.eval import ragas_compat
from app.eval.gen_harness import (
    FAILURE_CLASSES,
    FAILURE_FLOORS_SURROGATE,
    ExtractiveGenerator,
    GenQueryResult,
    LexicalJudge,
    LexicalReranker,
    LLMGenerator,
    LLMJudge,
    check_gate,
    classify_failure,
    evaluate_gen_query,
    format_gen_report,
    get_gen_backends,
    _context_precision,
    _context_recall,
)
from app.pipeline.retrieval import ScoredChunk


def _chunk(chunk_id: str, text: str, doc_id: str = "doc1") -> ScoredChunk:
    return ScoredChunk(
        chunk_id=chunk_id, text=text, doc_id=doc_id, source="src.docx",
        page_number=1, fused_score=1.0,
    )


def _gqr(**kw) -> GenQueryResult:
    """A default-'ok' GenQueryResult; override individual signals to test the taxonomy."""
    base = dict(
        id="q", query="q", category="general", flagged=False, expected_doc_id="d",
        answer="an answer", reference=None, contexts=["ctx"], abstained=False,
        num_retrieved=5, num_relevant_retrieved=1, num_expected=1,
        faithfulness=1.0, answer_relevancy=0.9, context_precision=0.9,
        context_recall=1.0, answer_correctness=None, failure_class="ok",
    )
    base.update(kw)
    return GenQueryResult(**base)


# ── Deterministic context metrics (no model) ──────────────────────────────────

def test_context_recall_full_and_partial():
    assert _context_recall(["a", "b", "c"], {"a", "b"}) == 1.0
    assert _context_recall(["a", "x", "y"], {"a", "b"}) == 0.5
    assert _context_recall(["x"], {"a", "b"}) == 0.0
    assert _context_recall(["a"], set()) == 0.0  # no expected → 0, never a divide error


def test_context_precision_rewards_early_relevant_chunks():
    # relevant at ranks 1 and 3 → AP = mean(1/1, 2/3)
    ap = _context_precision(["b", "a", "d"], {"b", "d"})
    assert ap == round((1.0 + 2 / 3) / 2, 4)
    # relevant at ranks 2 and 4 → AP = mean(1/2, 2/4) = 0.5
    assert _context_precision(["a", "b", "c", "d"], {"b", "d"}) == 0.5
    # none retrieved → 0
    assert _context_precision(["x", "y"], {"b", "d"}) == 0.0


# ── Lexical (surrogate) judge ─────────────────────────────────────────────────

def test_lexical_faithfulness_full_vs_partial_vs_empty():
    j = LexicalJudge()
    chunks = [_chunk("c1", "The battery is part number 88-AZ-0097 for the GX-4200.")]
    # every content word of the answer is in the context → 1.0
    assert j.faithfulness("q", "part number 88-AZ-0097", chunks) == 1.0
    # half the answer's content words are absent from the context → < 1.0
    partial = j.faithfulness("q", "part number 88-AZ-0097 costs fifty dollars", chunks)
    assert 0.0 < partial < 1.0
    # empty answer → 0.0 (cannot be faithful to nothing)
    assert j.faithfulness("q", "", chunks) == 0.0


def test_lexical_relevancy_measures_question_coverage():
    j = LexicalJudge()
    q = "which replacement battery does the GX-4200 use"
    assert j.answer_relevancy(q, "the GX-4200 replacement battery is 88-AZ-0097") > 0.5
    assert j.answer_relevancy(q, "the weather is nice today") == 0.0


def test_faithful_but_offtopic_answer_scores_high_faithfulness_low_relevancy():
    """The Example-C case: a TRUE-to-source answer to the WRONG question. This is the
    axis the live groundedness check cannot see — faithfulness stays high, relevancy
    collapses. It is the core justification for building offline gen-eval."""
    j = LexicalJudge()
    query = "What was Apple Inc.'s annual revenue?"
    chunks = [_chunk(
        "c1",
        "Apple Inc. reported total annual revenue of 394 billion dollars. "
        "Apple was founded in 1976 by Steve Jobs.",
    )]
    offtopic = "Apple was founded in 1976 by Steve Jobs."
    assert j.faithfulness(query, offtopic, chunks) == 1.0   # every word is in the source
    assert j.answer_relevancy(query, offtopic) < 0.4        # but it dodged the question


# ── Extractive (surrogate) generator ──────────────────────────────────────────

def test_extractive_generator_is_grounded_and_query_focused():
    gen = ExtractiveGenerator(top_sentences=1)
    chunks = [_chunk(
        "c1",
        "The GX-4200 operates from minus 20 to 60 degrees Celsius. "
        "The replacement battery is part number 88-AZ-0097.",
    )]
    answer = gen.generate("which replacement battery does the GX-4200 use", chunks)
    # picks the battery sentence, not the temperature one
    assert "88-AZ-0097" in answer
    # grounded by construction: every content word came from the context
    assert LexicalJudge().faithfulness("q", answer, chunks) == 1.0


def test_extractive_generator_handles_empty_chunks():
    assert ExtractiveGenerator().generate("q", []) == ""


# ── Lexical reranker plugs into the real rerank() ─────────────────────────────

def test_lexical_reranker_orders_by_overlap():
    from app.pipeline.retrieval import rerank
    cands = [
        _chunk("c1", "unrelated content about weather and clouds"),
        _chunk("c2", "the GX-4200 battery replacement part number"),
    ]
    ranked = rerank("GX-4200 battery", cands, top_k=2, reranker=LexicalReranker())
    assert ranked[0].chunk_id == "c2"  # higher query overlap ranked first


# ── LLM (real) judge, driven by a fake llm_fn ─────────────────────────────────

def test_llm_judge_faithfulness_reuses_groundedness():
    # groundedness_check treats "none" as fully supported → confidence 1.0.
    judge = LLMJudge(judge_fn=lambda p: "none")
    chunks = [_chunk("c1", "Apple revenue was 394 billion dollars.")]
    assert judge.faithfulness("q", "Apple made 394 billion.", chunks) == 1.0


@pytest.mark.parametrize("raw,expected", [
    ("0.9", 0.9),
    ("Score: 0.85 out of 1.0", 0.85),
    ("1.0", 1.0),
    ("0", 0.0),
    ("the answer looks good", 0.0),   # no parseable number → 0.0
])
def test_llm_judge_relevancy_parses_score(raw, expected):
    judge = LLMJudge(judge_fn=lambda p: raw)
    assert judge.answer_relevancy("q", "an answer") == expected


def test_llm_judge_relevancy_empty_answer_is_zero_without_calling_model():
    calls = []
    judge = LLMJudge(judge_fn=lambda p: calls.append(p) or "1.0")
    assert judge.answer_relevancy("q", "   ") == 0.0
    assert calls == []  # short-circuits — no wasted model call


# ── Backend selection mirrors get_eval_provider ───────────────────────────────

def test_get_gen_backends_offline_vs_llm():
    gen, judge, semantic = get_gen_backends(None, None)
    assert isinstance(gen, ExtractiveGenerator) and isinstance(judge, LexicalJudge)
    assert semantic is False

    gen, judge, semantic = get_gen_backends(lambda p: "x", None, gen_model="gemini-x")
    assert isinstance(gen, LLMGenerator) and isinstance(judge, LLMJudge)
    assert semantic is True


# ── evaluate_gen_query wires generation + scoring together ────────────────────

def test_evaluate_gen_query_offline_end_to_end():
    q = {"id": "q1", "query": "which battery does the GX-4200 use",
         "category": "identifier", "expected_doc_id": "doc1"}
    chunks = [
        _chunk("c1", "The replacement battery is part number 88-AZ-0097."),
        _chunk("c2", "unrelated sentence about clouds"),
    ]
    gen, judge, _ = get_gen_backends(None, None)
    r = evaluate_gen_query(q, chunks, expected={"c1"}, generator=gen, judge=judge)
    assert r.num_relevant_retrieved == 1 and r.num_expected == 1
    assert r.context_recall == 1.0
    assert r.context_precision == 1.0     # relevant chunk is at rank 1
    assert r.faithfulness == 1.0
    assert not r.abstained


# ── Regression gate ───────────────────────────────────────────────────────────

def _report(**agg) -> dict:
    base = {"faithfulness": 1.0, "answer_relevancy": 0.9,
            "context_precision": 0.9, "context_recall": 1.0}
    base.update(agg)
    return {"aggregate": base, "gate_thresholds": gh.GATE_THRESHOLDS_SURROGATE}


def test_check_gate_pass():
    passed, failures = check_gate(_report())
    assert passed and failures == []


def test_check_gate_fails_when_metric_below_floor():
    passed, failures = check_gate(_report(context_recall=0.5))  # floor 0.80
    assert not passed
    assert any("context_recall" in f for f in failures)


def test_check_gate_flags_missing_metric():
    rep = _report()
    del rep["aggregate"]["faithfulness"]
    passed, failures = check_gate(rep)
    assert not passed and any("faithfulness" in f and "MISSING" in f for f in failures)


# ── Reporting ─────────────────────────────────────────────────────────────────

def test_format_gen_report_shows_gate_and_metrics():
    report = {
        "harness": "generation_quality", "k": 5, "profile": "surrogate",
        "generator": "extractive_surrogate", "judge": "lexical_surrogate",
        "embedding_provider": {"name": "local", "model": "m", "semantic": True},
        "num_documents": 10, "num_queries": 33, "num_abstained": 0,
        "aggregate": {"faithfulness": 1.0, "answer_relevancy": 0.78,
                      "context_precision": 0.92, "context_recall": 1.0},
        "flagged_aggregate": {"faithfulness": 1.0, "answer_relevancy": 0.84,
                              "context_precision": 0.95, "context_recall": 1.0},
        "by_category": {"identifier": {"n": 15, "flagged": True, "faithfulness": 1.0,
                        "answer_relevancy": 0.95, "context_precision": 0.94,
                        "context_recall": 1.0}},
        "failure_breakdown": {"retrieval_miss": 0, "over_refusal": 0, "hallucination": 0,
                              "off_topic": 0, "partial_answer": 2, "ok": 31},
        "failure_rates": {"retrieval_miss": 0.0, "over_refusal": 0.0, "hallucination": 0.0,
                          "off_topic": 0.0, "partial_answer": 0.061, "ok": 0.939},
        "gate_thresholds": gh.GATE_THRESHOLDS_SURROGATE,
        "gate_max_rates": {"hallucination": 0.0, "retrieval_miss": 0.10},
        "note": "surrogate note",
    }
    report["aggregate"]["answer_correctness"] = 0.54
    md = format_gen_report(report)
    assert "Generation-quality baseline" in md
    assert "PASS" in md and "Gate:" in md
    assert "answer-relevancy" in md and "context-recall" in md
    # new sections: correctness row, failure breakdown table + routing legend
    assert "answer-correctness" in md
    assert "Failure breakdown" in md and "routes to" in md
    for cls in FAILURE_CLASSES:
        assert cls in md


# ── Full offline integration run (ingests the fixture through the real pipeline) ──

def test_run_gen_eval_offline_gate_passes():
    # Derive the expected size from the fixture rather than hardcoding it, so growing
    # the eval set (e.g. adding the multi_hop queries) is not a test failure.
    n_queries = len(gh.load_eval_set(gh.DEFAULT_EVAL_SET)["queries"])

    report = gh.run_gen_eval(k=5)  # llm_fn=None → offline surrogate path
    assert report["profile"] == "surrogate"
    assert report["num_queries"] == n_queries
    # deterministic metrics are meaningful even offline — retrieval must be intact.
    assert report["aggregate"]["context_recall"] >= 0.9
    # every query carries a gold answer now → correctness is scored fixture-wide.
    assert report["aggregate"]["answer_correctness"] is not None
    # failure taxonomy partitions the whole fixture; extractive gen never hallucinates.
    assert sum(report["failure_breakdown"].values()) == n_queries
    assert report["failure_rates"]["hallucination"] == 0.0
    # retrieval_miss is allowed a small budget (the corpus carries near-duplicate
    # distractors by design); the gate ceiling is the contract, not zero.
    assert report["failure_rates"]["retrieval_miss"] <= gh.GATE_MAX_RATES["retrieval_miss"]
    passed, failures = check_gate(report)
    assert passed, failures


# ── Failure taxonomy (Phase 11.5) ─────────────────────────────────────────────

def test_classify_failure_each_class():
    f = FAILURE_FLOORS_SURROGATE
    assert classify_failure(_gqr(context_recall=0.0), f) == "retrieval_miss"
    assert classify_failure(_gqr(abstained=True), f) == "over_refusal"
    assert classify_failure(_gqr(faithfulness=0.2), f) == "hallucination"
    assert classify_failure(_gqr(faithfulness=1.0, answer_relevancy=0.1), f) == "off_topic"
    assert classify_failure(_gqr(answer_relevancy=0.4), f) == "partial_answer"
    assert classify_failure(_gqr(), f) == "ok"


def test_classify_failure_precedence_retrieval_beats_refusal():
    # no evidence retrieved AND the model abstained → the root cause is retrieval.
    f = FAILURE_FLOORS_SURROGATE
    assert classify_failure(_gqr(context_recall=0.0, abstained=True), f) == "retrieval_miss"


def test_offtopic_example_c_classifies_as_off_topic():
    # the Apple 'founded' answer to the 'revenue' question: faithful, not relevant.
    r = _gqr(faithfulness=1.0, answer_relevancy=0.2)
    assert classify_failure(r, FAILURE_FLOORS_SURROGATE) == "off_topic"


# ── answer_correctness (surrogate token-F1) ───────────────────────────────────

def test_lexical_answer_correctness_f1():
    j = LexicalJudge()
    ref = "Apple revenue was 394 billion dollars"
    assert j.answer_correctness("q", "Apple revenue was 394 billion dollars", ref) == 1.0
    assert j.answer_correctness("q", "the weather is sunny", ref) == 0.0
    partial = j.answer_correctness("q", "Apple revenue 394", ref)
    assert 0.0 < partial < 1.0
    assert j.answer_correctness("q", "", ref) == 0.0


# ── Gate: failure-rate ceilings ───────────────────────────────────────────────

def test_check_gate_fails_when_hallucination_rate_exceeds_ceiling():
    report = {
        "aggregate": {"faithfulness": 1.0, "answer_relevancy": 0.9,
                      "context_precision": 0.9, "context_recall": 1.0},
        "gate_thresholds": gh.GATE_THRESHOLDS_SURROGATE,
        "gate_max_rates": {"hallucination": 0.0, "retrieval_miss": 0.10},
        "failure_rates": {"hallucination": 0.1, "retrieval_miss": 0.0},
    }
    passed, failures = check_gate(report)
    assert not passed and any("hallucination_rate" in f for f in failures)


# ── RAGAS interop ─────────────────────────────────────────────────────────────

def _mini_report() -> dict:
    return {
        "aggregate": {"faithfulness": 0.9, "answer_relevancy": 0.8,
                      "context_precision": 0.7, "context_recall": 1.0,
                      "answer_correctness": 0.6},
        "queries": [
            {"query": "who runs Pear?", "answer": "Dana Whitfield.",
             "contexts": ["Pear CEO is Dana Whitfield."], "reference": "Dana Whitfield."},
            {"query": "no-ref q", "answer": "x", "contexts": ["c"], "reference": None},
        ],
    }


def test_to_ragas_dataset_schema():
    rows = ragas_compat.to_ragas_dataset(_mini_report())
    assert len(rows) == 2
    assert set(rows[0]) >= {"question", "answer", "contexts", "reference"}
    assert isinstance(rows[0]["contexts"], list) and rows[0]["contexts"]
    assert "reference" not in rows[1]  # omitted when the query had no gold answer


def test_export_ragas_format_renames_and_drops_nulls():
    out = ragas_compat.export_ragas_format(_mini_report())
    assert out["faithfulness"] == 0.9 and out["answer_correctness"] == 0.6
    # a None aggregate metric is dropped, not exported as null
    rep = _mini_report()
    rep["aggregate"]["answer_correctness"] = None
    assert "answer_correctness" not in ragas_compat.export_ragas_format(rep)


def test_run_ragas_crosscheck_degrades_gracefully_without_ragas():
    res = ragas_compat.run_ragas_crosscheck(_mini_report())
    assert "available" in res
    if not res["available"]:
        assert "note" in res and "ragas" in res["note"].lower()




# ── Provider pacing / transient-failure retry (what makes a live run survivable) ──

def test_pacer_classifies_transient_vs_real_errors():
    """A 40-minute neural run against a free tier WILL hit rate limits and 5xx blips.
    Those must be retried; a real bug must not be."""
    p = gh._SharedPacer
    for msg in ("429 RESOURCE_EXHAUSTED", "504 DEADLINE_EXCEEDED",
                "503 UNAVAILABLE", "500 Internal error"):
        assert p._is_transient(msg), msg
    for msg in ("400 INVALID_ARGUMENT", "KeyError: chunk_id", "API key not valid"):
        assert not p._is_transient(msg), msg


def test_pacer_retries_transient_then_succeeds(monkeypatch):
    monkeypatch.setattr(gh.time, "sleep", lambda _s: None)  # no real waiting
    calls = {"n": 0}

    def flaky(_prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("504 DEADLINE_EXCEEDED")
        return "recovered"

    wrapped = gh._SharedPacer(rpm=600).wrap(flaky)
    assert wrapped("p") == "recovered"
    assert calls["n"] == 2


def test_pacer_does_not_retry_real_errors(monkeypatch):
    monkeypatch.setattr(gh.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    def broken(_prompt: str) -> str:
        calls["n"] += 1
        raise ValueError("400 INVALID_ARGUMENT")

    wrapped = gh._SharedPacer(rpm=600).wrap(broken)
    with pytest.raises(ValueError):
        wrapped("p")
    assert calls["n"] == 1, "a non-transient error must not be retried"
