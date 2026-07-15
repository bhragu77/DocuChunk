"""
Generation-quality fix (Part 3) — query classification + prompt hints.

Pure-heuristic tests (no LLM, no network):
  * classify_query buckets navigational / definitional / analytical / informational.
  * heading-style inputs (short, title-case, no question words) → NAVIGATIONAL.
  * the classifier runs in well under 1ms (it's a string heuristic).
  * build_grounded_prompt appends the right QUERY-TYPE HINT for each type, and
    nothing extra for INFORMATIONAL.
"""
import time

import pytest

from app.generation.prompt_builder import build_grounded_prompt
from app.generation.query_classifier import (
    QueryType,
    classify_query,
    hint_for,
)
from app.pipeline.retrieval import ScoredChunk


def _chunk(cid="c1", text="body", source="m.pdf", page=1):
    return ScoredChunk(
        chunk_id=cid, text=text, doc_id="d", source=source, page_number=page,
        fused_score=0.5, dense_rank=1, bm25_rank=1, reranker_score=1.0,
        metadata={"source": source, "page_number": page},
    )


# ── classify_query ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("query,expected", [
    # From the spec's worked examples:
    ("Operating Temperature", QueryType.NAVIGATIONAL),
    ("What is the GX-4200?", QueryType.DEFINITIONAL),
    ("Compare the GX-4200 and HX-9000", QueryType.ANALYTICAL),
    ("How do I reset the firmware?", QueryType.INFORMATIONAL),
    # Extra coverage:
    ("What are the safety limits?", QueryType.DEFINITIONAL),
    ("Define torque", QueryType.DEFINITIONAL),
    ("Explain the calibration process", QueryType.DEFINITIONAL),
    ("Difference between AC and DC modes", QueryType.ANALYTICAL),
    ("GX-4200 vs HX-9000", QueryType.ANALYTICAL),
    ("Pros and cons of the fast setting", QueryType.ANALYTICAL),
    ("Chapter 3", QueryType.NAVIGATIONAL),
    ("Section 2.1", QueryType.NAVIGATIONAL),
    ("SAFETY WARNINGS", QueryType.NAVIGATIONAL),
    ("Where is the reset button located?", QueryType.INFORMATIONAL),
    ("", QueryType.INFORMATIONAL),
])
def test_classify_query_buckets(query, expected):
    assert classify_query(query) is expected


def test_heading_style_inputs_are_navigational():
    """Short, title-case, no question words → NAVIGATIONAL (the heading-as-query
    problem the hint is meant to fix)."""
    for heading in ["Operating Temperature", "Battery Maintenance", "Power Supply Unit"]:
        assert classify_query(heading) is QueryType.NAVIGATIONAL


def test_question_words_defeat_navigational():
    """A short title-case phrase that IS a question is not a heading."""
    assert classify_query("Is Operating Temperature High?") is not QueryType.NAVIGATIONAL


def test_classifier_is_fast_no_llm():
    """Pure heuristic: classifying a query is essentially free (<1ms)."""
    q = "Compare the operating temperature of the GX-4200 and the HX-9000 units"
    start = time.perf_counter()
    for _ in range(1000):
        classify_query(q)
    per_call_ms = (time.perf_counter() - start) / 1000 * 1000
    assert per_call_ms < 1.0, f"classify_query too slow: {per_call_ms:.4f}ms/call"


# ── prompt hints ──────────────────────────────────────────────────────────────

def test_navigational_query_appends_summarize_hint():
    prompt = build_grounded_prompt(
        "Operating Temperature", [_chunk()], query_type=QueryType.NAVIGATIONAL
    )
    assert "QUERY-TYPE HINT:" in prompt
    assert "looking for a specific section" in prompt
    assert "do not just repeat it" in prompt


def test_definitional_query_appends_define_hint():
    prompt = build_grounded_prompt(
        "What is the GX-4200?", [_chunk()], query_type=QueryType.DEFINITIONAL
    )
    assert "QUERY-TYPE HINT:" in prompt
    assert "wants a clear explanation of this concept" in prompt


def test_analytical_query_appends_comparison_hint():
    prompt = build_grounded_prompt(
        "Compare A and B", [_chunk()], query_type=QueryType.ANALYTICAL
    )
    assert "QUERY-TYPE HINT:" in prompt
    assert "comparison or analysis" in prompt


def test_informational_query_appends_no_extra_hint():
    """INFORMATIONAL adds NO hint — the base instructions are sufficient."""
    prompt = build_grounded_prompt(
        "How do I reset the firmware?", [_chunk()], query_type=QueryType.INFORMATIONAL
    )
    assert "QUERY-TYPE HINT:" not in prompt
    assert hint_for(QueryType.INFORMATIONAL) == ""


def test_no_query_type_means_no_hint():
    """Omitting query_type (the pure Story-2 call path) appends no hint."""
    prompt = build_grounded_prompt("anything", [_chunk()])
    assert "QUERY-TYPE HINT:" not in prompt
