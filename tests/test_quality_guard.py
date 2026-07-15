"""
Generation-quality fix (Part 5) — the anti-verbatim guard (monitoring only).

  * a 100%-copied answer → verbatim_ratio 1.0, quality_warning "high_verbatim".
  * a fully rephrased answer → verbatim_ratio near 0, no warning.
  * the guard never mutates the answer text.
"""
from app.generation.quality_guard import check_verbatim
from app.pipeline.retrieval import ScoredChunk


def _chunk(text, cid="c1"):
    return ScoredChunk(
        chunk_id=cid, text=text, doc_id="d", source="s.pdf", page_number=1,
        fused_score=0.5, dense_rank=1, bm25_rank=1, reranker_score=1.0,
        metadata={"source": "s.pdf", "page_number": 1},
    )


_SOURCE = (
    "The GX-4200 industrial controller regulates internal temperature using a "
    "closed-loop feedback system that samples the thermistor every 200 milliseconds."
)


def test_copy_pasted_answer_flags_high_verbatim():
    """An answer lifted verbatim from a chunk → ratio 1.0 + high_verbatim."""
    report = check_verbatim(_SOURCE, [_chunk(_SOURCE)])
    assert report.verbatim_ratio == 1.0
    assert report.quality_warning == "high_verbatim"
    assert report.verbatim_sentence_count == report.total_sentences == 1
    assert report.flagged_sentences  # carries the matching source for debugging


def test_rephrased_answer_has_low_ratio_no_warning():
    """A genuine rephrasing shares few contiguous tokens → low ratio, no warning."""
    rephrased = (
        "It keeps its own heat steady by continuously checking a heat sensor five "
        "times per second and adjusting accordingly [1]."
    )
    report = check_verbatim(rephrased, [_chunk(_SOURCE)])
    assert report.verbatim_ratio < 0.5
    assert report.quality_warning is None


def test_guard_does_not_modify_answer():
    """Monitoring only — check_verbatim returns a report, never a changed answer."""
    answer = _SOURCE
    _ = check_verbatim(answer, [_chunk(_SOURCE)])
    assert answer == _SOURCE  # untouched


def test_citation_markers_ignored_in_comparison():
    """[n] markers in the answer don't count as copied (or non-copied) tokens."""
    answer = _SOURCE.rstrip(".") + " [1]."
    report = check_verbatim(answer, [_chunk(_SOURCE)])
    assert report.verbatim_ratio == 1.0


def test_empty_answer_is_not_verbatim():
    report = check_verbatim("", [_chunk(_SOURCE)])
    assert report.verbatim_ratio == 0.0
    assert report.quality_warning is None


def test_threshold_and_warning_ratio_are_tunable():
    """Explicit thresholds override config."""
    partial = (
        "The GX-4200 industrial controller regulates internal temperature, though "
        "the exact sampling interval is configured separately by the operator [1]."
    )
    strict = check_verbatim(partial, [_chunk(_SOURCE)], threshold=0.99, warning_ratio=0.5)
    assert strict.verbatim_ratio == 0.0  # nothing hits a 99% contiguous copy
