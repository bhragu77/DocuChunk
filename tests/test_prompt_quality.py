"""
Generation-quality fix (Part 2) — the rewritten grounded prompt.

Confirms the new synthesis-first instructions are in place while the Story 2
citation contract is preserved:
  * new prompt contains the "Explain, don't copy" directive.
  * the old "Be concise. Every sentence must have at least one [n] citation."
    parroting phrasing is GONE from app/.
  * source labels still follow [n] (filename, p.N).
  * the abstention clause is still present.
"""
from pathlib import Path

from app.generation.prompt_builder import (
    ABSTENTION_PHRASE,
    build_grounded_prompt,
)
from app.pipeline.retrieval import ScoredChunk


def _chunk(cid, text, source="test.pdf", page=1):
    return ScoredChunk(
        chunk_id=cid, text=text, doc_id="doc1", source=source, page_number=page,
        fused_score=0.5, dense_rank=1, bm25_rank=1, reranker_score=1.0,
        metadata={"source": source, "page_number": page},
    )


def test_new_prompt_says_explain_dont_copy():
    prompt = build_grounded_prompt("What is X?", [_chunk("c1", "body")])
    assert "EXPLAIN, don't copy" in prompt
    # the synthesis / multi-source / definition directives are all present
    assert "synthesize" in prompt.lower()
    assert "MULTIPLE sources" in prompt
    assert "provide a clear DEFINITION" in prompt
    # the "your knowledge helps you EXPLAIN, not supplement" distinction
    assert "helps you EXPLAIN the source material" in prompt


def test_old_concise_parroting_phrasing_is_gone():
    """The old terse 'Be concise. Every sentence must have at least one [n]
    citation.' instruction — the parroting driver — must be removed from app/."""
    app_dir = Path(__file__).resolve().parent.parent / "app"
    needles = (
        "Be concise. Every sentence must have at least one [n] citation.",
        "You are a document QA assistant.",
    )
    for needle in needles:
        offenders = [
            str(py) for py in app_dir.rglob("*.py")
            if needle in py.read_text(encoding="utf-8")
        ]
        assert not offenders, f"old parroting phrasing {needle!r} still in: {offenders}"


def test_citation_contract_preserved():
    """Story 2 contract: [n] (filename, p.N) labels + [n] markers + abstention."""
    chunks = [
        _chunk("c1", "First.", source="a.pdf", page=1),
        _chunk("c2", "Second.", source="b.docx", page=3),
    ]
    prompt = build_grounded_prompt("q", chunks)
    assert "[1] (a.pdf, p.1)" in prompt
    assert "[2] (b.docx, p.3)" in prompt
    assert "[n] markers" in prompt
    assert ABSTENTION_PHRASE in prompt


def test_thin_context_note_appended_when_flagged():
    prompt = build_grounded_prompt("q", [_chunk("c1", "body")], thin_context=True)
    assert "limited source material was found" in prompt


def test_thin_context_note_absent_by_default():
    prompt = build_grounded_prompt("q", [_chunk("c1", "body")])
    assert "limited source material was found" not in prompt
