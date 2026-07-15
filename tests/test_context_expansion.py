"""
Story 5 — Context Expansion.

Hermetic. The merge algorithm is exercised on synthetic char spans over a fixed,
NON-periodic document string so `merged == doc[a:b]` is an airtight dedup check.
Prompt/endpoint integration uses injected fetchers and a StubProvider — no live
API and no reliance on a real Chroma corpus (except the one metadata-fix test,
which uses a throwaway persistent client).
"""
import random
import string
import tempfile
from types import SimpleNamespace

import chromadb
import pytest
from fastapi.testclient import TestClient

from app.core.dependencies import get_current_user
from app.generation.context_expander import (
    ChunkMeta,
    apply_token_budget,
    estimate_tokens,
    expand_context,
)
from app.generation.prompt_builder import build_grounded_prompt
from app.main import app
from app.pipeline.bm25_index import BM25Index
from app.pipeline.chunker import Chunk, make_chunk_id, make_content_hash
from app.pipeline.embedder import EmbeddedChunk
from app.pipeline.retrieval import ScoredChunk
from app.pipeline.vector_store import batch_upsert_sink, get_collection

# A deterministic, NON-periodic 600-char document. Non-periodic matters: it lets
# the gap test assert "the skipped span does NOT reappear" without coincidental
# substring collisions that a repeating alphabet would produce.
_rng = random.Random(42)
_DOC = "".join(_rng.choice(string.ascii_letters + string.digits) for _ in range(600))


def _cm(idx: int, cs: int, ce: int) -> ChunkMeta:
    return ChunkMeta(chunk_id=f"c{idx}", chunk_index=idx, doc_id="d",
                     char_start=cs, char_end=ce, text=_DOC[cs:ce])


def _target(idx: int, cs: int, ce: int, source="src.pdf", page=1) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=f"c{idx}", text=_DOC[cs:ce], doc_id="d", source=source,
        page_number=page, fused_score=1.0,
        metadata={"chunk_index": idx, "char_start": cs, "char_end": ce,
                  "doc_id": "d", "source": source, "page_number": page},
    )


# ── MERGE ALGORITHM (pure) ────────────────────────────────────────────────────

def test_two_chunks_overlap_deduplicated():
    """Chunks 0 (0–200) and 1 (100–300) share 100 chars → merged is doc[0:300],
    length = sum - overlap, with the repeated span removed exactly once."""
    a, b = _cm(0, 0, 200), _cm(1, 100, 300)
    ec = expand_context(_target(0, 0, 200), [a, b], window=1)

    assert ec.merged_text == _DOC[0:300]
    assert len(ec.merged_text) == 300 == len(a.text) + len(b.text) - 100
    assert ec.chunk_indices == [0, 1]
    assert ec.char_range == (0, 300)
    assert ec.original_target_index == 0


def test_three_chunks_both_overlaps_removed():
    a, b, c = _cm(0, 0, 200), _cm(1, 100, 300), _cm(2, 250, 450)
    ec = expand_context(_target(1, 100, 300), [a, b, c], window=1)

    assert ec.merged_text == _DOC[0:450]
    assert len(ec.merged_text) == 450
    assert ec.chunk_indices == [0, 1, 2]
    assert ec.char_range == (0, 450)


def test_chunk_at_document_start_truncates_window():
    """Target index 0 with window 1 → only indices 0 and 1, no wrap-around."""
    a, b, c = _cm(0, 0, 200), _cm(1, 100, 300), _cm(2, 250, 450)
    ec = expand_context(_target(0, 0, 200), [a, b, c], window=1)

    assert ec.chunk_indices == [0, 1]
    assert ec.merged_text == _DOC[0:300]


def test_chunk_at_document_end_truncates_window():
    a, b, c = _cm(0, 0, 200), _cm(1, 100, 300), _cm(2, 250, 450)
    ec = expand_context(_target(2, 250, 450), [a, b, c], window=1)

    assert ec.chunk_indices == [1, 2]
    assert ec.merged_text == _DOC[100:450]


def test_non_overlapping_chunks_preserve_gap():
    """A gap between char_end and the next char_start is kept as a visible seam —
    the skipped document span is NOT fabricated into the merged text."""
    a, b = _cm(0, 0, 100), _cm(1, 150, 250)
    ec = expand_context(_target(0, 0, 100), [a, b], window=1)

    assert _DOC[0:100] in ec.merged_text
    assert _DOC[150:250] in ec.merged_text
    assert _DOC[100:150] not in ec.merged_text  # gap not invented
    assert ec.char_range == (0, 250)


def test_single_chunk_returns_original_text():
    ec = expand_context(_target(0, 0, 100), [_cm(0, 0, 100)], window=1)

    assert ec.merged_text == _DOC[0:100]
    assert ec.chunk_indices == [0]


def test_missing_offsets_degrades_to_target_text():
    """A pre-Story-5 chunk (no char offsets in metadata) can't be merged — it
    degrades to its own text rather than crashing."""
    target = ScoredChunk(
        chunk_id="c0", text="lonely fragment", doc_id="d", source="s.pdf",
        page_number=1, fused_score=1.0, metadata={"chunk_index": 0},  # no offsets
    )
    ec = expand_context(target, [], window=1)
    assert ec.merged_text == "lonely fragment"
    assert ec.chunk_indices == [0]


# ── PROMPT INTEGRATION ────────────────────────────────────────────────────────

def _fetcher(_doc_id):
    return [_cm(0, 0, 200), _cm(1, 100, 300), _cm(2, 250, 450)]


def test_prompt_uses_expanded_text_but_original_label():
    target = _target(1, 100, 300, source="report.pdf", page=7)
    settings = SimpleNamespace(
        context_expansion_enabled=True, context_expansion_window=1, context_max_tokens=3000,
    )
    prompt = build_grounded_prompt("q", [target], doc_chunks_fetcher=_fetcher, settings=settings)

    # Label still names the ORIGINAL chunk's source/page (the citation target)...
    assert "[1] (report.pdf, p.7)" in prompt
    # ...but the text block is the widened passage (doc[0:450]), longer than the
    # target fragment alone (doc[100:300]).
    assert _DOC[0:450] in prompt
    assert len(_DOC[0:450]) > len(target.text)


def test_prompt_disabled_uses_original_text_only():
    target = _target(1, 100, 300, source="report.pdf", page=7)
    settings = SimpleNamespace(
        context_expansion_enabled=False, context_expansion_window=1, context_max_tokens=3000,
    )
    prompt = build_grounded_prompt("q", [target], doc_chunks_fetcher=_fetcher, settings=settings)

    assert target.text in prompt              # the raw fragment
    assert _DOC[0:100] not in prompt          # neighbor-only region absent → no expansion
    # Identical to the pre-Story-5 call path (no fetcher).
    assert prompt == build_grounded_prompt("q", [target])


# ── TOKEN BUDGET ──────────────────────────────────────────────────────────────

def test_token_budget_under_limit_untouched():
    exps = ["x" * 40, "y" * 40]  # ~10 + ~10 tokens
    assert apply_token_budget(exps, max_tokens=100) == exps


def test_token_budget_preserves_top_rank_trims_lower():
    exps = ["A" * 400, "B" * 400, "C" * 400]  # ~100 tokens each, 300 total
    result = apply_token_budget(exps, max_tokens=150)

    assert result[0] == exps[0]                       # highest-ranked preserved fully
    assert len(result[-1]) < len(exps[-1])            # lowest-ranked trimmed
    assert "(truncated)" in result[-1]                # middle-truncation marker
    assert sum(estimate_tokens(t) for t in result) < sum(estimate_tokens(t) for t in exps)


# ── METADATA FIX ──────────────────────────────────────────────────────────────

def test_chroma_metadata_includes_char_offsets(tmp_path):
    """A newly ingested chunk carries char_start/char_end into Chroma metadata,
    readable back via collection.get — the prerequisite for expansion."""
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    coll = get_collection(client, "u")

    chunk = Chunk(
        chunk_id=make_chunk_id("d", 10, 60), doc_id="d", text="hello world",
        chunk_index=3, page_number=1, char_start=10, char_end=60,
        token_count=2, content_hash=make_content_hash("hello world"), doc_version=1,
        metadata={"source": "f.pdf", "strategy": "sentence"},
    )
    batch_upsert_sink(coll)([EmbeddedChunk(chunk=chunk, embedding=[0.1] * 384)])

    got = coll.get(include=["metadatas"])
    meta = got["metadatas"][0]
    assert meta["char_start"] == 10
    assert meta["char_end"] == 60
    assert meta["chunk_index"] == 3


# ── END-TO-END ────────────────────────────────────────────────────────────────

class _FakeUser:
    id = "u_story5"
    email = "story5@example.com"


@pytest.fixture
def endpoint_client():
    app.dependency_overrides[get_current_user] = lambda: _FakeUser()
    prev_bm25 = getattr(app.state, "bm25", None)
    prev_embed = getattr(app.state, "embed_fn", None)
    prev_llm = getattr(app.state, "llm_fn", None)
    app.state.bm25 = BM25Index(persist_dir=tempfile.mkdtemp(prefix="bm25_s5_"))
    app.state.embed_fn = lambda text: [0.1] * 384
    app.state.llm_fn = None
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.state.bm25 = prev_bm25
        app.state.embed_fn = prev_embed
        app.state.llm_fn = prev_llm


def test_end_to_end_expansion_feeds_merged_context(endpoint_client, monkeypatch):
    """A question spanning a chunk boundary: retrieval returns only the middle
    fragment, but expansion merges its neighbors so the model's prompt contains
    the full range — and the stub's citation validates against the source."""
    frag0 = "Intro paragraph about the GX-4200 device family. "
    frag1 = "The GX-4200 is designed for extended operation in environments ranging from "
    frag2 = "0C to 45C, with brief excursions to 55C permitted under warranty."
    doc = frag0 + frag1 + frag2
    cs0, ce0 = 0, len(frag0)
    cs1, ce1 = ce0, ce0 + len(frag1)
    cs2, ce2 = ce1, ce1 + len(frag2)
    assert doc[cs2:ce2] == frag2  # offsets are honest

    target = ScoredChunk(
        chunk_id="mid", text=frag1, doc_id="docE", source="spec.pdf", page_number=4,
        fused_score=1.0,
        metadata={"chunk_index": 1, "char_start": cs1, "char_end": ce1,
                  "doc_id": "docE", "source": "spec.pdf", "page_number": 4},
    )

    monkeypatch.setattr("app.routers.search.hybrid_search", lambda **kw: [target])
    monkeypatch.setattr("app.routers.search.rerank", lambda q, c, top_k: c[:top_k])
    monkeypatch.setattr(
        "app.routers.search.fetch_doc_chunks",
        lambda chroma, uid, doc_id: [
            ChunkMeta("c0", 0, "docE", cs0, ce0, frag0),
            ChunkMeta("mid", 1, "docE", cs1, ce1, frag1),
            ChunkMeta("c2", 2, "docE", cs2, ce2, frag2),
        ],
    )

    prompts: list[str] = []

    def fake_llm(prompt: str) -> str:
        prompts.append(prompt)
        if "Does the source support this claim" in prompt:  # citation judge
            return "supported"
        return "The operating range is 0C to 45C [1]."

    app.state.llm_fn = fake_llm

    resp = endpoint_client.post(
        "/generate/answer?stream=false",
        json={"query": "What is the GX-4200's full operating temperature range?"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # The GENERATION prompt (first llm call) contains the merged passage — the
    # neighbor text ("0C to 45C") that the retrieved fragment alone did not have.
    gen_prompt = prompts[0]
    assert "0C to 45C" in gen_prompt
    assert frag0 in gen_prompt and frag2 in gen_prompt

    # Citation still resolves to the retrieved chunk and survives validation.
    assert [c["chunk_id"] for c in body["cited_sources"]] == ["mid"]
    assert body["answer"] == "The operating range is 0C to 45C [1]."
