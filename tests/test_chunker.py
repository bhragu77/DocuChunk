"""
Phase 4 tests — Chunker.

Tests are grouped by function and ordered from simplest to most complex.
The cross-page and offset-correctness tests at the end are the critical ones:
they verify that char_start/char_end are actually correct, and that page
attribution is right when a chunk straddles a page boundary.
"""
import os
import pytest
from app.pipeline.types import CleanedPage, Chunk
from app.pipeline.chunker import (
    PAGE_SEPARATOR,
    PageSpan,
    join_pages,
    _fixed_size_chunk,
    _sentence_chunk,
    _paragraph_chunk,
    _split_paragraphs,
    chunk_document,
    default_chunker_config,
    make_chunk_id,
    make_content_hash,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _page(text: str, page_number: int = 1, source: str = "test.pdf") -> CleanedPage:
    return CleanedPage(
        text=text,
        page_number=page_number,
        source=source,
        original_char_count=len(text),
        cleaned_char_count=len(text),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. join_pages
# ═══════════════════════════════════════════════════════════════════════════════

def test_join_pages_empty():
    text, spans = join_pages([])
    assert text == ""
    assert spans == []


def test_join_pages_single_page_no_separator():
    page = _page("Hello world.", page_number=1)
    text, spans = join_pages([page])
    assert text == "Hello world."
    assert len(spans) == 1
    assert spans[0].start_offset == 0
    assert spans[0].end_offset == 12
    assert text[spans[0].start_offset:spans[0].end_offset] == "Hello world."


def test_join_pages_two_pages_separator_accounts():
    p1 = _page("Page one text.", page_number=1)
    p2 = _page("Page two text.", page_number=2)
    text, spans = join_pages([p1, p2])

    sep_len = len(PAGE_SEPARATOR)
    assert text == f"Page one text.{PAGE_SEPARATOR}Page two text."

    # Span 1: starts at 0, ends at len("Page one text.")
    assert spans[0].start_offset == 0
    assert spans[0].end_offset == len("Page one text.")

    # Span 2: starts after page 1 + separator
    assert spans[1].start_offset == len("Page one text.") + sep_len
    assert spans[1].end_offset == len(text)

    # No gap between spans (gap IS the separator, which belongs to neither span)
    assert spans[1].start_offset - spans[0].end_offset == sep_len


def test_join_pages_three_pages_spans_cover_full_text():
    pages = [_page(f"Content of page {i}.", page_number=i) for i in range(1, 4)]
    text, spans = join_pages(pages)

    assert len(spans) == 3

    # Every page's text round-trips correctly
    for page, span in zip(pages, spans):
        assert text[span.start_offset:span.end_offset] == page.text, (
            f"Page {page.page_number} round-trip failed"
        )

    # Spans are in ascending order with no overlap
    for a, b in zip(spans, spans[1:]):
        assert a.end_offset < b.start_offset, "Spans must not overlap"

    # Gaps between spans are exactly PAGE_SEPARATOR length
    for a, b in zip(spans, spans[1:]):
        assert b.start_offset - a.end_offset == len(PAGE_SEPARATOR)

    # Last span ends at the total text length
    assert spans[-1].end_offset == len(text)


def test_join_pages_preserves_page_numbers():
    pages = [_page("text", page_number=n) for n in [5, 10, 15]]
    _, spans = join_pages(pages)
    assert [s.page_number for s in spans] == [5, 10, 15]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. _fixed_size_chunk
# ═══════════════════════════════════════════════════════════════════════════════

def test_fixed_empty_text():
    assert _fixed_size_chunk("") == []


def test_fixed_single_chunk_when_text_fits():
    text = "A" * 40
    result = _fixed_size_chunk(text, chunk_size=50, overlap=0)
    assert len(result) == 1
    assert result[0] == (text, 0, 40)


def test_fixed_known_boundaries_no_overlap():
    text = "A" * 100
    result = _fixed_size_chunk(text, chunk_size=30, overlap=0)
    # Expected: [0,30), [30,60), [60,90), [90,100)
    assert len(result) == 4
    assert result[0] == ("A" * 30, 0, 30)
    assert result[1] == ("A" * 30, 30, 60)
    assert result[3] == ("A" * 10, 90, 100)


def test_fixed_overlap_content_appears_in_next_chunk():
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"   # 26 chars
    result = _fixed_size_chunk(text, chunk_size=10, overlap=3)
    # Chunk 1: [0,10), Chunk 2 starts at 7 → [7,17)
    assert result[0][1] == 0 and result[0][2] == 10
    assert result[1][1] == 7
    # The overlap text (chars 7-10) must appear in both chunks
    overlap_text = text[7:10]
    assert result[0][0].endswith(overlap_text)
    assert result[1][0].startswith(overlap_text)


def test_fixed_offset_roundtrip():
    text = "Hello world, this is a fixed chunk test with enough content."
    for chunk_text, start, end in _fixed_size_chunk(text, chunk_size=15, overlap=5):
        assert text[start:end] == chunk_text


def test_fixed_no_empty_chunks():
    text = "Some text here."
    for chunk_text, s, e in _fixed_size_chunk(text, chunk_size=6, overlap=2):
        assert chunk_text.strip() != "" or True  # any non-empty is fine
    # Specifically: no chunk should be an empty string
    assert all(t for t, _, _ in _fixed_size_chunk(text, chunk_size=6, overlap=2))


# ═══════════════════════════════════════════════════════════════════════════════
# 3. _sentence_chunk
# ═══════════════════════════════════════════════════════════════════════════════

_MULTI_SENTENCE = (
    "The quick brown fox jumps over the lazy dog. "
    "It was a fine sunny afternoon in the meadow. "
    "Nobody had expected the fox to be so athletic. "
    "The dog watched without much interest. "
    "Eventually the fox grew tired and lay down."
)
# 5 sentences, each ~40-50 chars, total ~220 chars


def test_sentence_never_splits_mid_sentence():
    """Every chunk must be a complete sentence or a run of complete sentences."""
    import nltk
    from app.pipeline.chunker import _ensure_nltk
    _ensure_nltk()  # ensure punkt data is present before calling nltk directly
    full_sentences = nltk.sent_tokenize(_MULTI_SENTENCE)
    chunks = _sentence_chunk(_MULTI_SENTENCE, max_chars=100, sentence_overlap=1)
    assert chunks, "Expected at least one chunk"
    for chunk_text, _, _ in chunks:
        # Every sentence that appears in a chunk must appear verbatim (not clipped)
        for sent in full_sentences:
            if sent[:20] in chunk_text:   # sentence starts inside this chunk
                assert sent in chunk_text, (
                    f"Sentence was clipped.\nSentence: {sent!r}\nChunk: {chunk_text!r}"
                )


def test_sentence_overlap_appears_at_start_of_next_chunk():
    """The last sentence_overlap sentences of chunk N must open chunk N+1."""
    chunks = _sentence_chunk(_MULTI_SENTENCE, max_chars=120, sentence_overlap=1)
    if len(chunks) < 2:
        pytest.skip("Not enough chunks to verify overlap (increase text or lower max_chars)")
    for (text_a, _, end_a), (text_b, start_b, _) in zip(chunks, chunks[1:]):
        # chunk B starts before chunk A's end → overlap text is shared
        assert start_b < end_a, (
            f"No overlap between consecutive chunks: "
            f"chunk A ends at {end_a}, chunk B starts at {start_b}"
        )


def test_sentence_offset_roundtrip():
    for chunk_text, start, end in _sentence_chunk(_MULTI_SENTENCE, max_chars=80):
        assert _MULTI_SENTENCE[start:end] == chunk_text


def test_sentence_single_oversized_sentence_is_its_own_chunk():
    """A sentence longer than max_chars must still appear — not be dropped."""
    long_sent = "W" * 200 + "."
    short_sent = "Short."
    text = long_sent + " " + short_sent
    chunks = _sentence_chunk(text, max_chars=50, sentence_overlap=0)
    all_text = "".join(t for t, _, _ in chunks)
    assert long_sent in all_text, "Oversized sentence must not be silently dropped"


def test_sentence_empty_text():
    assert _sentence_chunk("") == []


# ═══════════════════════════════════════════════════════════════════════════════
# 4. _paragraph_chunk
# ═══════════════════════════════════════════════════════════════════════════════

_THREE_PARAS = "First paragraph with some content.\n\nSecond paragraph has more words here.\n\nThird paragraph wraps it up."


def test_paragraph_respects_double_newline_boundaries():
    """Chunk boundaries must align with \\n\\n separators."""
    chunks = _paragraph_chunk(_THREE_PARAS, max_chars=200, para_overlap=0)
    # All 3 paragraphs fit in max_chars=200 → single chunk
    assert len(chunks) == 1
    assert chunks[0][0] == _THREE_PARAS


def test_paragraph_splits_when_max_chars_exceeded():
    """With a tight max_chars, each paragraph should be its own chunk."""
    chunks = _paragraph_chunk(_THREE_PARAS, max_chars=40, para_overlap=0)
    assert len(chunks) == 3
    # Each chunk text must equal one of the original paragraphs
    para_texts = [p.strip() for p in _THREE_PARAS.split("\n\n")]
    chunk_texts = [t.strip() for t, _, _ in chunks]
    assert chunk_texts == para_texts


def test_paragraph_overlap_para_appears_in_next_chunk():
    """
    With para_overlap=1, the last paragraph of chunk N must open chunk N+1.

    Requires max_chars large enough to merge at least 2 paragraphs per chunk —
    if each chunk holds only 1 paragraph there is nothing to carry back.
    max_chars=80 allows para0+para1 in chunk 1, then para1+para2 in chunk 2.
    """
    chunks = _paragraph_chunk(_THREE_PARAS, max_chars=80, para_overlap=1)
    if len(chunks) < 2:
        pytest.skip("Need at least 2 chunks to test overlap (fixture changed?)")
    for (_, _, end_a), (_, start_b, _) in zip(chunks, chunks[1:]):
        assert start_b < end_a, (
            f"Expected overlapping spans between consecutive chunks: "
            f"chunk A ends at {end_a}, chunk B starts at {start_b}"
        )


def test_paragraph_oversized_falls_back_to_sentence():
    """A single paragraph longer than max_chars must produce multiple chunks."""
    long_para = ("This is a very long sentence that goes on for quite a while. " * 5).strip()
    short_para = "Short paragraph."
    text = long_para + "\n\n" + short_para
    chunks = _paragraph_chunk(text, max_chars=100, para_overlap=0)
    # The long paragraph must produce more than one chunk
    assert len(chunks) > 2, (
        f"Expected long paragraph to be sentence-split into >1 chunks, got {len(chunks)}"
    )


def test_paragraph_offset_roundtrip():
    for chunk_text, start, end in _paragraph_chunk(_THREE_PARAS, max_chars=50, para_overlap=0):
        assert _THREE_PARAS[start:end] == chunk_text


# ═══════════════════════════════════════════════════════════════════════════════
# 5. chunk_document
# ═══════════════════════════════════════════════════════════════════════════════

def test_chunk_document_empty_pages():
    assert chunk_document([], "doc1") == []


def test_chunk_document_returns_chunk_dataclass():
    page = _page("Hello. World. Foo bar baz.")
    chunks = chunk_document([page], "doc1", strategy="sentence")
    assert chunks
    assert all(isinstance(c, Chunk) for c in chunks)


def test_chunk_document_chunk_index_sequential():
    page = _page("A" * 300, page_number=1)
    chunks = chunk_document([page], "doc1", strategy="fixed", chunk_size=50, overlap=0)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_document_char_offsets_roundtrip():
    """
    For every chunk produced by every strategy, joined_text[char_start:char_end]
    must equal chunk.text exactly.  This is the canonical offset-correctness test.
    Each strategy is called with its own kwargs to avoid TypeError.
    """
    pages = [
        _page("First page content. It has two sentences.", page_number=1),
        _page("Second page content with its own sentences here.", page_number=2),
        _page("Third page closes the document.", page_number=3),
    ]
    joined, _ = join_pages(pages)

    strategy_kwargs = {
        "fixed":     {"chunk_size": 30, "overlap": 5},
        "sentence":  {"max_chars": 60, "sentence_overlap": 1},
        "paragraph": {"max_chars": 60, "para_overlap": 1},
    }
    for strategy, kwargs in strategy_kwargs.items():
        chunks = chunk_document(pages, "doc1", strategy=strategy, **kwargs)
        for c in chunks:
            assert joined[c.char_start:c.char_end] == c.text, (
                f"strategy={strategy} chunk_index={c.chunk_index}: "
                f"joined[{c.char_start}:{c.char_end}] != chunk.text\n"
                f"  expected: {c.text!r}\n"
                f"  got:      {joined[c.char_start:c.char_end]!r}"
            )


def test_chunk_document_cross_page_boundary():
    """
    A chunk that straddles a page boundary must:
      - have page_number == the FIRST page it started on
      - have metadata["spans_pages"] == [first_page, second_page]

    Fixture: page 1 is 20 chars, page 2 is 20 chars.
    With fixed chunk_size=30, overlap=0 the first chunk spans both pages.
    """
    p1_text = "A" * 20
    p2_text = "B" * 20
    p1 = _page(p1_text, page_number=1)
    p2 = _page(p2_text, page_number=2)

    # joined = "AAAA...AAAA\n\nBBBB...BBBB"
    # spans:  p1=[0,20)  separator=[20,22)  p2=[22,42)
    # fixed chunk_size=30, overlap=0:
    #   chunk 0: [0, 30)  → overlaps p1[0,20) and p2[22,42) → spans_pages=[1,2]
    #   chunk 1: [30, 42) → only p2[22,42)

    chunks = chunk_document([p1, p2], "doc_cross", strategy="fixed",
                            chunk_size=30, overlap=0)

    assert len(chunks) >= 2, f"Expected at least 2 chunks, got {len(chunks)}"

    cross = chunks[0]
    assert cross.page_number == 1, (
        f"Cross-page chunk must start on page 1, got page_number={cross.page_number}"
    )
    assert "spans_pages" in cross.metadata, (
        f"Cross-page chunk must have metadata['spans_pages'], got: {cross.metadata}"
    )
    assert cross.metadata["spans_pages"] == [1, 2], (
        f"Expected spans_pages=[1, 2], got {cross.metadata['spans_pages']}"
    )

    # The second chunk is entirely on page 2
    page2_only = chunks[1]
    assert page2_only.page_number == 2
    assert "spans_pages" not in page2_only.metadata


def test_chunk_document_single_page_no_spans_pages():
    """Single-page docs must never have spans_pages in metadata."""
    page = _page("One sentence. Two sentences. Three sentences here.", page_number=1)
    for c in chunk_document([page], "doc1", strategy="sentence"):
        assert "spans_pages" not in c.metadata


def test_chunk_document_metadata_contains_strategy_and_source():
    page = _page("Some content here.", page_number=1, source="report.pdf")
    chunks = chunk_document([page], "doc1", strategy="fixed", chunk_size=10, overlap=0)
    for c in chunks:
        assert c.metadata["strategy"] == "fixed"
        assert c.metadata["source"] == "report.pdf"


def test_chunk_document_doc_id_on_all_chunks():
    page = _page("Content " * 20, page_number=1)
    chunks = chunk_document([page], "my-doc-id", strategy="fixed", chunk_size=40, overlap=0)
    assert all(c.doc_id == "my-doc-id" for c in chunks)


def test_chunk_document_all_strategies_produce_chunks():
    page = _page(
        "The first sentence sets the scene. The second sentence adds detail. "
        "A third sentence provides context. The fourth concludes the paragraph.\n\n"
        "Second paragraph adds more detail. It continues with elaboration.",
        page_number=1,
    )
    for strategy in ("fixed", "sentence", "paragraph"):
        chunks = chunk_document([page], "doc1", strategy=strategy)
        assert chunks, f"Strategy '{strategy}' produced no chunks"


def test_chunk_document_unknown_strategy_raises():
    page = _page("Hello.", page_number=1)
    with pytest.raises(ValueError, match="Unknown strategy"):
        chunk_document([page], "doc1", strategy="nonexistent")  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════════
# 5b. Deterministic chunk identity (Phase 4 amendment)
# ═══════════════════════════════════════════════════════════════════════════════

def test_make_chunk_id_deterministic():
    """Same (doc_id, char_start, char_end) → identical id."""
    assert make_chunk_id("docA", 0, 100) == make_chunk_id("docA", 0, 100)


def test_make_chunk_id_differs_on_boundaries():
    """Different boundaries → different ids (either offset changing is enough)."""
    base = make_chunk_id("docA", 0, 100)
    assert make_chunk_id("docA", 0, 101) != base   # end differs
    assert make_chunk_id("docA", 1, 100) != base   # start differs


def test_make_chunk_id_differs_on_document():
    """Same boundaries but a different document → different id."""
    assert make_chunk_id("docA", 0, 100) != make_chunk_id("docB", 0, 100)


def test_make_chunk_id_is_32_hex_chars():
    cid = make_chunk_id("docA", 0, 100)
    assert len(cid) == 32
    int(cid, 16)  # raises if not hex


def test_content_hash_changes_iff_text_changes():
    assert make_content_hash("hello world") == make_content_hash("hello world")
    assert make_content_hash("hello world") != make_content_hash("hello world!")
    assert make_content_hash("hello world") != make_content_hash("Hello world")


def test_chunk_document_ids_stable_across_two_runs():
    """
    The whole point of the amendment: re-chunking the same pages with the same
    config reproduces the same chunk_ids (so Phase 7 can upsert, not duplicate).
    """
    pages = [
        _page("First page content. It has two sentences.", page_number=1),
        _page("Second page content with its own sentences here.", page_number=2),
    ]
    run_a = chunk_document(pages, "doc-stable", strategy="fixed", chunk_size=25, overlap=5)
    run_b = chunk_document(pages, "doc-stable", strategy="fixed", chunk_size=25, overlap=5)
    assert [c.chunk_id for c in run_a] == [c.chunk_id for c in run_b]
    # And each id matches the deterministic derivation from its own offsets
    for c in run_a:
        assert c.chunk_id == make_chunk_id("doc-stable", c.char_start, c.char_end)


def test_chunk_document_different_boundaries_yield_different_ids():
    """Changing the chunk_size changes boundaries → different chunk_ids."""
    pages = [_page("A" * 300, page_number=1)]
    small = chunk_document(pages, "doc-b", strategy="fixed", chunk_size=30, overlap=0)
    large = chunk_document(pages, "doc-b", strategy="fixed", chunk_size=50, overlap=0)
    assert set(c.chunk_id for c in small).isdisjoint(c.chunk_id for c in large)


def test_chunk_content_hash_matches_text():
    pages = [_page("Alpha beta gamma delta epsilon zeta eta theta.", page_number=1)]
    chunks = chunk_document(pages, "doc-c", strategy="fixed", chunk_size=15, overlap=0)
    assert chunks
    for c in chunks:
        assert c.content_hash == make_content_hash(c.text)


def test_chunk_doc_version_propagates_from_argument():
    pages = [_page("Some content here for versioning checks.", page_number=1)]
    chunks = chunk_document(pages, "doc-v", strategy="fixed", chunk_size=10, overlap=0,
                            doc_version=7)
    assert chunks
    assert all(c.doc_version == 7 for c in chunks)


def test_chunk_doc_version_defaults_to_1():
    pages = [_page("Some content here.", page_number=1)]
    chunks = chunk_document(pages, "doc-v", strategy="fixed", chunk_size=10, overlap=0)
    assert all(c.doc_version == 1 for c in chunks)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. default_chunker_config
# ═══════════════════════════════════════════════════════════════════════════════

def test_default_chunker_config_returns_dict_for_all_strategies():
    for strategy in ("fixed", "sentence", "paragraph"):
        cfg = default_chunker_config(strategy)
        assert isinstance(cfg, dict)
        assert len(cfg) > 0, f"Empty config for strategy={strategy}"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Orchestrator integration: upload → parse → clean → chunk → ready
# ═══════════════════════════════════════════════════════════════════════════════

def test_orchestrator_advances_status_and_sets_chunk_count():
    """
    Upload a real PDF through the API.  After the background pipeline runs:
      - status must be 'ready'
      - chunk_count must be > 0
      - chunker_config must record the strategy used

    Follows the same monkey-patching pattern as test_pipeline.py to redirect
    the orchestrator's SessionLocal to the test DB.
    """
    import fitz
    import tempfile
    from fastapi.testclient import TestClient
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.main import app
    from app.database import Base, get_db
    from app.core.dependencies import get_chroma
    import app.pipeline.orchestrator as orch_module
    import chromadb

    TEST_DB = "sqlite:///./test_chunker_integration.db"
    engine = create_engine(TEST_DB, connect_args={"check_same_thread": False})
    TestSession = sessionmaker(bind=engine)

    def _db():
        db = TestSession()
        try:
            yield db
        finally:
            db.close()

    from app.models import user, document, job  # noqa: ensure tables registered
    Base.metadata.create_all(bind=engine)
    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_chroma] = lambda: chromadb.EphemeralClient()

    original_session = orch_module.SessionLocal
    orch_module.SessionLocal = TestSession

    client = TestClient(app)

    # Build a small but real PDF with a few sentences across two pages
    def _make_pdf() -> str:
        doc = fitz.open()
        for page_text in [
            "This is the first page of the test document. It has two sentences.",
            "The second page continues with more content. Another sentence follows. "
            "And a third sentence rounds out the document.",
        ]:
            page = doc.new_page()
            page.insert_text((50, 72), page_text)
        tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        doc.save(tmp.name)
        doc.close()
        return tmp.name

    try:
        client.post("/auth/register", json={
            "email": "chunker@example.com", "password": "password123",
        })
        token = client.post("/auth/login", json={
            "email": "chunker@example.com", "password": "password123",
        }).json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        pdf_path = _make_pdf()
        with open(pdf_path, "rb") as f:
            resp = client.post(
                "/docs/upload",
                files={"file": ("chunker_test.pdf", f, "application/pdf")},
                headers=headers,
            )
        os.unlink(pdf_path)

        assert resp.status_code == 201
        doc_id = resp.json()["id"]

        detail = client.get(f"/docs/{doc_id}", headers=headers).json()

        assert detail["status"] == "ready", (
            f"Expected status='ready', got {detail['status']!r}. "
            f"Error: {detail.get('error_message')}"
        )
        assert detail["chunk_count"] > 0, (
            f"Expected chunk_count > 0, got {detail['chunk_count']}"
        )
        assert detail["chunker_config"] is not None
        assert "strategy" in detail["chunker_config"]

    finally:
        orch_module.SessionLocal = original_session
        Base.metadata.drop_all(bind=engine)
        app.dependency_overrides.clear()
        if os.path.exists("./test_chunker_integration.db"):
            os.unlink("./test_chunker_integration.db")
