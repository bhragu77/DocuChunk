"""
Pipeline Stage 3 — Chunking

Splits a list of CleanedPage objects into Chunk objects ready for embedding.

Design decisions
----------------
* All character offsets (Chunk.char_start, char_end, PageSpan.start/end_offset)
  are relative to the JOINED string produced by join_pages().  They follow
  Python slice convention: text[start:end] == the text at that position.

* join_pages() inserts PAGE_SEPARATOR ("\n\n") between consecutive pages.
  The page spans never include the separator characters; chunks that straddle a
  page boundary will appear to span the separator, which is intentional — the
  separator is just whitespace so it does no harm inside a chunk.

* Splitter functions return list[tuple[str, int, int]] — (text, start, end).
  Offsets are positions WITHIN the string passed to the splitter (i.e., the
  full joined document text).  chunk_document maps those back to page numbers
  using the PageSpan index.

One-time NLTK setup (sentence strategy):
  Run once:  python -m nltk.downloader punkt_tab
  Or:        import nltk; nltk.download("punkt_tab")
  The chunker downloads this automatically on first use if missing.
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import Literal

import nltk

from app.config import get_settings
from app.pipeline.tokenization import count_tokens, split_by_tokens
from app.pipeline.types import Chunk, CleanedPage

logger = logging.getLogger(__name__)

PAGE_SEPARATOR = "\n\n"   # inserted between consecutive page texts when joining


# ── Content-addressable chunk identity (Phase 4 amendment) ────────────────────
#
# Why deterministic ids instead of uuid4(): incremental re-indexing needs stable,
# content-addressable chunk identity so a re-upload of the same logical document
# overwrites the right vectors instead of duplicating stale content into the
# retrievable set. Random UUIDs make that impossible — every run mints new ids, so
# Phase 7 can't tell "this is the same chunk, updated" from "this is a new chunk".
# Identity is fixed here at chunk-creation and preserved through embedding (Phase 6).

def make_chunk_id(doc_id: str, char_start: int, char_end: int) -> str:
    """
    Deterministic chunk id derived from document identity + location.
    Same document + same boundaries => same id across runs.

    Phase 7 uses this as the vector store's primary key so re-index is an
    upsert-by-chunk_id, not an append.
    """
    key = f"{doc_id}:{char_start}:{char_end}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def make_content_hash(text: str) -> str:
    """
    Hash of the chunk's text. Changes iff the text changes.

    Phase 7 stores this as vector metadata and skips re-embedding any chunk whose
    content_hash is unchanged across a re-index.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


_nltk_ready = False


def _ensure_nltk() -> None:
    global _nltk_ready
    if _nltk_ready:
        return
    for resource in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
            _nltk_ready = True
            return
        except LookupError:
            pass
    # Neither cached — download punkt_tab (NLTK >= 3.8) with punkt as fallback
    try:
        nltk.download("punkt_tab", quiet=True)
        _nltk_ready = True
    except Exception:
        try:
            nltk.download("punkt", quiet=True)
            _nltk_ready = True
        except Exception as exc:
            logger.warning("NLTK punkt data unavailable: %s — sentence chunker may fail", exc)


# ── Step 1: Page-boundary-aware joining ───────────────────────────────────────

@dataclass
class PageSpan:
    """
    Marks where one page's text lives within the joined multi-page string.
    Invariant: joined_text[start_offset:end_offset] == original page text.
    Offsets are exclusive-end (Python slice convention).
    The PAGE_SEPARATOR characters that sit between two adjacent spans belong
    to neither span.
    """
    page_number: int
    start_offset: int   # inclusive
    end_offset: int     # exclusive


def join_pages(pages: list[CleanedPage]) -> tuple[str, list[PageSpan]]:
    """
    Concatenate page texts with PAGE_SEPARATOR and return a PageSpan index.

    The PageSpan list lets chunk_document map any character offset in the
    joined string back to the page it originally belonged to, which is how
    Chunk.page_number and metadata["spans_pages"] are populated.

    Returns:
        joined_text:  full document string
        page_spans:   one PageSpan per input page, in page order
    """
    if not pages:
        return "", []

    spans: list[PageSpan] = []
    offset = 0
    for i, page in enumerate(pages):
        start = offset
        end = offset + len(page.text)
        spans.append(PageSpan(
            page_number=page.page_number,
            start_offset=start,
            end_offset=end,
        ))
        offset = end
        if i < len(pages) - 1:
            offset += len(PAGE_SEPARATOR)   # account for separator before next page

    joined = PAGE_SEPARATOR.join(p.text for p in pages)

    # Hard invariant: the last span's end must equal the total length.
    # A mismatch here means the separator accounting above is wrong.
    assert spans[-1].end_offset == len(joined), (
        f"PageSpan accounting bug: last span ends at {spans[-1].end_offset} "
        f"but joined text length is {len(joined)}"
    )
    return joined, spans


# ── Internal helpers ──────────────────────────────────────────────────────────

def _sent_tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    """
    Tokenize text into sentences and return (sentence, start, end) tuples
    whose positions are character offsets within the input string.

    Uses a forward-scanning find() so identical sentences resolve correctly.
    """
    _ensure_nltk()
    sentences = nltk.sent_tokenize(text)
    results: list[tuple[str, int, int]] = []
    pos = 0
    for sent in sentences:
        idx = text.find(sent, pos)
        if idx == -1:
            idx = pos   # shouldn't happen with standard punkt, safe fallback
        results.append((sent, idx, idx + len(sent)))
        pos = idx + len(sent)
    return results


def _split_paragraphs(text: str) -> list[tuple[str, int, int]]:
    """
    Split text on \\n\\n boundaries (one or more consecutive newlines count).
    Returns (paragraph_text, start_offset, end_offset) for non-empty paragraphs.
    """
    paras: list[tuple[str, int, int]] = []
    prev_end = 0
    for m in re.finditer(r'\n\n+', text):
        para_text = text[prev_end:m.start()]
        if para_text.strip():
            paras.append((para_text, prev_end, m.start()))
        prev_end = m.end()
    # Final paragraph after the last separator (or the whole text if no separator)
    if prev_end <= len(text):
        para_text = text[prev_end:]
        if para_text.strip():
            paras.append((para_text, prev_end, len(text)))
    return paras


# ── Step 2: Splitting strategies ─────────────────────────────────────────────

def _fixed_size_chunk(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[tuple[str, int, int]]:
    """
    Mechanical character-window split.
    Baseline strategy — no sentence or paragraph awareness.
    Returns (chunk_text, start_offset, end_offset) within text.
    """
    if not text:
        return []
    # Clamp overlap to prevent an infinite loop when overlap >= chunk_size
    overlap = min(overlap, chunk_size - 1)
    results: list[tuple[str, int, int]] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        results.append((text[start:end], start, end))
        if end == len(text):
            break
        start = end - overlap
    return results


def _sentence_chunk(
    text: str,
    max_chars: int = 800,
    sentence_overlap: int = 2,
) -> list[tuple[str, int, int]]:
    """
    Sentence-boundary-aware chunking.

    Accumulates sentences (by span in the original text) until including the
    next sentence would push the chunk span over max_chars, then closes the
    chunk and carries back sentence_overlap sentences into the next one.

    Never cuts mid-sentence — that is the entire value of this strategy over
    fixed.  The carried-over sentences provide context continuity across chunks.

    Returns (chunk_text, start_offset, end_offset) within text.
    """
    sents = _sent_tokenize_with_offsets(text)
    if not sents:
        return [(text, 0, len(text))] if text.strip() else []

    results: list[tuple[str, int, int]] = []
    i = 0
    while i < len(sents):
        chunk_sents: list[tuple[str, int, int]] = []
        j = i

        while j < len(sents):
            _, sent_start, sent_end = sents[j]
            # Projected span of the chunk if we include this sentence
            span_start = chunk_sents[0][1] if chunk_sents else sent_start
            projected = sent_end - span_start
            if chunk_sents and projected > max_chars:
                break
            chunk_sents.append(sents[j])
            j += 1

        if not chunk_sents:
            # Single sentence already exceeds max_chars — include it alone
            # rather than skipping it (would cause infinite loop) or truncating.
            chunk_sents = [sents[i]]
            j = i + 1

        chunk_start = chunk_sents[0][1]
        chunk_end = chunk_sents[-1][2]
        results.append((text[chunk_start:chunk_end], chunk_start, chunk_end))

        # Carry back sentence_overlap sentences: start next chunk that many back
        i = max(i + 1, j - sentence_overlap)

    return results


def _paragraph_chunk(
    text: str,
    max_chars: int = 1000,
    para_overlap: int = 1,
) -> list[tuple[str, int, int]]:
    """
    Paragraph-boundary-aware chunking.

    Merges consecutive paragraphs until the next would push the total span over
    max_chars, then closes the chunk and carries back para_overlap paragraph(s)
    into the next one.

    If a SINGLE paragraph exceeds max_chars, falls back to _sentence_chunk for
    that paragraph only.  This avoids two bad outcomes: truncating real content,
    or producing one giant unsearchable chunk from a dense, undivided block
    (e.g. a block-quoted legal clause or a packed academic abstract).

    Returns (chunk_text, start_offset, end_offset) within text.
    """
    paras = _split_paragraphs(text)
    if not paras:
        return [(text, 0, len(text))] if text.strip() else []

    results: list[tuple[str, int, int]] = []
    i = 0
    while i < len(paras):
        pt, ps, pe = paras[i]

        # Oversized single paragraph → sentence-chunk it and advance past it.
        if len(pt) > max_chars:
            for sub_text, sub_start, sub_end in _sentence_chunk(pt, max_chars=max_chars):
                abs_start = ps + sub_start
                abs_end = ps + sub_end
                results.append((text[abs_start:abs_end], abs_start, abs_end))
            i += 1
            continue

        # Accumulate paragraphs until the next would push the span over max_chars.
        chunk_paras: list[tuple[str, int, int]] = []
        j = i
        while j < len(paras):
            pt_j, ps_j, pe_j = paras[j]
            if len(pt_j) > max_chars:
                break   # oversized — handled on the next outer iteration
            if chunk_paras:
                # Span from first accumulated para to end of candidate para
                new_span = pe_j - chunk_paras[0][1]
                if new_span > max_chars:
                    break
            chunk_paras.append((pt_j, ps_j, pe_j))
            j += 1

        if chunk_paras:
            start = chunk_paras[0][1]
            end = chunk_paras[-1][2]
            results.append((text[start:end], start, end))

        i = max(i + 1, j - para_overlap)

    return results


# ── Step 3: Assembly ──────────────────────────────────────────────────────────

def chunk_document(
    pages: list[CleanedPage],
    doc_id: str,
    strategy: Literal["fixed", "sentence", "paragraph"] = "sentence",
    doc_version: int = 1,
    **kwargs,
) -> list[Chunk]:
    """
    Main entry point: CleanedPage list → Chunk list.

    Steps:
      1. join_pages → one string + PageSpan index
      2. Run the chosen splitter → list of (text, start, end) in joined_text
      3. Map each chunk's [start, end) to page(s) using the PageSpan index
      4. Wrap into Chunk dataclasses with full provenance metadata

    The char_start / char_end on every Chunk is an offset into the joined
    string: joined_text[chunk.char_start:chunk.char_end] == chunk.text.

    Chunk identity (Phase 4 amendment):
      chunk_id      = make_chunk_id(doc_id, char_start, char_end) — deterministic,
                      so re-chunking the same document with the same boundaries
                      reproduces the same ids (stable re-indexing key for Phase 7).
      content_hash  = make_content_hash(text) — changes iff the chunk text changes.
      doc_version   = the caller-supplied version of the logical document. Pass the
                      Document's current `version` here; it is copied verbatim onto
                      every chunk so Phase 7 can tombstone chunks from old versions.
    """
    if not pages:
        return []

    joined_text, page_spans = join_pages(pages)

    if strategy == "fixed":
        raw = _fixed_size_chunk(joined_text, **kwargs)
    elif strategy == "sentence":
        raw = _sentence_chunk(joined_text, **kwargs)
    elif strategy == "paragraph":
        raw = _paragraph_chunk(joined_text, **kwargs)
    else:
        raise ValueError(
            f"Unknown strategy {strategy!r}. Choose 'fixed', 'sentence', or 'paragraph'."
        )

    # Drop any empty-string chunks that slipped through
    raw = [(t, s, e) for t, s, e in raw if t.strip()]

    # ── Token guard ───────────────────────────────────────────────────────────
    # The character-based strategies above do not bound TOKENS, but the embedding
    # model truncates inputs over embed_max_tokens (256 for all-MiniLM-L6-v2).
    # Split any over-budget chunk into token-bounded sub-chunks, rebasing the
    # sub-offsets onto the joined-document offsets so char_start/char_end (and thus
    # chunk identity + provenance) stay exact. Chunks already within budget pass
    # through untouched (one cheap token count each).
    settings = get_settings()
    max_tokens = settings.embed_max_tokens
    overlap_tokens = settings.embed_overlap_tokens
    bounded: list[tuple[str, int, int]] = []
    for t, s, e in raw:
        for sub_t, sub_s, sub_e in split_by_tokens(t, max_tokens, overlap_tokens):
            bounded.append((sub_t, s + sub_s, s + sub_e))
    raw = bounded

    source = pages[0].source if pages else ""
    chunks: list[Chunk] = []

    for idx, (chunk_text, start, end) in enumerate(raw):
        # Pages whose text spans overlap with [start, end).
        # Overlap condition: start < span.end_offset  AND  end > span.start_offset
        overlapping = [
            sp.page_number for sp in page_spans
            if start < sp.end_offset and end > sp.start_offset
        ]

        page_number = overlapping[0] if overlapping else page_spans[0].page_number

        meta: dict = {
            "strategy": strategy,
            "params": kwargs,
            "source": source,
        }
        if len(overlapping) > 1:
            meta["spans_pages"] = overlapping

        chunks.append(Chunk(
            chunk_id=make_chunk_id(doc_id, start, end),
            doc_id=doc_id,
            text=chunk_text,
            chunk_index=idx,
            page_number=page_number,
            char_start=start,
            char_end=end,
            token_count=count_tokens(chunk_text),   # exact model token count (see tokenization.py)
            content_hash=make_content_hash(chunk_text),
            doc_version=doc_version,
            metadata=meta,
        ))

    logger.info(
        "chunk_document: doc=%s strategy=%s pages=%d → %d chunks",
        doc_id, strategy, len(pages), len(chunks),
    )
    return chunks


# ── Config helper ─────────────────────────────────────────────────────────────

def default_chunker_config(strategy: str) -> dict:
    """
    Returns the default kwargs for the given strategy from Settings.
    Usage: chunk_document(pages, doc_id, strategy, **default_chunker_config(strategy))
    """
    s = get_settings()
    if strategy == "fixed":
        return {"chunk_size": s.fixed_chunk_size, "overlap": s.fixed_overlap}
    if strategy == "sentence":
        return {"max_chars": s.sentence_max_chars, "sentence_overlap": s.sentence_overlap}
    if strategy == "paragraph":
        return {"max_chars": s.paragraph_max_chars, "para_overlap": s.paragraph_overlap}
    return {}
