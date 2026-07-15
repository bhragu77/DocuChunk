"""
Story 5 — Context Expansion.

Retrieval returns fragment-sized chunks. A question whose answer straddles a
chunk boundary ("...ranging from" | "0°C to 45°C...") can't be answered from
either fragment alone. This module widens each retrieved chunk to include its
NEIGHBORS (chunk_index ± window within the same document) and stitches them into
one continuous passage.

THE OVERLAP PROBLEM: Phase 4 built chunks with sentence_overlap, so consecutive
chunks share their boundary sentences. Naive concatenation repeats them. We avoid
that by merging on the stored char_start/char_end spans (offsets into the joined
full-document text, where joined_text[char_start:char_end] == chunk.text): the
overlap is a numeric span intersection, sliced out by offset arithmetic — never
by fragile string matching.

Everything here is pure (no I/O). fetch_doc_chunks() is the one Chroma-touching
helper, kept separate so the merge algorithm stays trivially testable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a runtime import cycle with retrieval.py
    from app.pipeline.retrieval import ScoredChunk

logger = logging.getLogger(__name__)

# Inserted where two included spans are NOT contiguous (a gap in the document
# between them). We keep a visible seam rather than fabricate the missing text.
_GAP_SEP = " … "

# Rough token estimate — len//4, matching Chunk.token_count's convention. Not a
# real tokenizer; good enough for a budget cap.
def estimate_tokens(text: str) -> int:
    return len(text) // 4


@dataclass
class ChunkMeta:
    """Minimal per-chunk provenance the merge needs. Built from Chroma metadata
    (see fetch_doc_chunks) — one entry per chunk in a document."""
    chunk_id: str
    chunk_index: int
    doc_id: str
    char_start: int
    char_end: int
    text: str


@dataclass
class ExpandedContext:
    merged_text: str            # deduplicated, continuous passage
    chunk_indices: list[int]    # chunk_indices included, ascending (e.g. [13,14,15])
    char_range: tuple[int, int] # merged char_start..char_end span
    original_target_index: int  # the retrieved chunk's index (for prompt placement)


# ── The merge algorithm (pure) ────────────────────────────────────────────────

def _merge_spans(spans: list[ChunkMeta]) -> tuple[str, tuple[int, int]]:
    """Interval-merge char spans into one continuous string.

    Precondition: spans sorted by char_start. Because char offsets index the same
    joined document text, the portion of the next span already covered by the
    running span is exactly (running_end - next.char_start) characters — we slice
    that many chars off the FRONT of next.text to drop the repeated overlap. A
    fully-contained span contributes nothing; a gap (next starts after running
    end) is joined with a visible separator instead of inventing the missing text.
    """
    if not spans:
        return "", (0, 0)

    first = spans[0]
    parts: list[str] = [first.text]
    running_end = first.char_end
    max_end = first.char_end
    start = first.char_start

    for nxt in spans[1:]:
        if nxt.char_start >= running_end:
            # Contiguous (==) → seamless join; gap (>) → visible seam.
            if nxt.char_start > running_end:
                parts.append(_GAP_SEP)
            parts.append(nxt.text)
            running_end = nxt.char_end
        else:
            # Overlap: skip the already-covered prefix of nxt.text.
            overlap = running_end - nxt.char_start
            if nxt.char_end > running_end:  # nxt extends past current end
                parts.append(nxt.text[overlap:])
                running_end = nxt.char_end
            # else nxt fully contained → add nothing
        max_end = max(max_end, nxt.char_end)

    return "".join(parts), (start, max_end)


def expand_context(
    target_chunk: "ScoredChunk",
    all_doc_chunks: list[ChunkMeta],
    window: int = 1,
) -> ExpandedContext:
    """Widen target_chunk to include ±window neighbors, merged and deduplicated.

    Near the document edges the window is simply truncated (no wrap-around). If
    the target's char offsets are missing (a document ingested before the Story 5
    metadata fix) or it has no locatable neighbors, this degrades gracefully to
    the target's own text — expansion is best-effort, never fatal.
    """
    meta = target_chunk.metadata or {}
    target_index = _as_int(meta.get("chunk_index"))

    # Without a chunk_index we can't locate neighbors — return the target as-is.
    if target_index is None:
        return _single(target_chunk, fallback_index=0)

    lo, hi = target_index - window, target_index + window
    selected = [c for c in all_doc_chunks if lo <= c.chunk_index <= hi]

    # Ensure the target itself is represented even if the fetch missed it.
    if not any(c.chunk_index == target_index for c in selected):
        as_meta = _target_as_meta(target_chunk, target_index)
        if as_meta is None:
            return _single(target_chunk, fallback_index=target_index)
        selected.append(as_meta)

    # Any span lacking usable offsets can't be interval-merged safely → bail to
    # the target text rather than risk a bad slice.
    if any(c.char_end <= c.char_start for c in selected):
        return _single(target_chunk, fallback_index=target_index)

    selected.sort(key=lambda c: (c.char_start, c.char_end))
    merged_text, char_range = _merge_spans(selected)

    return ExpandedContext(
        merged_text=merged_text,
        chunk_indices=sorted(c.chunk_index for c in selected),
        char_range=char_range,
        original_target_index=target_index,
    )


def _single(target_chunk: "ScoredChunk", fallback_index: int) -> ExpandedContext:
    """Degenerate expansion: just the target chunk's own text."""
    meta = target_chunk.metadata or {}
    cs = _as_int(meta.get("char_start")) or 0
    ce = _as_int(meta.get("char_end"))
    if ce is None:
        ce = cs + len(target_chunk.text)
    return ExpandedContext(
        merged_text=target_chunk.text,
        chunk_indices=[fallback_index],
        char_range=(cs, ce),
        original_target_index=fallback_index,
    )


def _target_as_meta(target_chunk: "ScoredChunk", target_index: int) -> ChunkMeta | None:
    meta = target_chunk.metadata or {}
    cs = _as_int(meta.get("char_start"))
    ce = _as_int(meta.get("char_end"))
    if cs is None or ce is None:
        return None
    return ChunkMeta(
        chunk_id=getattr(target_chunk, "chunk_id", ""),
        chunk_index=target_index,
        doc_id=getattr(target_chunk, "doc_id", "") or meta.get("doc_id", ""),
        char_start=cs,
        char_end=ce,
        text=target_chunk.text,
    )


def _as_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# ── Token budget (the "lost in the middle" guard) ─────────────────────────────

_TRUNC_MARK = " … (truncated) … "


def apply_token_budget(expansions: list[str], max_tokens: int) -> list[str]:
    """Cap total context tokens, preserving the highest-ranked expansion.

    `expansions` is ordered by rank (index 0 = top-ranked). If the estimated
    total exceeds max_tokens, trim from the MIDDLE of the lowest-ranked
    expansions first, walking upward, until it fits. Index 0 is never trimmed.
    Middle-truncation keeps each passage's head and tail (where a boundary answer
    tends to live) and drops the less-load-bearing center.
    """
    if max_tokens <= 0 or not expansions:
        return list(expansions)

    result = list(expansions)

    def total() -> int:
        return sum(estimate_tokens(t) for t in result)

    # Trim lowest-ranked first (skip index 0 — its full expansion is preserved).
    for i in range(len(result) - 1, 0, -1):
        if total() <= max_tokens:
            break
        overshoot_tokens = total() - max_tokens
        result[i] = _truncate_middle(result[i], remove_chars=overshoot_tokens * 4)

    return result


def _truncate_middle(text: str, remove_chars: int) -> str:
    """Remove ~remove_chars from the middle, keeping head + tail. Never collapses
    below a small floor so the passage stays recognizable."""
    if remove_chars <= 0:
        return text
    floor = 80  # keep at least this many chars of real content
    keep = max(floor, len(text) - remove_chars)
    if keep >= len(text):
        return text
    head = keep // 2
    tail = keep - head
    return text[:head] + _TRUNC_MARK + text[len(text) - tail:]


# ── Chroma fetch (the one I/O touch point) ────────────────────────────────────

def fetch_doc_chunks(chroma_client, user_id: str, doc_id: str) -> list[ChunkMeta]:
    """Fetch ALL chunks of one document from Chroma, ascending by chunk_index.

    Reads the per-user collection with a where={"doc_id": ...} filter. Chunks
    missing char offsets (pre-Story-5 ingest) are still returned but will make
    expand_context degrade to the target text. Any Chroma error yields [] — the
    caller then falls back to un-expanded chunk text.
    """
    collection_name = f"user_{user_id}"
    try:
        coll = chroma_client.get_collection(collection_name)
        fetched = coll.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
    except Exception as exc:
        logger.warning("fetch_doc_chunks failed for doc_id=%s: %s", doc_id, exc)
        return []

    out: list[ChunkMeta] = []
    ids = fetched.get("ids") or []
    docs = fetched.get("documents") or []
    metas = fetched.get("metadatas") or []
    for cid, text, meta in zip(ids, docs, metas):
        meta = meta or {}
        ci = _as_int(meta.get("chunk_index"))
        cs = _as_int(meta.get("char_start"))
        ce = _as_int(meta.get("char_end"))
        if ci is None or cs is None or ce is None:
            continue  # can't place or merge this chunk
        out.append(ChunkMeta(
            chunk_id=cid,
            chunk_index=ci,
            doc_id=meta.get("doc_id", doc_id),
            char_start=cs,
            char_end=ce,
            text=text or "",
        ))
    out.sort(key=lambda c: c.chunk_index)
    return out
