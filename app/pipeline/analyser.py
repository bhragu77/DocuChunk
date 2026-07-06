"""
Pipeline Stage 4 — Analysis

Computes quality statistics for a list of Chunk objects, using the exact
char_start/char_end offsets established in Phase 4 — no text re-scanning.

Two public functions:
  analyse_chunks   — stats + anomaly warnings for one strategy's output
  compare_strategies — runs analyse_chunks for each strategy side by side
"""

from __future__ import annotations

import logging
from typing import Any

from app.pipeline.types import Chunk, CleanedPage

logger = logging.getLogger(__name__)

MIN_HEALTHY_CHARS: int = 50
DEFAULT_BUCKET_BOUNDARIES: list[int] = [0, 200, 400, 600, 800]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_distribution(
    char_sizes: list[int],
    boundaries: list[int],
) -> dict[str, int]:
    """
    Bucket char_sizes into a histogram.
    Buckets are half-open intervals [boundaries[i], boundaries[i+1]),
    with an open-ended final bucket 'boundaries[-1]+'.
    """
    labels: list[str] = [
        f"{boundaries[i]}-{boundaries[i + 1]}"
        for i in range(len(boundaries) - 1)
    ]
    labels.append(f"{boundaries[-1]}+")

    dist: dict[str, int] = {label: 0 for label in labels}
    for size in char_sizes:
        placed = False
        for i in range(len(boundaries) - 1):
            if boundaries[i] <= size < boundaries[i + 1]:
                dist[labels[i]] += 1
                placed = True
                break
        if not placed:
            dist[labels[-1]] += 1
    return dist


# ── Public API ────────────────────────────────────────────────────────────────

def analyse_chunks(
    chunks: list[Chunk],
    pages: list[CleanedPage],
    *,
    min_healthy_chars: int = MIN_HEALTHY_CHARS,
    bucket_boundaries: list[int] | None = None,
) -> dict[str, Any]:
    """
    Compute quality statistics and anomaly warnings for a set of chunks.

    overlap_coverage_pct formula
    ----------------------------
    For each consecutive pair of chunks (sorted by chunk_index), the
    overlapping character range is:
        max(0, min(end_i, end_{i+1}) - max(start_i, start_{i+1}))
    where start/end are the chunks' char_start/char_end offsets into the
    joined document string from Phase 4 — NOT re-derived from text content.

    total_doc_chars = max(char_end) across all chunks, i.e. the length of
    the joined document string.  This is the denominator that makes the
    percentage meaningful: "what fraction of the document's character span
    exists in more than one chunk."
    """
    if not chunks:
        return {
            "doc_id": None,
            "strategy": None,
            "total_chunks": 0,
            "avg_chars": 0.0,
            "min_chars": 0,
            "max_chars": 0,
            "avg_tokens": 0.0,
            "min_tokens": 0,
            "max_tokens": 0,
            "total_tokens": 0,
            "size_distribution": {},
            "overlap_coverage_pct": 0.0,
            "chunks_per_page": {},
            "warnings": [],
        }

    if bucket_boundaries is None:
        bucket_boundaries = DEFAULT_BUCKET_BOUNDARIES

    sorted_chunks = sorted(chunks, key=lambda c: c.chunk_index)
    first = sorted_chunks[0]

    doc_id: str = first.doc_id
    strategy: str = first.metadata.get("strategy", "unknown")

    # ── Character stats ───────────────────────────────────────────────────────
    char_sizes = [len(c.text) for c in sorted_chunks]
    avg_chars = sum(char_sizes) / len(char_sizes)
    min_chars = min(char_sizes)
    max_chars = max(char_sizes)

    # ── Token stats ───────────────────────────────────────────────────────────
    token_counts = [c.token_count for c in sorted_chunks]
    total_tokens = sum(token_counts)
    avg_tokens = total_tokens / len(token_counts)
    min_tokens = min(token_counts)
    max_tokens = max(token_counts)

    # ── Size distribution ─────────────────────────────────────────────────────
    size_distribution = _build_distribution(char_sizes, bucket_boundaries)

    # ── Overlap coverage ──────────────────────────────────────────────────────
    total_overlap_chars = 0
    for a, b in zip(sorted_chunks, sorted_chunks[1:]):
        overlap = max(0, min(a.char_end, b.char_end) - max(a.char_start, b.char_start))
        total_overlap_chars += overlap

    total_doc_chars = max(c.char_end for c in sorted_chunks)
    overlap_coverage_pct = (
        total_overlap_chars / total_doc_chars * 100
        if total_doc_chars > 0 else 0.0
    )

    # ── Chunks per page ───────────────────────────────────────────────────────
    chunks_per_page: dict[int, int] = {}
    for c in sorted_chunks:
        chunks_per_page[c.page_number] = chunks_per_page.get(c.page_number, 0) + 1

    # ── Anomaly warnings ──────────────────────────────────────────────────────
    warnings: list[str] = []

    tiny = [c for c in sorted_chunks if len(c.text) < min_healthy_chars]
    if tiny:
        indices = [c.chunk_index for c in tiny]
        warnings.append(
            f"{len(tiny)} chunk(s) under {min_healthy_chars} chars "
            f"(indices: {indices}) — likely a parsing artifact or nearly-empty chunk"
        )

    params = first.metadata.get("params", {})
    strategy_max = params.get("max_chars") or params.get("chunk_size")
    if strategy_max:
        oversized = [c for c in sorted_chunks if len(c.text) > 2 * strategy_max]
        if oversized:
            indices = [c.chunk_index for c in oversized]
            warnings.append(
                f"{len(oversized)} chunk(s) exceed 2× strategy max of {strategy_max} chars "
                f"(indices: {indices}) — paragraph fallback to sentence may not be working"
            )

    if pages:
        chunked_page_nums = set(chunks_per_page.keys())
        for page in pages:
            if page.page_number not in chunked_page_nums and page.cleaned_char_count > 0:
                warnings.append(
                    f"Page {page.page_number} has no chunks but "
                    f"{page.cleaned_char_count} source chars — "
                    "content may have been dropped in Phase 4"
                )

    logger.debug(
        "analyse_chunks: doc=%s strategy=%s chunks=%d overlap_pct=%.2f%%",
        doc_id, strategy, len(sorted_chunks), overlap_coverage_pct,
    )

    return {
        "doc_id": doc_id,
        "strategy": strategy,
        "total_chunks": len(sorted_chunks),
        "avg_chars": round(avg_chars, 2),
        "min_chars": min_chars,
        "max_chars": max_chars,
        "avg_tokens": round(avg_tokens, 2),
        "min_tokens": min_tokens,
        "max_tokens": max_tokens,
        "total_tokens": total_tokens,
        "size_distribution": size_distribution,
        "overlap_coverage_pct": round(overlap_coverage_pct, 6),
        "chunks_per_page": dict(sorted(chunks_per_page.items())),
        "warnings": warnings,
    }


def compare_strategies(
    doc_chunks_by_strategy: dict[str, list[Chunk]],
    pages: list[CleanedPage],
) -> dict[str, dict]:
    """
    Run analyse_chunks for each strategy on the same document's chunks.
    Keys in the result mirror the keys in doc_chunks_by_strategy.
    """
    return {
        strategy: analyse_chunks(chunks, pages)
        for strategy, chunks in doc_chunks_by_strategy.items()
    }
