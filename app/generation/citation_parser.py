"""
Story 2 — parsing the model's citation markers back out of the answer.

build_grounded_prompt() asks the model to cite sources with [n] markers. This
module reads them back so the endpoint can report which sources the model
ACTUALLY used (cited_sources) vs everything it was handed (all_sources) — the
core Story 2 upgrade. Story 3 will validate that each marker maps to a real
claim; here we only extract them.

Handled marker shapes (all 1-based, matching the prompt's SOURCES numbering):
  [1]        single
  [1][2]     adjacent
  [1] [2]    spaced
  [1, 2]     comma-separated inside one bracket
  [1-3]      range → expands to 1, 2, 3
Duplicates are deduplicated; markers are preserved in the returned text (for
display); out-of-range indices are dropped when the chunk count is known.
"""
from __future__ import annotations

import re

# Matches a bracketed group of index tokens: digits, commas, hyphen ranges, spaces.
# e.g. "[1]", "[1,2]", "[1-3]", "[1, 2 , 4-6]". Bare "[]" or "[foo]" won't match.
_MARKER_RE = re.compile(r"\[\s*(\d+(?:\s*[-,]\s*\d+)*)\s*\]")

# Abstention key phrases — fuzzy signals that the model declined to answer from
# the sources. Matched case-insensitively against the apostrophe-normalized answer.
_ABSTENTION_SIGNALS = (
    "don't have enough information",
    "do not have enough information",
    "not contained in the provided",
    "not contained in the sources",
)


def _expand(token_group: str) -> list[int]:
    """Expand one bracket's contents ("1,3-5") into a flat list of ints."""
    out: list[int] = []
    for token in token_group.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            lo, hi = int(lo_s.strip()), int(hi_s.strip())
            if lo <= hi:
                out.extend(range(lo, hi + 1))
            else:  # tolerate reversed ranges like [3-1]
                out.extend(range(hi, lo + 1))
        else:
            out.append(int(token))
    return out


def parse_citations(
    answer_text: str, num_chunks: int | None = None
) -> tuple[str, list[int]]:
    """
    Extract [n] citation markers from an answer.

    Args:
        answer_text: the model's answer, with markers left in place.
        num_chunks:  number of sources fed to the model. When given, indices
                     outside 1..num_chunks are treated as invalid and dropped
                     (flagged, never raised) — a model that writes [99] against a
                     3-source set doesn't crash the endpoint. When None, every
                     extracted index is returned (no range filtering).

    Returns:
        (answer_text unchanged, sorted deduplicated list of valid 1-based indices).
    """
    if not answer_text:
        return answer_text, []

    seen: dict[int, None] = {}  # dict preserves first-seen order; we sort at the end
    for group in _MARKER_RE.findall(answer_text):
        for idx in _expand(group):
            if idx < 1:
                continue  # 0 / negatives are never valid citations
            if num_chunks is not None and idx > num_chunks:
                continue  # out of range → flagged invalid, dropped
            seen.setdefault(idx, None)

    return answer_text, sorted(seen)


def is_abstention(answer_text: str) -> bool:
    """
    Fuzzy-detect that the model abstained (answer not in the provided documents).

    Not an exact-string match: models rephrase the instructed abstention sentence,
    so we look for key phrases ("don't have enough information", "not contained in
    the provided") rather than requiring ABSTENTION_PHRASE verbatim.
    """
    if not answer_text:
        return False
    normalized = answer_text.lower().replace("’", "'")  # curly → straight apostrophe
    return any(signal in normalized for signal in _ABSTENTION_SIGNALS)
