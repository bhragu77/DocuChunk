"""
Generation-quality fix (Part 5) — the anti-verbatim guard.

The prompt rewrite (Part 2) is the real fix for parroting. This is the SAFETY
NET: a cheap, post-generation string check that catches the failure even when a
weaker model still copies. It is MONITORING ONLY — it never blocks or rewrites
the answer; it just measures how much of the answer is lifted verbatim from the
sources and surfaces that as verbatim_ratio (+ a "high_verbatim" warning) so the
endpoint and frontend can flag a suspiciously source-hugging answer.

No LLM call. For each answer SENTENCE we find the longest CONTIGUOUS run of
tokens it shares with any source chunk (difflib's longest-match, i.e. a
token-level longest-common-substring). If that run covers more than
VERBATIM_THRESHOLD of the sentence, the sentence is a near-verbatim copy.
verbatim_ratio = flagged_sentences / total_sentences.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from app.config import get_settings

if TYPE_CHECKING:  # avoid a runtime import cycle with retrieval.py
    from app.pipeline.retrieval import ScoredChunk

# Citation markers are stripped before comparison so "[1]" never counts as a
# copied (or non-copied) token.
_MARKER_RE = re.compile(r"\[[^\]]*\]")
# Same simple sentence splitter the citation validator uses.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"\w+")

# Sentences shorter than this (in tokens) are skipped — too short for a
# contiguous-run ratio to be meaningful, and they'd falsely inflate the ratio.
# Matches the spec's n=6 n-gram notion.
_MIN_SENTENCE_TOKENS = 6


@dataclass
class VerbatimReport:
    """How much of an answer is lifted verbatim from its sources (Part 5)."""
    verbatim_ratio: float                     # flagged / total sentences (0-1)
    verbatim_sentence_count: int
    total_sentences: int
    flagged_sentences: list[dict] = field(default_factory=list)  # debugging detail
    quality_warning: str | None = None        # "high_verbatim" or None


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _longest_run_ratio(sent_tokens: list[str], chunk_tokens: list[str]) -> float:
    """Longest contiguous shared token run between the two, ÷ sentence length."""
    if not sent_tokens or not chunk_tokens:
        return 0.0
    match = SequenceMatcher(
        None, sent_tokens, chunk_tokens, autojunk=False
    ).find_longest_match(0, len(sent_tokens), 0, len(chunk_tokens))
    return match.size / len(sent_tokens)


def check_verbatim(
    answer: str,
    source_chunks: "list[ScoredChunk]",
    *,
    threshold: float | None = None,
    warning_ratio: float | None = None,
) -> VerbatimReport:
    """Measure verbatim copying of `answer` against `source_chunks`.

    Args:
        answer:        the model's answer (citation markers are stripped here).
        source_chunks: the chunks fed to the model.
        threshold:     per-sentence copy fraction to flag as VERBATIM
                       (default VERBATIM_THRESHOLD).
        warning_ratio: verbatim_ratio above which quality_warning is set
                       (default VERBATIM_WARNING_RATIO).

    Never mutates the answer — pure measurement.
    """
    settings = get_settings()
    threshold = settings.verbatim_threshold if threshold is None else threshold
    warning_ratio = (
        settings.verbatim_warning_ratio if warning_ratio is None else warning_ratio
    )

    clean = _MARKER_RE.sub(" ", answer or "").strip()
    sentences = [s.strip() for s in _SENT_SPLIT.split(clean) if s.strip()]
    chunk_token_lists = [_tokens(c.text) for c in source_chunks]

    total = 0
    flagged: list[dict] = []
    for sent in sentences:
        sent_tokens = _tokens(sent)
        if len(sent_tokens) < _MIN_SENTENCE_TOKENS:
            continue  # too short to judge — skip
        total += 1

        best_ratio = 0.0
        best_source = ""
        for chunk, chunk_tokens in zip(source_chunks, chunk_token_lists):
            ratio = _longest_run_ratio(sent_tokens, chunk_tokens)
            if ratio > best_ratio:
                best_ratio = ratio
                best_source = chunk.text
        if best_ratio > threshold:
            flagged.append({
                "sentence": sent,
                "overlap": round(best_ratio, 3),
                "source_text": best_source[:200],
            })

    verbatim_ratio = round(len(flagged) / total, 3) if total else 0.0
    quality_warning = "high_verbatim" if verbatim_ratio > warning_ratio else None

    return VerbatimReport(
        verbatim_ratio=verbatim_ratio,
        verbatim_sentence_count=len(flagged),
        total_sentences=total,
        flagged_sentences=flagged,
        quality_warning=quality_warning,
    )
