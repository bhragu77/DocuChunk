"""
Story 3 — Citation Validation.

Story 2 read the model's [n] markers back out of the answer (citation_parser).
That told us which chunks the model *referenced* — but "referenced" is not
"supported". A model can write:

    "The GX-4200 supports temperatures up to 85°C [2]."

while chunk [2] actually says:

    "The GX-4200 is rated for indoor use between 0°C and 45°C."

The marker looks authoritative; the claim contradicts the source. This module
closes that gap: for each cited [n], it checks whether the cited chunk's text
actually SUPPORTS the sentence(s) carrying the marker, and drops the ones that
don't.

Two tiers, cheap first:

  Tier 1 — lexical, NO LLM. Pull the "specific facts" out of the claim (numbers,
  codes like GX-4200). If a fact the claim asserts does not appear anywhere in
  the cited chunk, the citation is SUSPECT (the 85°C-vs-45°C case: 85 simply
  isn't in the source). If every specific fact is present — or, absent any
  specific facts, the claim's content words overlap the chunk strongly — the
  citation is CLEARLY supported and we accept it WITHOUT spending an LLM call.
  Most valid citations settle here.

  Tier 2 — LLM judge, ONLY for the SUSPECT cases Tier 1 flagged. It adjudicates
  supported / contradicted / not_mentioned — the distinction Tier 1 can't make
  cheaply (a missing "2.3kg" could be a contradiction or an off-topic chunk).

Failure policy is per-citation FAIL-OPEN: if the judge is disabled
(VALIDATE_USE_LLM=false) or the judge call raises, that ONE citation becomes
"unverified" — NOT dropped — and is flagged so Story 4's confidence formula can
penalize the uncertainty. One flaky judge call must not silently delete a
citation the user might have relied on.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from app.generation.citation_parser import is_abstention, parse_citations

if TYPE_CHECKING:  # avoid a runtime import cycle with retrieval.py
    from app.pipeline.retrieval import ScoredChunk

logger = logging.getLogger(__name__)

# Fraction of a claim's content words that must appear in the chunk for a
# fact-free claim to count as clearly supported (skip the LLM). Below this the
# claim is SUSPECT and goes to the Tier 2 judge.
OVERLAP_THRESHOLD = 0.6

# Any bracketed marker — stripped from a claim before fact extraction so the
# citation number itself ("[1]") is never mistaken for an asserted fact.
_MARKER_RE = re.compile(r"\[[^\]]*\]")

# A "specific fact": a code (a token carrying BOTH a letter and a digit, e.g.
# GX-4200, 4000mAh) or a bare number (integer or decimal, e.g. 85, 2.3).
_CODE_RE = re.compile(r"\b(?=[A-Za-z0-9-]*[A-Za-z])(?=[A-Za-z0-9-]*\d)[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\b")
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

# Sentence splitter — split AFTER ., !, ? followed by whitespace. Deliberately
# simple (per the story's "simple heuristic"): good enough to attribute a [n]
# marker to the sentence(s) that carry it.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Word tokenizer for overlap scoring.
_WORD_RE = re.compile(r"[a-z0-9]+")

# Function words that carry no support signal — excluded from overlap scoring so
# "The X is Y" doesn't score as supported purely on "the/is".
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "and", "or", "in", "on", "for", "up", "at", "by", "with",
    "can", "could", "will", "would", "this", "that", "it", "its", "has",
    "have", "had", "between", "from", "as", "than", "then", "into", "about",
})

# Verdict → confidence-in-the-verdict. Consumed by Story 4; kept coarse on
# purpose (these are heuristics, not calibrated probabilities).
_CONFIDENCE = {
    "supported": 0.9,       # Tier 1 clear support
    "supported_llm": 0.75,  # Tier 2 judge said supported
    "contradicted": 0.9,
    "not_mentioned": 0.75,
    "unverified": 0.0,
    "invalid": 0.0,
}


@dataclass
class ValidatedCitation:
    """The verdict for a single cited [n] marker."""
    index: int              # 1-based marker index as written by the model
    claim_text: str         # the sentence(s) carrying this marker
    chunk_id: str | None    # cited chunk's id, or None when the index is out of range
    verdict: str            # supported | contradicted | not_mentioned | unverified | invalid
    confidence: float


@dataclass
class ValidationResult:
    validated_citations: list[ValidatedCitation] = field(default_factory=list)
    valid_indices: list[int] = field(default_factory=list)       # verdict == supported
    dropped_indices: list[int] = field(default_factory=list)     # contradicted / not_mentioned / invalid
    unverified_indices: list[int] = field(default_factory=list)  # judge skipped or failed (fail-open)


# ── Claim extraction ──────────────────────────────────────────────────────────

def _claim_for_index(answer_text: str, index: int) -> str:
    """Return the sentence(s) in the answer that carry the [index] marker.

    A claim is the sentence bearing the marker — not the whole answer. Sentence
    membership is decided by re-parsing each sentence's own markers (so [1,2],
    [1-3] and adjacent forms all attribute correctly). If nothing matches (an
    odd split), fall back to the whole answer so Tier 1 still has text to work on.
    """
    matched: list[str] = []
    for sentence in _SENT_SPLIT.split(answer_text):
        _, indices = parse_citations(sentence)
        if index in indices:
            matched.append(sentence.strip())
    return " ".join(matched) if matched else answer_text.strip()


# ── Tier 1: lexical ───────────────────────────────────────────────────────────

def _specific_facts(claim: str) -> set[str]:
    """Codes and numbers asserted by the claim, lowercased, markers stripped."""
    stripped = _MARKER_RE.sub(" ", claim)
    facts = {m.lower() for m in _CODE_RE.findall(stripped)}
    facts |= set(_NUM_RE.findall(stripped))
    return facts


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD_RE.findall(text.lower()) if t not in _STOPWORDS}


def _overlap_ratio(claim: str, chunk_text: str) -> float:
    claim_tokens = _tokens(_MARKER_RE.sub(" ", claim))
    if not claim_tokens:
        return 1.0  # nothing to disagree with
    return len(claim_tokens & _tokens(chunk_text)) / len(claim_tokens)


# ── Tier 2: LLM judge ─────────────────────────────────────────────────────────

def _judge_prompt(claim: str, chunk_text: str) -> str:
    return (
        f"Claim: {claim}\n"
        f"Source: {chunk_text}\n"
        "Does the source support this claim? Answer ONLY 'supported', "
        "'contradicted', or 'not_mentioned'."
    )


def _parse_judge(response: str) -> str:
    """Map a free-form judge reply onto one verdict. contradicted/not_mentioned
    are checked before 'support' so a "not supported" reply can't read as
    supported. An unrecognizable reply is treated as unverified (fail-open)."""
    t = response.strip().lower()
    if "contradict" in t:
        return "contradicted"
    if "not_mentioned" in t or "not mentioned" in t or "no mention" in t or "not support" in t:
        return "not_mentioned"
    if "support" in t:
        return "supported"
    return "unverified"


def _assess(
    claim: str,
    chunk_text: str,
    llm_fn: Callable[[str], str] | None,
    use_llm: bool,
) -> tuple[str, float]:
    """Assess one claim against one chunk. Returns (verdict, confidence)."""
    chunk_lower = chunk_text.lower()
    facts = _specific_facts(claim)
    missing = [f for f in facts if f not in chunk_lower]

    if not missing:
        # Tier 1 clears it: either every asserted fact is present, or (no facts)
        # the content words overlap strongly. No LLM spent.
        if facts:
            return "supported", _CONFIDENCE["supported"]
        if _overlap_ratio(claim, chunk_text) >= OVERLAP_THRESHOLD:
            return "supported", _CONFIDENCE["supported"]
    # else: a specific fact the claim asserts is absent from the chunk → SUSPECT.

    # SUSPECT → Tier 2. Judge disabled or unavailable → fail-open as unverified.
    if not use_llm or llm_fn is None:
        return "unverified", _CONFIDENCE["unverified"]

    try:
        raw = llm_fn(_judge_prompt(claim, chunk_text))
    except Exception as exc:  # a flaky judge must not delete the citation
        logger.warning("Citation judge call failed: %s", exc)
        return "unverified", _CONFIDENCE["unverified"]

    verdict = _parse_judge(raw)
    if verdict == "supported":
        return "supported", _CONFIDENCE["supported_llm"]
    return verdict, _CONFIDENCE.get(verdict, 0.0)


# ── Public entry point ────────────────────────────────────────────────────────

def validate_citations(
    answer_text: str,
    cited_indices: list[int],
    chunks: "list[ScoredChunk]",
    llm_fn: Callable[[str], str] | None = None,
    *,
    use_llm: bool = True,
) -> ValidationResult:
    """Validate every cited [n] against the text of the chunk it points to.

    Args:
        answer_text:   the model's answer, markers left in place.
        cited_indices: RAW 1-based indices parsed from the answer (may include
                       out-of-range markers — they are recorded as invalid here,
                       never crashed on).
        chunks:        the chunks fed to the model, in [n] order (chunk n == chunks[n-1]).
        llm_fn:        the Tier 2 judge callable. May be None (→ unverified).
        use_llm:       enable the Tier 2 judge (VALIDATE_USE_LLM). When False the
                       Tier 1 SUSPECT cases become unverified instead of judged.

    Returns a ValidationResult. When the answer is an abstention there is nothing
    to validate — an empty result is returned.
    """
    if not answer_text or is_abstention(answer_text):
        return ValidationResult()

    n_chunks = len(chunks)
    result = ValidationResult()

    for index in cited_indices:
        # Edge: the model cited a source that doesn't exist. Record it as invalid
        # and drop it — no chunk to check, no crash.
        if index < 1 or index > n_chunks:
            result.validated_citations.append(
                ValidatedCitation(index, "", None, "invalid", _CONFIDENCE["invalid"])
            )
            result.dropped_indices.append(index)
            continue

        chunk = chunks[index - 1]
        claim = _claim_for_index(answer_text, index)
        verdict, confidence = _assess(claim, chunk.text, llm_fn, use_llm)

        result.validated_citations.append(
            ValidatedCitation(index, claim, chunk.chunk_id, verdict, confidence)
        )
        if verdict == "supported":
            result.valid_indices.append(index)
        elif verdict in ("contradicted", "not_mentioned"):
            result.dropped_indices.append(index)
        else:  # unverified — fail-open, flagged for Story 4
            result.unverified_indices.append(index)

    return result
