"""
Story 2 — The Grounded Prompt.

Single responsibility: turn (query, retrieved chunks) into a prompt that makes the
model produce CITABLE, ABSTAINABLE answers.

The old prompt (retrieval.py, now deleted) said "cite the source" against an
unlabeled "---"-joined blob — the model had nothing to reference, no format to
follow, and no escape hatch, so it invented citation styles and confabulated when
the context was empty. This builder fixes all three:

  * every source is LABELED  [1], [2], [3] with its real filename + page, so the
    model has a concrete token to cite;
  * the citation FORMAT is specified ([n] markers), so the output is parseable
    (see citation_parser.parse_citations);
  * an ABSTENTION clause gives the model an exact phrase to emit when the answer
    is not in the sources, instead of making something up
    (see citation_parser.is_abstention / ABSTENTION_PHRASE).

Story 2 PRODUCES the [n] markers. Story 3 VALIDATES them. This module does not
filter chunks — it labels all of them and lets the model choose.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

from app.config import get_settings
from app.generation.context_expander import (
    ChunkMeta,
    apply_token_budget,
    estimate_tokens,
    expand_context,
)
from app.generation.query_classifier import (
    AnswerTask,
    QueryType,
    hint_for,
    task_instructions_for,
)
from app.observability import set_prompt_version

logger = logging.getLogger(__name__)

# ── Prompt-template versions (Phase 9B) ───────────────────────────────────────
# Git remains the SINGLE SOURCE OF TRUTH for the template bodies below; prompts are
# NOT fetched from Langfuse at request time. These identifiers are recorded on the
# generate span so any answer can be tied back to the exact template that produced
# it. Bump the matching version when you edit that template's body.
GROUNDED_SYSTEM_VERSION = "grounded-system/v2"      # _SYSTEM_INSTRUCTIONS
THIN_CONTEXT_NOTE_VERSION = "thin-context/v1"       # _THIN_CONTEXT_NOTE
# The composite identifier for the assembled grounded-QA prompt.
GROUNDED_PROMPT_VERSION = "grounded-qa/v3"

if TYPE_CHECKING:  # avoid a runtime import cycle with retrieval.py
    from app.pipeline.retrieval import ScoredChunk

# The exact sentence the model is instructed to emit when the sources don't answer
# the question. citation_parser.is_abstention() fuzzy-matches against it (models
# rephrase), but this is the canonical string the prompt asks for.
ABSTENTION_PHRASE = (
    "I don't have enough information in the provided documents to answer this."
)

# Generation-quality rewrite (Part 2).
#
# KEY DIFFERENCES from the OLD instructions (which produced verbatim parroting):
#   * "EXPLAIN, don't copy" (Rule 1) replaces the old "Be concise. Every sentence
#     must have at least one [n] citation." — that terse "transcribe-and-cite"
#     framing was what pushed the model to copy the shortest citable source
#     sentence. It is GONE (test_old_concise_phrasing_removed guards it stays gone).
#   * "Synthesize across MULTIPLE sources" (Rule 3) — the old prompt said nothing
#     about combining chunks, so the model answered from a single chunk. Now it is
#     told (and shown) how to weave several sources together.
#   * "Provide a DEFINITION even if not stated" (Rule 5) — fixes the "what is X"
#     failure where no chunk holds a single definitional sentence, so the model
#     used to parrot the nearest fragment or abstain instead of synthesizing one.
#   * "Your knowledge helps you EXPLAIN the source material, not supplement it"
#     (Rule 7) — the CRITICAL distinction. The old "Do NOT use outside information"
#     read as "don't think, just quote"; this explicitly PERMITS reasoning about
#     the sources while still forbidding new external facts. This is what turns a
#     parrot into an expert who uses evidence.
#   * "Structure longer answers" (Rule 8) — grants permission to write more than
#     one terse sentence, which the old "Be concise" was actively suppressing.
# The Story 2 citation contract is PRESERVED: labeled [n] sources, inline [n]
# markers on every claim (Rule 2), and the exact abstention clause (Rule 6).
_SYSTEM_INSTRUCTIONS = (
    "SYSTEM INSTRUCTIONS:\n"
    "You are an expert document analyst. Your job is to read the provided sources "
    "carefully and give CLEAR, HELPFUL answers in your own words.\n\n"
    "RULES:\n"
    "1. EXPLAIN, don't copy. Rephrase information from the sources into a clear, "
    "natural explanation. Do NOT copy sentences verbatim from the sources — "
    "synthesize and explain as an expert would to a colleague.\n"
    "2. ALWAYS cite your sources using [n] markers after each claim (e.g. [1], "
    "[2]). Every factual statement must be traceable to a specific source.\n"
    "3. When information comes from MULTIPLE sources, synthesize them into a "
    'coherent answer. Say things like "According to the manual [1], X works by '
    'doing Y, and the specification [2] adds that Z is also relevant."\n'
    "4. If the sources contain the information but it's technical or dense, EXPLAIN "
    "it in accessible language while staying faithful to the facts.\n"
    '5. If the question asks "what is" something, provide a clear DEFINITION or '
    "EXPLANATION, even if the sources don't contain a single definitional "
    "sentence — synthesize the definition from what the sources describe.\n"
    "6. If the sources genuinely do not contain enough information to answer, "
    f'respond EXACTLY with: "{ABSTENTION_PHRASE}"\n'
    "7. Do NOT use information from outside the provided sources. Your knowledge "
    "helps you EXPLAIN the source material, not supplement it with new facts.\n"
    "8. Structure longer answers with brief context first, then the detailed "
    "explanation, then any caveats or limitations mentioned in the sources."
)

# Part 4b — appended when retrieval found very little (still < 2 chunks after the
# thin-result retry). Without it the model blindly abstains on thin-but-present
# evidence; with it, it answers as far as the sparse sources allow and says so.
_THIN_CONTEXT_NOTE = (
    "Note: limited source material was found for this question. Answer as "
    "completely as you can from what is available, and clearly state if the "
    "sources don't fully cover the topic."
)


def _label(chunk: "ScoredChunk", n: int) -> str:
    """Render the one-line source header: ``[n] (source_filename, p.page_number)``.

    Prefers the metadata dict (the real original filename tracked from parser.py),
    falling back to the ScoredChunk fields hybrid_search populates from that same
    metadata. n is 1-based (user-facing).
    """
    source = chunk.metadata.get("source") or getattr(chunk, "source", "") or "unknown"
    page = chunk.metadata.get("page_number")
    if page is None:
        page = getattr(chunk, "page_number", 0)
    return f"[{n}] ({source}, p.{page})"


def _display_texts(
    chunks: "list[ScoredChunk]",
    doc_chunks_fetcher: "Callable[[str], list[ChunkMeta]] | None",
    settings,
) -> list[str]:
    """The text block shown per chunk. With expansion off (or no fetcher) this is
    the original fragment; with expansion on, each chunk is widened to its
    neighbors (Story 5) and the total is capped by the token budget.

    Doc-chunk fetches are cached per doc_id within this call, so top-k chunks from
    the same document trigger exactly one Chroma query for that document.
    """
    # No fetcher → expansion is impossible; don't even resolve settings (keeps the
    # pure build_grounded_prompt(query, chunks) call path free of config access).
    if not chunks or doc_chunks_fetcher is None:
        return [c.text for c in chunks]

    settings = settings or get_settings()
    if not settings.context_expansion_enabled:
        return [c.text for c in chunks]

    window = settings.context_expansion_window
    cache: dict[str, list[ChunkMeta]] = {}
    texts: list[str] = []
    for c in chunks:
        doc_id = getattr(c, "doc_id", "") or (c.metadata or {}).get("doc_id", "")
        if doc_id not in cache:
            cache[doc_id] = doc_chunks_fetcher(doc_id) if doc_id else []
        texts.append(expand_context(c, cache[doc_id], window=window).merged_text)

    # Highest-ranked chunk (index 0) is preserved fully; lower ones trim first.
    return apply_token_budget(texts, settings.context_max_tokens)


def _assemble(
    query: str,
    chunks: "list[ScoredChunk]",
    display_texts: list[str],
    *,
    query_hint: str = "",
    thin_note: str = "",
    task_block: str = "",
) -> str:
    """Render the final prompt from the (possibly budgeted) source texts.

    query_hint (Part 3), thin_note (Part 4b) and task_block (Scope-Aware Routing)
    are appended AFTER the question and BEFORE the ANSWER: cue so they steer the
    response without disturbing the labeled-sources / QUESTION layout the citation
    contract relies on.
    """
    if chunks:
        blocks = "\n\n".join(
            f"{_label(c, n)}\n{text}"
            for n, (c, text) in enumerate(zip(chunks, display_texts), 1)
        )
    else:
        blocks = "(no sources retrieved)"

    parts = [
        _SYSTEM_INSTRUCTIONS,
        "",
        "SOURCES:",
        blocks,
        "",
        f"QUESTION: {query}",
    ]
    if query_hint:
        parts += ["", f"QUERY-TYPE HINT: {query_hint}"]
    if task_block:
        parts += ["", f"TASK INSTRUCTIONS: {task_block}"]
    if thin_note:
        parts += ["", thin_note]
    parts += ["", "ANSWER:"]
    return "\n".join(parts)


def _enforce_input_budget(
    query: str,
    chunks: "list[ScoredChunk]",
    display_texts: list[str],
    max_input_tokens: int,
    *,
    query_hint: str = "",
    thin_note: str = "",
    task_block: str = "",
) -> list[str]:
    """Story 7 — cap the WHOLE prompt at max_input_tokens.

    The overhead (system instructions + [n] labels + question + any hint/note,
    everything but the source texts) is fixed; whatever budget remains goes to
    the source texts, trimmed LOWEST-ranked first via Story 5's middle-truncation
    (index 0 — the top-ranked chunk — is preserved). A no-op when under budget.
    """
    if max_input_tokens <= 0 or not chunks:
        return display_texts

    assembled = _assemble(
        query, chunks, display_texts,
        query_hint=query_hint, thin_note=thin_note, task_block=task_block,
    )
    if estimate_tokens(assembled) <= max_input_tokens:
        return display_texts

    # Overhead = the prompt with all source texts blanked out.
    overhead = estimate_tokens(
        _assemble(
            query, chunks, ["" for _ in display_texts],
            query_hint=query_hint, thin_note=thin_note, task_block=task_block,
        )
    )
    source_budget = max(max_input_tokens - overhead, 0)
    budgeted = apply_token_budget(display_texts, source_budget)
    logger.warning(
        "Prompt over GEN_MAX_INPUT_TOKENS (%d): truncated sources from ≈%d to ≈%d tokens",
        max_input_tokens,
        sum(estimate_tokens(t) for t in display_texts),
        sum(estimate_tokens(t) for t in budgeted),
    )
    return budgeted


def build_grounded_prompt(
    query: str,
    chunks: "list[ScoredChunk]",
    *,
    doc_chunks_fetcher: "Callable[[str], list[ChunkMeta]] | None" = None,
    settings=None,
    query_type: "QueryType | None" = None,
    thin_context: bool = False,
    answer_task: "AnswerTask | None" = None,
) -> str:
    """
    Build the grounded QA prompt: labeled, citable sources + abstention contract.

    Sources are numbered 1-based to match the [n] citation markers the model is
    asked to emit; citation_parser.parse_citations() reads those same 1-based
    indices back out of the answer.

    Story 5: when a doc_chunks_fetcher is supplied and CONTEXT_EXPANSION_ENABLED,
    the TEXT block under each [n] is the chunk merged with its neighbors, not the
    raw fragment. The [n] LABEL still names the ORIGINAL chunk's source/page —
    that's what the citation points at — only the readable text is widened.

    Story 7: the assembled prompt is capped at GEN_MAX_INPUT_TOKENS, trimming the
    lowest-ranked sources first. A well-sized prompt is unaffected (no-op).

    Generation-quality fix:
      * query_type (Part 3) selects a QUERY-TYPE HINT appended after the question
        so the model adjusts its response STYLE (summarize / define / compare).
        None or INFORMATIONAL adds no hint.
      * thin_context (Part 4b) appends the limited-material note so the model
        answers from sparse-but-present evidence instead of blindly abstaining.
    """
    query_hint = hint_for(query_type) if query_type is not None else ""
    thin_note = _THIN_CONTEXT_NOTE if thin_context else ""
    task_block = task_instructions_for(answer_task)

    # Record WHICH template version served this answer (read by the generate span via
    # the generation seam). The composite reflects the variants that actually shaped
    # the prompt — the answer-task block and the thin-context note — so the recorded
    # id maps 1:1 to the emitted template. Observability only; behaviour unchanged.
    task_tag = getattr(answer_task, "value", None) or "answer"
    version = f"{GROUNDED_PROMPT_VERSION}+task:{task_tag}"
    if thin_context:
        version += "+thin"
    set_prompt_version(version)

    display_texts = _display_texts(chunks, doc_chunks_fetcher, settings)

    # Story 7 input budget. Only touch config when there are chunks to trim; a
    # small prompt never exceeds the cap, so the pure build_grounded_prompt(query,
    # chunks) path stays behavior-identical (just an estimate + comparison).
    if chunks:
        resolved = settings or get_settings()
        display_texts = _enforce_input_budget(
            query, chunks, display_texts, getattr(resolved, "gen_max_input_tokens", 0),
            query_hint=query_hint, thin_note=thin_note, task_block=task_block,
        )

    return _assemble(
        query, chunks, display_texts,
        query_hint=query_hint, thin_note=thin_note, task_block=task_block,
    )
