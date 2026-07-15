"""
Generation-quality fix (Part 3) — lightweight query classification.

The prompt builder used ONE static instruction block for every query, so a bare
heading typed as a query ("Operating Temperature") got the same "be concise and
cite" treatment as a real question — and the model just parroted the matching
heading chunk. This module tags each query with a QueryType and supplies a
QUERY-TYPE HINT the prompt builder appends so the model adjusts its response
STYLE (summarize vs. define vs. compare).

Design constraints (from the spec):
  * PURE HEURISTIC — no LLM call, must run in well under 1ms. It only inspects
    the query string; it never touches retrieval (retrieval is unchanged).
  * The classification adjusts FRAMING only. INFORMATIONAL (the default) adds no
    hint at all — the base instructions already cover it.
"""
from __future__ import annotations

import re
from enum import Enum


class QueryType(str, Enum):
    """How the user's query reads — drives the prompt's response-style hint."""
    NAVIGATIONAL = "navigational"   # a heading/title — "show me section X"
    DEFINITIONAL = "definitional"   # "what is X" — wants a concept explained
    ANALYTICAL = "analytical"       # "compare X and Y" — wants analysis/tradeoffs
    INFORMATIONAL = "informational" # everything else (default; no extra hint)


# Question words that disqualify a short phrase from being a heading (NAVIGATIONAL).
_QUESTION_WORDS = frozenset({
    "what", "how", "why", "when", "where", "which", "is", "does", "can",
})

# DEFINITIONAL openers — the user is explicitly asking for a concept explained.
_DEFINITIONAL_PREFIXES = (
    "what is", "what are", "define", "explain", "describe",
)

# ANALYTICAL cues — comparison / evaluation intent. Matched as space-padded
# substrings so " vs " doesn't fire inside an unrelated word.
_ANALYTICAL_TERMS = (
    "compare", "comparison", "difference", "differences", "versus",
    " vs ", " vs. ", "pros and cons", "pros/cons", "better than", "which is better",
    "relate to", "related to", "tradeoff", "trade-off", "tradeoffs", "advantages",
)

# Common section-header shapes: "Chapter 3", "Section 2.1", "Appendix A",
# "2.3 Operating Limits". A heading, not a question.
_HEADER_RE = re.compile(
    r"^(chapter|section|part|appendix|figure|table|fig|no\.?)\s+[\divxlc]+",
    re.IGNORECASE,
)
_SECTION_NUM_RE = re.compile(r"^\d+(\.\d+)*[\s).:-]")


def _is_title_case(text: str) -> bool:
    """True when (nearly) every word starts uppercase — a heading/title look.

    One lowercase word is tolerated so titles with an article ("Care and Use")
    still count.
    """
    words = [w for w in text.split() if w[:1].isalpha()]
    if not words:
        return False
    capped = sum(1 for w in words if w[0].isupper())
    return capped >= max(1, len(words) - 1)


def classify_query(query: str) -> QueryType:
    """Classify `query` into one of four QueryTypes by pure heuristics.

    Precedence is intent-specificity first: an explicit comparison ("compare",
    "difference") or definition opener ("what is") outranks the heading shape,
    so "What is the difference between X and Y" reads as ANALYTICAL, not a
    heading. Anything that isn't a heading and carries no comparison/definition
    cue falls through to INFORMATIONAL.
    """
    q = (query or "").strip()
    if not q:
        return QueryType.INFORMATIONAL

    lower = q.lower()
    padded = f" {lower} "

    # ANALYTICAL — most specific intent, checked first.
    if any(term in padded for term in _ANALYTICAL_TERMS):
        return QueryType.ANALYTICAL

    # DEFINITIONAL — an explicit "explain this concept" opener.
    if lower.startswith(_DEFINITIONAL_PREFIXES):
        return QueryType.DEFINITIONAL

    # NAVIGATIONAL — short, no question words, and looks like a heading/title.
    words = q.split()
    has_question_word = any(
        w.strip("?.,:;").lower() in _QUESTION_WORDS for w in words
    )
    if len(words) <= 5 and not has_question_word and "?" not in q:
        if (
            q.isupper()
            or _HEADER_RE.match(q)
            or _SECTION_NUM_RE.match(q)
            or _is_title_case(q)
        ):
            return QueryType.NAVIGATIONAL

    # INFORMATIONAL — the default; the base prompt instructions already cover it.
    return QueryType.INFORMATIONAL


# QUERY-TYPE HINT text appended to the prompt (after QUESTION, before ANSWER).
# INFORMATIONAL intentionally has NO hint — the base instructions suffice.
_QUERY_TYPE_HINTS: dict[QueryType, str] = {
    QueryType.NAVIGATIONAL: (
        "The user is looking for a specific section. Summarize what that section "
        "covers and explain its key points — do not just repeat it."
    ),
    QueryType.DEFINITIONAL: (
        "The user wants a clear explanation of this concept. Define it, explain "
        "how it works, and note any important details from the sources."
    ),
    QueryType.ANALYTICAL: (
        "The user wants a comparison or analysis. Structure your answer to address "
        "each element and highlight the differences or tradeoffs the sources describe."
    ),
    QueryType.INFORMATIONAL: "",
}


def hint_for(query_type: QueryType) -> str:
    """The QUERY-TYPE HINT string for a type ("" for INFORMATIONAL / unknown)."""
    return _QUERY_TYPE_HINTS.get(query_type, "")


# ══════════════════════════════════════════════════════════════════════════════
# Scope-Aware Retrieval Routing (Keebler reasoning fix)
#
# A SECOND, ORTHOGONAL axis to QueryType. QueryType tweaks response STYLE; this
# decides the RETRIEVAL STRATEGY and the OUTPUT SHAPE for "global" questions that
# operate over the WHOLE document, not the top-k slice:
#
#   * classify_task  — what shape the answer must take (enumerate / table /
#                      classify / summarize / critique / infer / plain answer).
#   * classify_scope — LOCAL (top-k is correct) vs GLOBAL (needs the whole doc).
#
# Both are PURE HEURISTICS (no LLM, <1ms) in the spirit of classify_query. The
# design bias is DEFAULT-TO-LOCAL: a false GLOBAL is expensive (loads the whole
# doc); a false LOCAL merely degrades to the pre-existing top-k behavior. So the
# triggers are conservative and space-padded to avoid firing inside other words.
# ══════════════════════════════════════════════════════════════════════════════


class RetrievalScope(str, Enum):
    """Which slice of the corpus a query needs."""
    LOCAL = "local"    # the answer lives in a few chunks — top-k retrieval is correct
    GLOBAL = "global"  # the answer needs the WHOLE document — bypass top-k


class AnswerTask(str, Enum):
    """The shape the answer must take — drives the prompt's TASK block + scope."""
    ANSWER = "answer"        # a plain prose answer (default; local fact QA)
    ENUMERATE = "enumerate"  # list every item ("list all elves and their roles")
    TABLE = "table"          # a structured table ("map each elf to their role")
    CLASSIFY = "classify"    # group items into categories
    SUMMARIZE = "summarize"  # compress the whole document
    CRITIQUE = "critique"    # find inconsistencies / contradictions / oddities
    INFER = "infer"          # hypothetical / counterfactual reasoning over the doc


# Task triggers, checked in PRECEDENCE ORDER (most specific first). Each entry is
# (AnswerTask, tuple-of-space-padded-substrings). The first list with any hit wins.
# ENUMERATE is deliberately checked BEFORE SUMMARIZE so "list all elves in one
# sentence" reads as ENUMERATE (list every item), not SUMMARIZE (compress).
_TASK_TRIGGERS: list[tuple[AnswerTask, tuple[str, ...]]] = [
    (AnswerTask.TABLE, (
        " table ", " tabulate ", " tabular ", " as a table", " in a table",
        " columns ", " spreadsheet ",
    )),
    (AnswerTask.CLASSIFY, (
        " classify ", " categorize ", " categorise ", " categories ", " category ",
        " group them ", " group the ", " group into ", " into groups", " bucket ",
    )),
    (AnswerTask.CRITIQUE, (
        " inconsistenc", " contradict", " unusual ", " anomal", " discrepanc",
        " problems with ", " issues with ", " flaws ", " what is wrong ", " odd ",
    )),
    (AnswerTask.INFER, (
        " if ", " suppose ", " hypothetical", " what would happen", " would happen",
        " might they ", " what might ", " were added", " were removed", " if a new ",
        " based on existing", " what if ",
    )),
    (AnswerTask.ENUMERATE, (
        " list ", " all ", " every ", " each ", " enumerate ", " name all ",
        " how many ", " count ", " list out ", " give me all",
    )),
    (AnswerTask.SUMMARIZE, (
        " summarize ", " summarise ", " summary ", " tl;dr", " tldr ", " gist ",
        " overview ", " main points", " key points", " in a nutshell",
        " in one sentence", " in 2-3 sentences", " in a few sentences",
    )),
]

# Standalone GLOBAL cues that don't map to a specific task but still need the
# whole document (reconstruction / structure questions). Space-padded.
_GLOBAL_TRIGGERS: frozenset[str] = frozenset({
    " overall ", " throughout ", " entire document", " whole document",
    " across the ", " across all ", " reconstruct ", " hierarchy ", " workflow ",
    " pipeline ", " everything ", " the entire ", " in total ",
})


def classify_task(query: str) -> AnswerTask:
    """Classify the OUTPUT SHAPE the query demands (pure heuristic).

    Returns AnswerTask.ANSWER (the default) for ordinary fact questions; a more
    specific task only when a conservative, space-padded trigger fires. Precedence
    is most-specific-first (see _TASK_TRIGGERS).
    """
    q = (query or "").strip().lower()
    if not q:
        return AnswerTask.ANSWER
    padded = f" {q} "
    for task, triggers in _TASK_TRIGGERS:
        if any(t in padded for t in triggers):
            return task
    return AnswerTask.ANSWER


def classify_scope(query: str, task: AnswerTask | None = None) -> RetrievalScope:
    """Decide LOCAL (top-k) vs GLOBAL (whole-document) retrieval.

    A query is GLOBAL when EITHER its task is anything other than a plain ANSWER
    (enumeration/table/classify/summarize/critique/infer all operate over the full
    document), OR a standalone global cue fires. Otherwise LOCAL — the safe,
    cheap default the existing top-k pipeline already handles well.

    `task` may be passed in to avoid re-classifying; when omitted it is computed.
    """
    resolved_task = task if task is not None else classify_task(query)
    if resolved_task is not AnswerTask.ANSWER:
        return RetrievalScope.GLOBAL
    padded = f" {(query or '').strip().lower()} "
    if any(t in padded for t in _GLOBAL_TRIGGERS):
        return RetrievalScope.GLOBAL
    return RetrievalScope.LOCAL


# TASK INSTRUCTIONS appended to the prompt for non-ANSWER tasks. ANSWER adds
# nothing (the base instructions already cover plain QA). Each block tells the
# model the required output shape AND reinforces grounding for that shape.
_TASK_INSTRUCTIONS: dict[AnswerTask, str] = {
    AnswerTask.ANSWER: "",
    AnswerTask.ENUMERATE: (
        "List EVERY item the sources mention — be exhaustive, do not stop early or "
        "sample. If the sources name 15 items, list all 15, each with its detail. "
        "Include only items actually present in the sources."
    ),
    AnswerTask.TABLE: (
        "Output a Markdown table with one row per item. Include every item found in "
        "the sources. Use 'not specified' for an unknown cell — never invent a value. "
        "Do not omit items to keep the table short."
    ),
    AnswerTask.CLASSIFY: (
        "Group the items into the requested categories. Every item from the sources "
        "must be placed in exactly one category. State the criterion for each group. "
        "Do not add items that are not in the sources."
    ),
    AnswerTask.SUMMARIZE: (
        "Write a faithful summary in the requested length, covering the main points "
        "across the WHOLE document — not just the opening. Introduce no facts that "
        "are not in the sources."
    ),
    AnswerTask.CRITIQUE: (
        "Identify inconsistencies, contradictions, or unusual descriptions in the "
        "sources. Quote the conflicting parts and explain the tension. If you find "
        "none, say so plainly rather than inventing one."
    ),
    AnswerTask.INFER: (
        "Reason from the patterns in the sources to answer. Clearly distinguish what "
        "the sources STATE from what you are INFERRING by analogy. Base every "
        "inference on information actually in the sources; do not introduce outside "
        "entities or facts."
    ),
}


def task_instructions_for(task: AnswerTask | None) -> str:
    """The TASK INSTRUCTIONS block for a task ("" for ANSWER / None / unknown)."""
    if task is None:
        return ""
    return _TASK_INSTRUCTIONS.get(task, "")
