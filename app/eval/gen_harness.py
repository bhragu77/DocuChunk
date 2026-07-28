"""
Generation-quality evaluation harness — Phase 11.

WHAT THIS IS (and how it relates to the LIVE groundedness check)
================================================================
The `/generate/answer` path already VERIFIES each answer live: groundedness_check,
validate_citations and check_verbatim protect the one answer in front of the user.
That is a smoke detector. This module is the crash-test lab: it runs a FIXED set of
questions through the REAL retrieval+generation code, scores the answers, and emits a
reproducible before/after scorecard with a pass/fail gate. Same spirit as the live
check for faithfulness; but reproducible, batched, and it adds the axis the live path
structurally CANNOT see — answer-relevancy (a true-but-off-topic answer passes
groundedness yet fails relevancy).

It is the generation twin of app/eval/harness.py (retrieval). It deliberately reuses
that module's fixture, ingestion and ground-truth resolution, and mirrors its idiom of
"prefer the real model; fall back to a clearly-labelled deterministic SURROGATE so the
harness always runs and CI stays hermetic" (harness.py's HashingBoWProvider).

METRICS (RAGAS-aligned definitions)
-----------------------------------
  faithfulness       — fraction of the answer supported by the retrieved context.
                       REAL: reuses retrieval.groundedness_check (an LLM re-check) —
                       identical to what the live path computes. SURROGATE: fraction
                       of answer content-words present in the context.
  answer_relevancy   — how well the answer addresses the QUESTION (independent of
                       truth). REAL: an LLM judge scores 0-1. SURROGATE: fraction of
                       the question's content-words covered by the answer.
  context_precision  — average precision of the RELEVANT chunks within the retrieved
                       list. DETERMINISTIC — computed from the retrieval ground truth
                       (expected chunk ids), no LLM. Meaningful even offline.
  context_recall     — fraction of the expected (relevant) chunks that were retrieved.
                       DETERMINISTIC — no LLM. This is the metric that catches a
                       retrieval regression in CI regardless of the judge.

BACKENDS (chosen by whether a real llm_fn is supplied)
------------------------------------------------------
  real     (llm_fn given) → LLMGenerator (production retrieval.generate_answer) +
                            LLMJudge. Point it at Gemini for the DEFINITIVE baseline.
  offline  (no llm_fn)    → ExtractiveGenerator + LexicalJudge + LexicalReranker.
                            Deterministic, dependency-light, no API and no torch —
                            this is what the CI gate runs. Its numbers verify harness
                            correctness and gate the DETERMINISTIC context metrics; the
                            neural faithfulness/relevancy numbers are surrogate and are
                            labelled as such in the report.

Run:
  python -m app.eval.gen_harness                    # offline surrogate → eval/GEN_BASELINE.md
  python -m app.eval.gen_harness --provider gemini  # real generation + LLM judge (needs GEMINI_API_KEY)
  python -m app.eval.gen_harness --gate             # exit non-zero if a threshold is breached
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.eval.harness import (
    EVAL_USER_ID,
    FLAGGED_CATEGORIES,
    get_eval_provider,
    ingest,
    load_eval_set,
    resolve_ground_truth,
)
from app.pipeline.embedding_providers import embed_query

logger = logging.getLogger(__name__)

DEFAULT_EVAL_SET = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "eval_set.json"
DEFAULT_BASELINE_MD = Path(__file__).resolve().parents[2] / "eval" / "GEN_BASELINE.md"
DEFAULT_BASELINE_JSON = Path(__file__).resolve().parents[2] / "eval" / "gen_baseline.json"

# Default regression-gate thresholds. Two profiles because the offline SURROGATE
# judge and the real LLM judge live on different scales (lexical overlap is inherently
# lower than a neural faithfulness score). The DETERMINISTIC context metrics use the
# SAME bar in both profiles — they don't depend on the judge.
GATE_THRESHOLDS_SURROGATE = {
    "faithfulness": 0.55,
    "answer_relevancy": 0.30,
    "context_recall": 0.80,
    "context_precision": 0.30,
}
GATE_THRESHOLDS_NEURAL = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.70,
    "context_recall": 0.80,
    "context_precision": 0.30,
}

# Failure-class CEILINGS: the gate fails if a class's rate EXCEEDS its ceiling.
#
# These are PROFILE-SPECIFIC for the same reason the metric floors are. The offline
# extractive generator only ever copies sentences out of the context, so it is
# grounded BY CONSTRUCTION and any hallucination at all is a pure regression → 0.0.
# A real LLM writes prose, and the strict "not DIRECTLY supported" test counts
# ordinary elaboration ("IP67 is an ingress protection standard…" when the source
# only states the rating) as unsupported. Demanding 0% there is not a quality bar,
# it is a permanently red light, and a gate that is always red stops being a signal.
#
# The neural ceiling is a REGRESSION bar derived from a MEASURED baseline, not an
# aspiration: the 45-query neural run scores 0.156, so 0.20 leaves room for judge
# variance while still failing the build if grounding genuinely degrades. Lowering
# this number is the goal (tighten the generation prompt against elaboration); raising
# it to accommodate a regression would defeat the purpose.
GATE_MAX_RATES_SURROGATE = {
    "hallucination": 0.0,
    "retrieval_miss": 0.10,
}
GATE_MAX_RATES_NEURAL = {
    "hallucination": 0.20,
    "retrieval_miss": 0.10,
}
# Back-compat alias: the surrogate profile is what CI runs by default.
GATE_MAX_RATES = GATE_MAX_RATES_SURROGATE

# ── Failure taxonomy (Phase 11.5) ─────────────────────────────────────────────
# The failure class is a PURE FUNCTION of the already-computed per-query signals —
# no extra model call — so it works identically offline (CI) and neural. Precedence:
# retrieval failure dominates (generation cannot succeed without evidence), then
# refusal-despite-evidence, then the generation-quality ladder.
FAILURE_CLASSES = (
    "retrieval_miss",   # the relevant chunk was never retrieved → fix search/embeddings
    "over_refusal",     # evidence WAS retrieved but the model abstained → tune confidence
    "hallucination",    # answer contains unsupported claims → tighten grounding/prompt
    "off_topic",        # true to sources but answers the WRONG question → query/prompt
    "partial_answer",   # addressed the question but incompletely → generation quality
    "ok",               # not a failure
)

# Per-profile floors used ONLY by the failure classifier (distinct from the gate's
# aggregate floors: `partial` adds a middle band the aggregate gate does not need).
FAILURE_FLOORS_SURROGATE = {"faith": 0.55, "offtopic": 0.30, "partial": 0.50}
FAILURE_FLOORS_NEURAL = {"faith": 0.85, "offtopic": 0.40, "partial": 0.70}


def classify_failure(r: "GenQueryResult", floors: dict) -> str:
    """Map one query's signals to a single failure class (first match wins)."""
    if r.context_recall == 0.0:
        return "retrieval_miss"
    if r.abstained:
        return "over_refusal"
    if r.faithfulness < floors["faith"]:
        return "hallucination"
    if r.answer_relevancy < floors["offtopic"]:
        return "off_topic"
    if r.answer_relevancy < floors["partial"]:
        return "partial_answer"
    return "ok"


# ── Content-word tokenization (shared by the surrogate generator + judge) ──────

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]*")
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "of", "to",
    "in", "on", "for", "and", "or", "with", "as", "at", "by", "from", "that", "this",
    "these", "those", "it", "its", "which", "what", "who", "how", "when", "where",
    "why", "does", "do", "did", "can", "will", "would", "should", "has", "have", "had",
    "you", "your", "i", "me", "my", "we", "our", "they", "their", "them", "he", "she",
    "his", "her", "about", "into", "than", "then", "so", "such", "not", "no", "any",
    "all", "each", "per", "use", "used", "using", "up", "out", "over", "there",
})


def _content_tokens(text: str) -> list[str]:
    """Lowercase content words (stopwords + pure punctuation removed)."""
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS]


def _sentences(text: str) -> list[str]:
    """Sentence split — nltk when available, a period split otherwise (mirrors
    groundedness_check's fallback so behaviour is consistent across the codebase)."""
    try:
        import nltk
        return [s for s in nltk.sent_tokenize(text or "") if s.strip()]
    except Exception:
        return [s.strip() for s in (text or "").split(".") if s.strip()]


# ══════════════════════════════════════════════════════════════════════════════
# Generators — turn (query, retrieved chunks) into an answer.
# The REAL one calls the SAME production code the live endpoint calls, so the eval
# measures the real behaviour; the offline one is a deterministic extractive stand-in.
# ══════════════════════════════════════════════════════════════════════════════

class Generator(Protocol):
    name: str
    semantic: bool
    def generate(self, query: str, chunks: list) -> str: ...


class LLMGenerator:
    """Production path: retrieval.generate_answer(query, chunks, llm_fn) — the exact
    grounded-prompt + provider call the live /generate/answer uses."""
    semantic = True

    def __init__(self, llm_fn: Callable[[str], str], model_name: str = "llm"):
        self._llm_fn = llm_fn
        self.name = f"llm:{model_name}"

    def generate(self, query: str, chunks: list) -> str:
        from app.pipeline.retrieval import generate_answer
        return generate_answer(query, chunks, llm_fn=self._llm_fn)


class ExtractiveGenerator:
    """Offline deterministic generator: return the retrieved sentences that overlap
    the query most. Grounded BY CONSTRUCTION (every word comes from the context), so
    it is a legitimate known-good baseline — and if retrieval regresses, its answers
    come from the wrong chunks and answer_relevancy/context metrics drop. NOT an LLM."""
    name = "extractive_surrogate"
    semantic = False

    def __init__(self, top_sentences: int = 3):
        self.top_sentences = top_sentences

    def generate(self, query: str, chunks: list) -> str:
        q = set(_content_tokens(query))
        scored: list[tuple[int, int, str]] = []
        for order, c in enumerate(chunks):
            for s in _sentences(getattr(c, "text", "")):
                overlap = len(q & set(_content_tokens(s)))
                scored.append((overlap, -order, s))  # ties: prefer higher-ranked chunk
        if not scored:
            return ""
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        picked = [s for ov, _, s in scored[: self.top_sentences] if ov > 0]
        if not picked:  # no lexical overlap anywhere — fall back to the top chunk's lead
            picked = [scored[0][2]]
        return " ".join(picked)


# ══════════════════════════════════════════════════════════════════════════════
# Judges — score faithfulness + answer_relevancy for one answer.
# ══════════════════════════════════════════════════════════════════════════════

class Judge(Protocol):
    name: str
    semantic: bool
    def faithfulness(self, query: str, answer: str, chunks: list) -> float: ...
    def answer_relevancy(self, query: str, answer: str) -> float: ...
    def answer_correctness(self, query: str, answer: str, reference: str) -> float: ...


_SCORE_RE = re.compile(r"(?<![\d.])(0?\.\d+|[01](?:\.0+)?)(?![\d.])")


class LLMJudge:
    """Real judge. Faithfulness REUSES retrieval.groundedness_check (identical to the
    live check); answer_relevancy asks the model for a 0-1 rating."""
    semantic = True

    def __init__(self, judge_fn: Callable[[str], str], model_name: str = "llm"):
        self._judge_fn = judge_fn
        self.name = f"llm:{model_name}"

    def faithfulness(self, query: str, answer: str, chunks: list) -> float:
        from app.pipeline.retrieval import groundedness_check
        result = groundedness_check(query, answer, chunks, llm_fn=self._judge_fn)
        # confidence == fraction of answer sentences supported by the sources.
        return float(result.get("confidence", 0.0)) if result.get("verified") else 0.0

    def answer_relevancy(self, query: str, answer: str) -> float:
        if not (answer or "").strip():
            return 0.0
        prompt = (
            "Rate, from 0.0 to 1.0, how directly the ANSWER addresses the QUESTION. "
            "Judge ONLY relevance to what was asked, not whether it is true. "
            "1.0 = fully answers the question; 0.0 = unrelated or evasive. "
            "Reply with ONLY the number.\n\n"
            f"QUESTION: {query}\nANSWER: {answer}\nScore:"
        )
        try:
            raw = self._judge_fn(prompt) or ""
        except Exception as exc:
            logger.warning("relevancy judge call failed: %s", exc)
            return 0.0
        m = _SCORE_RE.search(raw)
        if not m:
            return 0.0
        return max(0.0, min(1.0, float(m.group(1))))

    def answer_correctness(self, query: str, answer: str, reference: str) -> float:
        if not (answer or "").strip():
            return 0.0
        prompt = (
            "Compare the ANSWER to the REFERENCE (the correct answer). Rate from 0.0 "
            "to 1.0 how factually correct and complete the ANSWER is relative to the "
            "REFERENCE. 1.0 = same facts; 0.0 = wrong or missing. Reply with ONLY the "
            f"number.\n\nQUESTION: {query}\nREFERENCE: {reference}\nANSWER: {answer}\nScore:"
        )
        try:
            raw = self._judge_fn(prompt) or ""
        except Exception as exc:
            logger.warning("correctness judge call failed: %s", exc)
            return 0.0
        m = _SCORE_RE.search(raw)
        return max(0.0, min(1.0, float(m.group(1)))) if m else 0.0


class LexicalJudge:
    """Deterministic offline surrogate judge (lexical overlap). Clearly NOT neural —
    its numbers verify the harness and gate structural regressions, never a substitute
    for the real LLM judge's semantic scores."""
    name = "lexical_surrogate"
    semantic = False

    def faithfulness(self, query: str, answer: str, chunks: list) -> float:
        ans = _content_tokens(answer)
        if not ans:
            return 0.0
        ctx = set()
        for c in chunks:
            ctx.update(_content_tokens(getattr(c, "text", "")))
        supported = sum(1 for t in ans if t in ctx)
        return round(supported / len(ans), 4)

    def answer_relevancy(self, query: str, answer: str) -> float:
        q = set(_content_tokens(query))
        a = set(_content_tokens(answer))
        if not q:
            return 0.0
        return round(len(q & a) / len(q), 4)

    def answer_correctness(self, query: str, answer: str, reference: str) -> float:
        """Token-level F1 of the answer against the reference (surrogate for RAGAS
        answer-correctness). Balances precision (no invented facts) and recall
        (nothing missing) — a single fabricated or omitted key term is penalised."""
        a = set(_content_tokens(answer))
        r = set(_content_tokens(reference))
        if not a or not r:
            return 0.0
        tp = len(a & r)
        if tp == 0:
            return 0.0
        precision = tp / len(a)
        recall = tp / len(r)
        return round(2 * precision * recall / (precision + recall), 4)


def get_gen_backends(
    llm_fn: Callable[[str], str] | None,
    judge_fn: Callable[[str], str] | None,
    gen_model: str = "llm",
) -> tuple[Generator, Judge, bool]:
    """Pick the backend pair, mirroring harness.get_eval_provider: a real llm_fn →
    (LLMGenerator, LLMJudge, semantic=True); none → the offline surrogate pair."""
    if llm_fn is not None:
        judge = LLMJudge(judge_fn or llm_fn, model_name=gen_model)
        return LLMGenerator(llm_fn, model_name=gen_model), judge, True
    return ExtractiveGenerator(), LexicalJudge(), False


# ── Deterministic reranker for the offline path (avoids loading torch in CI) ──

class LexicalReranker:
    """A .predict(pairs)->scores reranker scoring by query/chunk content-word overlap.
    Injected into retrieval.rerank on the offline path so CI never loads the real
    cross-encoder. The real path passes reranker=None → the true cross-encoder."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        out = []
        for query, text in pairs:
            q = set(_content_tokens(query))
            t = set(_content_tokens(text))
            out.append(float(len(q & t)) if q else 0.0)
        return out


# ── Metrics ───────────────────────────────────────────────────────────────────

@dataclass
class GenQueryResult:
    id: str
    query: str
    category: str
    flagged: bool
    expected_doc_id: str
    answer: str
    reference: str | None
    contexts: list[str]              # retrieved chunk texts (truncated) — for RAGAS export
    abstained: bool
    num_retrieved: int
    num_relevant_retrieved: int
    num_expected: int
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    answer_correctness: float | None  # None when the query has no reference answer
    failure_class: str = "ok"


def _context_precision(retrieved_ids: list[str], expected: set[str]) -> float:
    """Average precision of relevant chunks within the retrieved list (RAGAS-style:
    reward relevant chunks that appear EARLY). Mean of precision@i over the ranks i
    where a relevant chunk is found; 0 if none retrieved."""
    hits = 0
    precisions: list[float] = []
    for i, cid in enumerate(retrieved_ids, 1):
        if cid in expected:
            hits += 1
            precisions.append(hits / i)
    return round(sum(precisions) / len(precisions), 4) if precisions else 0.0


def _context_recall(retrieved_ids: list[str], expected: set[str]) -> float:
    if not expected:
        return 0.0
    return round(len(set(retrieved_ids) & expected) / len(expected), 4)


def _abstained(answer: str) -> bool:
    from app.generation.citation_parser import is_abstention
    return is_abstention(answer or "")


def evaluate_gen_query(
    q: dict, chunks: list, expected: set[str], generator: Generator, judge: Judge,
    floors: dict | None = None,
) -> GenQueryResult:
    floors = floors or FAILURE_FLOORS_SURROGATE
    answer = generator.generate(q["query"], chunks)
    reference = q.get("reference_answer")
    retrieved_ids = [c.chunk_id for c in chunks]
    relevant = set(retrieved_ids) & expected
    category = q.get("category", "general")
    correctness = (
        round(float(judge.answer_correctness(q["query"], answer, reference)), 4)
        if reference else None
    )
    result = GenQueryResult(
        id=q["id"],
        query=q["query"],
        category=category,
        flagged=category in FLAGGED_CATEGORIES,
        expected_doc_id=q["expected_doc_id"],
        answer=answer,
        reference=reference,
        contexts=[(getattr(c, "text", "") or "")[:300] for c in chunks],
        abstained=_abstained(answer),
        num_retrieved=len(retrieved_ids),
        num_relevant_retrieved=len(relevant),
        num_expected=len(expected),
        faithfulness=round(float(judge.faithfulness(q["query"], answer, chunks)), 4),
        answer_relevancy=round(float(judge.answer_relevancy(q["query"], answer)), 4),
        context_precision=_context_precision(retrieved_ids, expected),
        context_recall=_context_recall(retrieved_ids, expected),
        answer_correctness=correctness,
    )
    result.failure_class = classify_failure(result, floors)
    return result


_METRIC_KEYS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def _aggregate(results: list[GenQueryResult]) -> dict:
    if not results:
        return {**{m: 0.0 for m in _METRIC_KEYS}, "answer_correctness": None}
    n = len(results)
    agg = {m: round(sum(getattr(r, m) for r in results) / n, 4) for m in _METRIC_KEYS}
    # answer_correctness is scored only for queries that carry a reference answer.
    corr = [r.answer_correctness for r in results if r.answer_correctness is not None]
    agg["answer_correctness"] = round(sum(corr) / len(corr), 4) if corr else None
    return agg


def _failure_breakdown(results: list[GenQueryResult]) -> tuple[dict, dict]:
    """(counts, rates) over every class in FAILURE_CLASSES (classes with 0 included)."""
    counts = {cls: 0 for cls in FAILURE_CLASSES}
    for r in results:
        counts[r.failure_class] = counts.get(r.failure_class, 0) + 1
    n = len(results) or 1
    rates = {cls: round(counts[cls] / n, 4) for cls in FAILURE_CLASSES}
    return counts, rates


def _by_category(results: list[GenQueryResult]) -> dict:
    cats: dict[str, list[GenQueryResult]] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r)
    return {
        cat: {"n": len(rs), "flagged": cat in FLAGGED_CATEGORIES, **_aggregate(rs)}
        for cat, rs in sorted(cats.items())
    }


# ── Orchestration ─────────────────────────────────────────────────────────────

def run_gen_eval(
    k: int = 5,
    llm_fn: Callable[[str], str] | None = None,
    judge_fn: Callable[[str], str] | None = None,
    reranker=None,
    gen_model: str = "llm",
    eval_set_path: str | Path = DEFAULT_EVAL_SET,
    persist_dir: str | Path | None = None,
) -> dict:
    """
    Ingest the fixture ONCE into an isolated temp store (never the production
    Chroma/DB), then for every query run the REAL hybrid+rerank retrieval, generate an
    answer, and score it. Returns a JSON-able report. Used by the CLI, the router and
    tests.

    llm_fn None → the deterministic OFFLINE path (extractive generator + lexical judge
    + lexical reranker): no API, no torch, fully reproducible — this is what CI runs.
    Passing llm_fn (and optionally a cheaper judge_fn) selects the real generation +
    LLM-judge path for the definitive baseline.
    """
    import chromadb

    from app.pipeline.bm25_index import BM25Index
    from app.pipeline.retrieval import hybrid_search, rerank

    eval_set = load_eval_set(eval_set_path)
    provider, is_semantic_embed = get_eval_provider()
    generator, judge, is_semantic_gen = get_gen_backends(llm_fn, judge_fn, gen_model)
    floors = FAILURE_FLOORS_NEURAL if is_semantic_gen else FAILURE_FLOORS_SURROGATE

    # Offline path defaults to the lexical reranker so CI never loads the cross-encoder.
    if reranker is None and llm_fn is None:
        reranker = LexicalReranker()

    tmp_ctx = tempfile.TemporaryDirectory(prefix="docuchunk_geneval_")
    work = Path(persist_dir) if persist_dir else Path(tmp_ctx.name)
    work.mkdir(parents=True, exist_ok=True)

    engine = create_engine(
        f"sqlite:///{work/'geneval.db'}", connect_args={"check_same_thread": False}
    )
    Session = sessionmaker(bind=engine)
    from app.models import user, document, job, chunk  # noqa: F401
    Base.metadata.create_all(bind=engine)
    chroma = chromadb.PersistentClient(path=str(work / "chroma"))
    bm25 = BM25Index(persist_dir=str(work / "bm25"))

    try:
        collection = ingest(eval_set, Session, chroma, provider, work, bm25=bm25)
        truth = resolve_ground_truth(collection, eval_set)

        def embed_fn(text: str):
            return embed_query(text, provider)

        pool = max(k * 10, 50)
        results: list[GenQueryResult] = []
        for q in eval_set["queries"]:
            cands = hybrid_search(q["query"], EVAL_USER_ID, chroma, bm25, embed_fn, top_n=pool)
            ranked = rerank(q["query"], cands, top_k=k, reranker=reranker)
            results.append(
                evaluate_gen_query(q, ranked, truth[q["id"]], generator, judge, floors=floors)
            )
    finally:
        engine.dispose()
        tmp_ctx.cleanup()

    profile = "neural" if is_semantic_gen else "surrogate"
    failure_counts, failure_rates = _failure_breakdown(results)
    return {
        "harness": "generation_quality",
        "k": k,
        "profile": profile,
        "generator": generator.name,
        "judge": judge.name,
        "embedding_provider": {
            "name": provider.provider_name,
            "model": provider.model_name,
            "semantic": is_semantic_embed,
        },
        "note": (
            "Real generation (production generate_answer) scored by an LLM judge; "
            "faithfulness reuses the live groundedness check."
            if is_semantic_gen else
            "OFFLINE SURROGATE: a deterministic extractive generator scored by a "
            "lexical-overlap judge. faithfulness/answer_relevancy are lexical, NOT "
            "neural — they verify the harness and gate structural regressions. "
            "context_precision/context_recall are deterministic (from the retrieval "
            "ground truth) and fully meaningful. Re-run with --provider gemini for the "
            "definitive generation-quality numbers."
        ),
        "num_documents": len(eval_set["documents"]),
        "num_queries": len(results),
        "num_abstained": sum(1 for r in results if r.abstained),
        "aggregate": _aggregate(results),
        "flagged_aggregate": _aggregate([r for r in results if r.flagged]),
        "by_category": _by_category(results),
        "failure_breakdown": failure_counts,
        "failure_rates": failure_rates,
        "failure_floors": floors,
        "gate_thresholds": (
            GATE_THRESHOLDS_NEURAL if is_semantic_gen else GATE_THRESHOLDS_SURROGATE
        ),
        "gate_max_rates": dict(
            GATE_MAX_RATES_NEURAL if is_semantic_gen else GATE_MAX_RATES_SURROGATE
        ),
        "queries": [asdict(r) for r in results],
    }


# ── Regression gate ───────────────────────────────────────────────────────────

def check_gate(
    report: dict, thresholds: dict | None = None, max_rates: dict | None = None,
) -> tuple[bool, list[str]]:
    """Return (passed, failures). Two kinds of check:
      * aggregate FLOORS — a metric fails when its aggregate is BELOW the floor.
      * failure-rate CEILINGS — a class fails when its rate EXCEEDS the ceiling.
    Both default to the report's own profile settings; max_rates is a no-op when the
    report carries no failure_rates (backward compatible with metric-only reports)."""
    thresholds = thresholds or report.get("gate_thresholds") or {}
    max_rates = max_rates or report.get("gate_max_rates") or {}
    agg = report.get("aggregate", {})
    rates = report.get("failure_rates", {})
    failures: list[str] = []
    for metric, floor in thresholds.items():
        value = agg.get(metric)
        if value is None:
            failures.append(f"{metric}: MISSING from report")
        elif value < floor:
            failures.append(f"{metric}: {value:.4f} < {floor:.2f} (floor)")
    for cls, ceiling in max_rates.items():
        rate = rates.get(cls, 0.0)
        if rate > ceiling:
            failures.append(f"{cls}_rate: {rate:.4f} > {ceiling:.2f} (ceiling)")
    return (not failures), failures


# ── Reporting ─────────────────────────────────────────────────────────────────

def format_gen_report(report: dict) -> str:
    agg = report["aggregate"]
    fl = report["flagged_aggregate"]
    L: list[str] = []
    L.append("# Phase 11 — Generation-quality baseline")
    L.append("")
    L.append("Generated by `python -m app.eval.gen_harness`. Same fixture and ground "
             "truth as the retrieval harness; every answer is produced by the REAL "
             "retrieval + generation path and scored on four RAGAS-aligned metrics.")
    L.append("")
    L.append(f"- **Profile:** `{report['profile']}` "
             f"(generator `{report['generator']}`, judge `{report['judge']}`)")
    ep = report["embedding_provider"]
    L.append(f"- **Embeddings:** `{ep['name']}` — `{ep['model']}` (semantic={ep['semantic']})")
    L.append(f"- **Fixture:** {report['num_documents']} documents, "
             f"{report['num_queries']} queries. **k = {report['k']}.** "
             f"Abstentions: {report['num_abstained']}.")
    L.append("")
    L.append("## Aggregate")
    L.append("")
    L.append("| metric | all queries | flagged only | deterministic? |")
    L.append("|--------|:-----------:|:------------:|:--------------:|")
    det = {"faithfulness": "no (judge)", "answer_relevancy": "no (judge)",
           "context_precision": "**yes**", "context_recall": "**yes**"}
    labels = {"faithfulness": "faithfulness", "answer_relevancy": "answer-relevancy",
              "context_precision": "context-precision", "context_recall": "context-recall"}
    for m in _METRIC_KEYS:
        L.append(f"| {labels[m]} | {agg[m]:.3f} | {fl[m]:.3f} | {det[m]} |")
    if agg.get("answer_correctness") is not None:
        flc = fl.get("answer_correctness")
        flc_s = f"{flc:.3f}" if flc is not None else "—"
        L.append(f"| answer-correctness | {agg['answer_correctness']:.3f} | {flc_s} "
                 f"| no (vs reference) |")
    L.append("")
    L.append("## Failure breakdown")
    L.append("")
    L.append("Every query is assigned exactly ONE class from its metric signals — the "
             "class names the fix, turning eval into debugging.")
    L.append("")
    counts = report.get("failure_breakdown", {})
    rates = report.get("failure_rates", {})
    routes = {
        "retrieval_miss": "the relevant chunk was never retrieved → search / embeddings",
        "over_refusal": "evidence was present but the model abstained → confidence threshold",
        "hallucination": "answer has unsupported claims → grounding / prompt",
        "off_topic": "true to sources but wrong question → query understanding / prompt",
        "partial_answer": "addressed but incomplete → generation quality",
        "ok": "not a failure",
    }
    L.append("| class | count | rate | routes to |")
    L.append("|-------|:-----:|:----:|-----------|")
    for cls in FAILURE_CLASSES:
        L.append(f"| {cls} | {counts.get(cls, 0)} | {rates.get(cls, 0.0):.3f} | {routes[cls]} |")
    L.append("")
    L.append("## By category")
    L.append("")
    L.append("| category | flagged | n | faithfulness | answer-relevancy | context-precision | context-recall |")
    L.append("|----------|:-------:|:-:|:------------:|:----------------:|:-----------------:|:--------------:|")
    for cat, m in report["by_category"].items():
        star = "yes" if m["flagged"] else "no"
        L.append(f"| {cat} | {star} | {m['n']} | {m['faithfulness']:.3f} | "
                 f"{m['answer_relevancy']:.3f} | {m['context_precision']:.3f} | "
                 f"{m['context_recall']:.3f} |")
    L.append("")
    L.append("## Regression gate")
    L.append("")
    passed, failures = check_gate(report)
    thr = report.get("gate_thresholds", {})
    mr = report.get("gate_max_rates", {})
    L.append(f"Floors (`{report['profile']}` profile): " +
             ", ".join(f"{labels[m]} ≥ {thr[m]:.2f}" for m in _METRIC_KEYS if m in thr) + ".")
    if mr:
        L.append("")
        L.append("Failure-rate ceilings: " +
                 ", ".join(f"{cls} ≤ {ceil:.2f}" for cls, ceil in mr.items()) + ".")
    L.append("")
    L.append(f"**Gate: {'PASS ✅' if passed else 'FAIL ❌'}**" +
             ("" if passed else " — " + "; ".join(failures)))
    L.append("")
    L.append("## How to read this")
    L.append("")
    L.append("- **context-recall / context-precision are deterministic** — computed "
             "from the retrieval ground truth, not a model. They are meaningful in every "
             "profile and are the metrics that catch a *retrieval* regression.")
    L.append("- **faithfulness** answers \"is the answer true to the sources?\" — the same "
             "question the live groundedness check answers, but batched and scored.")
    L.append("- **answer-relevancy** answers \"did it actually address the question?\" — the "
             "axis the live check cannot see: a true-but-off-topic answer scores high on "
             "faithfulness yet low here.")
    L.append("- **The `surrogate` profile is deterministic and hermetic** (no API, no "
             "torch) so CI can gate on it; its faithfulness/relevancy are lexical "
             "stand-ins. Re-run with `--provider gemini` for the neural numbers.")
    L.append("")
    L.append(f"> {report['note']}")
    return "\n".join(L)


# ── CLI ───────────────────────────────────────────────────────────────────────

class _SharedPacer:
    """One requests-per-minute budget shared across several callables.

    A full neural run is ~4 calls per query (1 generate + 3 judge metrics), so a
    45-query fixture fires ~180 requests. Free tiers cap well below that — Gemini's
    free tier allows 15 rpm — and an unpaced harness dies on a 429 partway through,
    which is exactly why the neural profile had never produced a committed number.

    Generator and judge draw on the SAME per-model quota, so they share one pacer;
    throttling them independently would still burst to 2x the cap. Two mechanisms,
    because either alone is insufficient:
      * PACING — hold a minimum interval (60/rpm) between calls.
      * RETRY  — on a TRANSIENT failure, back off and try again, honouring the
        server's suggested retryDelay when it supplies one.

    "Transient" covers rate limits (429/RESOURCE_EXHAUSTED) AND server-side blips
    (500/502/503/504, DEADLINE_EXCEEDED, UNAVAILABLE, INTERNAL). A full run is ~40
    minutes of continuous calls against a free-tier endpoint, so a single 504 an hour
    in would otherwise throw away the entire run — which is exactly what happened on
    the first attempt. Everything else propagates untouched: this must never mask a
    real bug in the harness or the model wiring.
    """

    _TRANSIENT = (
        "429", "resource_exhausted",
        "500", "502", "503", "504",
        "deadline_exceeded", "unavailable", "internal error",
    )

    def __init__(self, rpm: int, max_retries: int = 5):
        self._min_interval = 60.0 / rpm
        self._max_retries = max_retries
        self._last = 0.0

    @classmethod
    def _is_transient(cls, msg: str) -> bool:
        low = msg.lower()
        return any(t in low for t in cls._TRANSIENT)

    def wrap(self, fn):
        if not fn:
            return fn

        def _fn(prompt: str) -> str:
            for attempt in range(self._max_retries + 1):
                wait = self._min_interval - (time.monotonic() - self._last)
                if wait > 0:
                    time.sleep(wait)
                self._last = time.monotonic()
                try:
                    return fn(prompt)
                except Exception as exc:
                    msg = str(exc)
                    if not self._is_transient(msg):
                        raise
                    if attempt >= self._max_retries:
                        raise
                    m = re.search(r"'retryDelay':\s*'(\d+(?:\.\d+)?)s'", msg)
                    delay = float(m.group(1)) if m else self._min_interval * (2 ** attempt)
                    logger.warning(
                        "transient provider error (attempt %d/%d) — sleeping %.1fs",
                        attempt + 1, self._max_retries, delay + 1.0,
                    )
                    time.sleep(delay + 1.0)
            raise RuntimeError("unreachable")

        return _fn


def _build_provider_fns(provider_name: str, rpm: int = 0):
    """Construct (llm_fn, judge_fn, model_name) for a real provider via the same
    generation factory the app uses. Returns (None, None, 'llm') for 'offline'.

    `rpm` > 0 paces BOTH callables through one shared throttle, since generator and
    judge consume the same provider quota."""
    if provider_name == "offline":
        return None, None, "llm"
    from app.config import get_settings
    from app.generation.factory import (
        build_gen_provider, build_verify_provider, make_llm_fn, make_verify_fn,
    )
    settings = get_settings()
    settings.gen_provider = provider_name
    gen_provider = build_gen_provider(settings)
    verify_provider = build_verify_provider(settings, gen_provider)
    llm_fn = make_llm_fn(gen_provider, settings)
    judge_fn = make_verify_fn(verify_provider, settings)
    if rpm > 0:
        gate = _SharedPacer(rpm)
        llm_fn = gate.wrap(llm_fn)
        judge_fn = gate.wrap(judge_fn)
    return llm_fn, judge_fn, gen_provider.model_name


def _main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="DocuChunk generation-quality eval harness")
    parser.add_argument("--k", type=int, default=5, help="top-k for retrieval")
    parser.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET))
    parser.add_argument(
        "--provider", default="offline",
        help="'offline' (deterministic surrogate, default) | 'gemini' | 'openai_compat' | 'stub'",
    )
    parser.add_argument("--out-md", default=str(DEFAULT_BASELINE_MD))
    parser.add_argument("--out-json", default=str(DEFAULT_BASELINE_JSON))
    parser.add_argument("--no-write", action="store_true", help="print only; do not write artifacts")
    parser.add_argument(
        "--rpm", type=int, default=0,
        help="pace live provider calls to this many requests/minute (e.g. 15 for the "
             "Gemini free tier); 0 disables pacing",
    )
    parser.add_argument(
        "--gate", action="store_true",
        help="exit non-zero if any aggregate metric falls below its threshold (CI regression gate)",
    )
    args = parser.parse_args()

    llm_fn, judge_fn, model_name = _build_provider_fns(args.provider, rpm=args.rpm)
    report = run_gen_eval(
        k=args.k, llm_fn=llm_fn, judge_fn=judge_fn, gen_model=model_name,
        eval_set_path=args.eval_set,
    )
    md = format_gen_report(report)
    print(md)

    if not args.no_write:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(md)
        Path(args.out_json).write_text(json.dumps(report, indent=2))
        print(f"\nWrote baseline → {out_md}")
        print(f"Wrote JSON     → {args.out_json}")

    if args.gate:
        passed, failures = check_gate(report)
        if not passed:
            print("\nGATE FAILED:")
            for f in failures:
                print(f"  - {f}")
            return 1
        print("\nGATE PASSED ✅")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
