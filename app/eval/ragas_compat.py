"""
RAGAS interoperability (Phase 11.5).

The point of this module is NOT to rename our metric keys to look like RAGAS — that
would prove nothing. It is to make our harness *interoperate with and validate against*
RAGAS: export our runs in RAGAS's dataset schema so the SAME fixture can be scored by
the real RAGAS library, and cross-check that our numbers track theirs. That is what
earns the claim "metrics are RAGAS-aligned (validated within ±ε)".

  to_ragas_dataset      — our report → RAGAS/HF `Dataset` rows
                          ({question, answer, contexts, reference?}).
  export_ragas_format   — the convenience metric-name view (the weak-but-handy shim).
  run_ragas_crosscheck  — OPTIONAL: if `ragas` is installed and an LLM/embeddings are
                          supplied, score the exported dataset with real RAGAS and
                          report the per-metric delta vs ours. No-ops cleanly otherwise.

`ragas` is intentionally NOT in requirements.txt — the cross-check is a developer /
demo tool (`pip install ragas datasets`), kept out of the hermetic CI path.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Our metric name → the RAGAS metric that measures the same thing.
_METRIC_ALIGNMENT = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer_relevancy",
    "context_precision": "context_precision",
    "context_recall": "context_recall",
    "answer_correctness": "answer_correctness",
}


def to_ragas_dataset(report: dict) -> list[dict]:
    """Our gen-eval report → a list of rows in RAGAS's expected schema. `reference` is
    included only for queries that carried a gold answer (RAGAS's reference-based
    metrics — context_recall, answer_correctness — need it; the rest do not)."""
    rows: list[dict] = []
    for q in report.get("queries", []):
        row = {
            "question": q.get("query", ""),
            "answer": q.get("answer", ""),
            "contexts": list(q.get("contexts", []) or []),
        }
        reference = q.get("reference")
        if reference:
            row["reference"] = reference
            row["ground_truth"] = reference  # RAGAS <0.2 field name, for compatibility
        rows.append(row)
    return rows


def export_ragas_format(report: dict) -> dict:
    """The simple aggregate view keyed by RAGAS metric names. Convenience only — the
    real interop artifact is to_ragas_dataset + run_ragas_crosscheck."""
    agg = report.get("aggregate", {})
    return {
        ragas_name: agg.get(ours)
        for ours, ragas_name in _METRIC_ALIGNMENT.items()
        if agg.get(ours) is not None
    }


def _extract_scores(result) -> dict:
    """Pull {metric: mean_score} out of a RAGAS EvaluationResult, across versions.

    `dict(result)` works on older RAGAS but raises KeyError on 0.4.x, where
    __getitem__ is ROW-indexed rather than metric-indexed. Prefer the internal
    per-metric score dict, then fall back to the dataframe.
    """
    scores = getattr(result, "_scores_dict", None)
    if isinstance(scores, dict) and scores:
        out = {}
        for name, val in scores.items():
            vals = [
                v for v in (val if isinstance(val, list) else [val])
                if isinstance(v, (int, float))
            ]
            if vals:
                out[name] = round(sum(vals) / len(vals), 4)
        if out:
            return out
    try:
        import numbers
        df = result.to_pandas()
        return {
            col: round(float(df[col].dropna().mean()), 4)
            for col in df.columns
            if df[col].dropna().size
            and isinstance(df[col].dropna().iloc[0], numbers.Number)
        }
    except Exception:  # pragma: no cover - depends on the installed RAGAS
        logger.warning("could not extract RAGAS scores from %r", type(result))
        return {}


def run_ragas_crosscheck(report: dict, llm=None, embeddings=None, run_config=None) -> dict:
    """Score the exported dataset with the REAL RAGAS library and report how far its
    numbers sit from ours. Returns a status dict; never raises on a missing dependency.

    Requires `pip install ragas datasets` and, for the LLM-judged metrics, an `llm` +
    `embeddings` RAGAS understands (e.g. LangChain wrappers). Metrics whose inputs are
    absent (no reference) are skipped by RAGAS itself.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy, context_precision, context_recall, faithfulness,
        )
    except ImportError as exc:
        return {
            "available": False,
            "note": f"RAGAS not installed ({exc}). `pip install ragas datasets` to enable "
                    "the cross-check; it is deliberately not a project dependency.",
        }

    rows = to_ragas_dataset(report)
    if not rows:
        return {"available": True, "note": "no queries in report", "ragas": {}, "delta_vs_ours": {}}

    dataset = Dataset.from_list(rows)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    try:
        kwargs = {"llm": llm, "embeddings": embeddings}
        if run_config is not None:
            kwargs["run_config"] = run_config
        result = evaluate(dataset, metrics=metrics, **kwargs)
    except Exception as exc:  # RAGAS/provider runtime error — report, do not crash the caller
        logger.warning("RAGAS evaluate failed: %s", exc)
        return {"available": True, "error": str(exc)}

    ragas_scores = _extract_scores(result)
    if not ragas_scores:
        return {"available": True, "error": "RAGAS returned no parseable scores"}
    ours = report.get("aggregate", {})
    delta = {}
    for ours_name, ragas_name in _METRIC_ALIGNMENT.items():
        o, r = ours.get(ours_name), ragas_scores.get(ragas_name)
        if o is not None and r is not None:
            delta[ragas_name] = round(abs(o - r), 4)
    return {
        "available": True,
        "ragas": ragas_scores,
        "ours": {a: ours.get(o) for o, a in _METRIC_ALIGNMENT.items() if ours.get(o) is not None},
        "delta_vs_ours": delta,
        "max_delta": round(max(delta.values()), 4) if delta else None,
    }
