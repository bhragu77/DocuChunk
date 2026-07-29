#!/usr/bin/env python3
"""Decision-level cost and latency report from recorded pipeline runs.

`latency_report.py` answers "where does the time go". This answers the question an
engineer actually has to decide on: **what does a query cost, in money and in
milliseconds, and which lever moves it.**

Nothing new is instrumented. Every pipeline run already stores per-stage
`duration_ms` and per-call `input_tokens` / `output_tokens` / `cost_usd` /
`cost_inr` in its artifact — this aggregates what is already there and groups it by
the dimensions that drive a decision:

  * by MODEL         — the tier choice (a cheap model vs an accurate one)
  * by VECTOR BACKEND — the storage choice
  * by STAGE          — where to optimise first
  * per 1,000 queries — the unit a budget is actually written in

Unpriced models are reported as unknown rather than as zero. A blank cost and a
free call must never look alike; that ambiguity is exactly how a model switch
silently removed cost attribution from this system.

Run:
    python scripts/cost_latency_report.py
    python scripts/cost_latency_report.py --out eval/COST_LATENCY.md
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile — exact on small samples, where interpolation would
    invent a latency that nothing actually took."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(1, min(len(ordered), int(round(pct / 100.0 * len(ordered) + 0.5))))
    return ordered[k - 1]


def collect(db_url: str) -> dict:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.models.pipeline_run import PipelineRun

    kw = {"connect_args": {"check_same_thread": False}} if db_url.startswith("sqlite") else {}
    engine = create_engine(db_url, **kw)
    session = sessionmaker(bind=engine)()
    try:
        runs = session.query(PipelineRun).all()
        per_stage_ms: dict[str, list[float]] = defaultdict(list)
        per_stage_cost: dict[str, list[float]] = defaultdict(list)
        by_model: dict[str, list[dict]] = defaultdict(list)
        by_backend: dict[str, list[float]] = defaultdict(list)
        rows = []

        for run in runs:
            art = run.artifact or {}
            totals = art.get("totals") or {}
            rows.append({
                "model": run.model_name or "unknown",
                "provider": run.provider or "",
                "total_ms": float(run.total_ms or 0.0),
                "input_tokens": totals.get("input_tokens"),
                "output_tokens": totals.get("output_tokens"),
                "cost_inr": totals.get("cost_inr"),
                "cost_usd": totals.get("cost_usd"),
                "backend": totals.get("vector_backend") or "unrecorded",
                "multi_hop": bool(totals.get("multi_hop")),
                "status": run.status,
            })
            by_model[run.model_name or "unknown"].append(rows[-1])
            by_backend[rows[-1]["backend"]].append(float(run.total_ms or 0.0))
            for step in art.get("steps", []) or []:
                label = step.get("kind") or step.get("id") or "unknown"
                per_stage_ms[label].append(float(step.get("duration_ms") or 0.0))
                detail = step.get("detail") or {}
                c = detail.get("cost") or {}
                inr = c.get("cost_inr") if isinstance(c, dict) else None
                if inr is None:
                    inr = detail.get("cost_inr")
                if inr is not None:
                    per_stage_cost[label].append(float(inr))

        return {
            "runs": rows,
            "per_stage_ms": dict(per_stage_ms),
            "per_stage_cost": dict(per_stage_cost),
            "by_model": dict(by_model),
            "by_backend": dict(by_backend),
        }
    finally:
        session.close()
        engine.dispose()


def _cost_cell(vals: list[float | None]) -> str:
    """Average cost, distinguishing 'free' from 'not priced'."""
    known = [v for v in vals if v is not None]
    if not known:
        return "unknown"
    avg = sum(known) / len(known)
    if avg == 0:
        return "₹0 (free tier / local)"
    return f"₹{avg:.4f}"


def render(data: dict, db_url: str) -> str:
    runs = data["runs"]
    L = ["# Cost & latency — measured", ""]
    if not runs:
        L.append("No recorded runs. Execute queries on `/pipeline` and re-run this script.")
        return "\n".join(L) + "\n"

    L.append(f"Aggregated from **{len(runs)} recorded pipeline runs** — real executions, "
             "not a synthetic benchmark. Every run already stores per-stage timing and "
             "per-call token usage and cost; this groups them by the dimensions that "
             "drive a decision.")
    L.append("")

    # ── headline ──
    all_ms = [r["total_ms"] for r in runs]
    L.append("## End-to-end")
    L.append("")
    L.append(f"- **p50:** {percentile(all_ms, 50):.0f} ms · **p95:** {percentile(all_ms, 95):.0f} ms "
             f"· **max:** {max(all_ms):.0f} ms")
    L.append(f"- **Cost per query (avg):** {_cost_cell([r['cost_inr'] for r in runs])}")
    known_costs = [r["cost_inr"] for r in runs if r["cost_inr"] is not None]
    if known_costs:
        L.append(f"- **Cost per 1,000 queries:** ₹{(sum(known_costs)/len(known_costs))*1000:.2f}")
    L.append("")

    # ── by model tier ──
    L.append("## By model tier")
    L.append("")
    L.append("| model | runs | p50 | p95 | avg tokens in/out | cost/query | cost/1k queries |")
    L.append("|---|:--:|:--:|:--:|:--:|:--:|:--:|")
    for model, rs in sorted(data["by_model"].items()):
        ms = [r["total_ms"] for r in rs]
        ti = [r["input_tokens"] for r in rs if r["input_tokens"] is not None]
        to = [r["output_tokens"] for r in rs if r["output_tokens"] is not None]
        costs = [r["cost_inr"] for r in rs]
        known = [c for c in costs if c is not None]
        per_1k = f"₹{(sum(known)/len(known))*1000:.2f}" if known else "unknown"
        L.append(
            f"| `{model}` | {len(rs)} | {percentile(ms,50):.0f} ms | {percentile(ms,95):.0f} ms | "
            f"{int(statistics.mean(ti)) if ti else '—'} / {int(statistics.mean(to)) if to else '—'} | "
            f"{_cost_cell(costs)} | {per_1k} |"
        )
    L.append("")

    # ── by stage ──
    L.append("## By stage — where time and money go")
    L.append("")
    L.append("| stage | n | p50 | p95 | max | share of p50 | avg cost |")
    L.append("|---|:--:|:--:|:--:|:--:|:--:|:--:|")
    stage_rows = []
    for stage, vals in data["per_stage_ms"].items():
        stage_rows.append({
            "stage": stage, "n": len(vals),
            "p50": percentile(vals, 50), "p95": percentile(vals, 95), "max": max(vals),
            "cost": data["per_stage_cost"].get(stage, []),
        })
    grand = sum(r["p50"] for r in stage_rows) or 1.0
    stage_rows.sort(key=lambda r: r["p95"], reverse=True)
    for r in stage_rows:
        share = 100.0 * r["p50"] / grand
        L.append(f"| {r['stage']} | {r['n']} | {r['p50']:.1f} ms | {r['p95']:.1f} ms | "
                 f"{r['max']:.1f} ms | {share:.1f}% | {_cost_cell(r['cost'])} |")
    L.append("")

    # ── by backend ──
    if len(data["by_backend"]) > 1 or "unrecorded" not in data["by_backend"]:
        L.append("## By vector backend")
        L.append("")
        L.append("| backend | runs | p50 end-to-end | p95 end-to-end |")
        L.append("|---|:--:|:--:|:--:|")
        for b, ms in sorted(data["by_backend"].items()):
            L.append(f"| {b} | {len(ms)} | {percentile(ms,50):.0f} ms | {percentile(ms,95):.0f} ms |")
        L.append("")
        L.append("> Retrieval is a small fraction of end-to-end time — see "
                 "`eval/BACKEND_COMPARISON.md` for the vector call measured in isolation.")
        L.append("")

    # ── reading guide ──
    L.append("## How to read this")
    L.append("")
    L.append("- **p95, not mean.** A RAG request is a chain of network calls; the mean "
             "hides the tail a user actually feels.")
    L.append("- **`unknown` cost is not free cost.** A model with no price entry reports "
             "unknown. Treating that as zero is how cost attribution silently "
             "disappears when a model name changes.")
    L.append("- **The LLM dominates.** Retrieval is single-digit milliseconds "
             "(`eval/BACKEND_COMPARISON.md`); generation and verification are seconds. "
             "Optimising retrieval further buys almost nothing — the levers that matter "
             "are model tier, caching, and cutting the number of LLM calls per answer.")
    L.append("- **The agent multiplies cost.** Each planner step is a full LLM call. "
             "`eval/AGENT_BASELINE.md` shows its recall benefit vanishes once the "
             "retrieval budget is generous, so routing every query through the agent "
             "pays for coverage that is already there.")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Cost & latency decision report")
    ap.add_argument("--db", default=os.getenv("DATABASE_URL", "sqlite:///./data/app.db"))
    ap.add_argument("--out", default="eval/COST_LATENCY.md")
    ap.add_argument("--out-json", default="eval/cost_latency.json")
    args = ap.parse_args()

    data = collect(args.db)
    md = render(data, args.db)
    print(md)
    (REPO / args.out).parent.mkdir(parents=True, exist_ok=True)
    (REPO / args.out).write_text(md)
    (REPO / args.out_json).write_text(json.dumps(
        {"runs": data["runs"], "per_stage_ms": data["per_stage_ms"]}, indent=2, default=str))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
