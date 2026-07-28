#!/usr/bin/env python3
"""Per-stage latency percentiles from recorded pipeline runs.

Closes the latency half of "cost & latency engineering". Cost was already
attributed per call (app/observability/pricing.py); this does the same for time,
and it needs no new instrumentation: every run recorded by POST /pipeline/runs
already stores a per-stage `duration_ms` inside its artifact, so the numbers come
from real executions rather than a synthetic benchmark.

p95 rather than the mean on purpose. A RAG request is a chain of network calls and
the mean hides exactly the tail a user notices; a stage whose p95 is 8x its p50 is
where a latency budget actually gets spent.

Usage:
    python scripts/latency_report.py                       # read the app database
    python scripts/latency_report.py --out eval/LATENCY.md
    python scripts/latency_report.py --db sqlite:///./data/app.db
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # runnable as `python scripts/latency_report.py`


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile. No numpy dependency, and exact on small samples
    where interpolation would invent a latency nothing actually took."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = max(1, min(len(ordered), int(round(pct / 100.0 * len(ordered) + 0.5))))
    return ordered[k - 1]


def collect(db_url: str) -> tuple[list[dict], list[float]]:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.models.pipeline_run import PipelineRun

    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    session = sessionmaker(bind=engine)()
    try:
        runs = session.query(PipelineRun).all()
        per_stage: dict[str, list[float]] = defaultdict(list)
        totals: list[float] = []
        for run in runs:
            totals.append(float(run.total_ms or 0.0))
            for step in (run.artifact or {}).get("steps", []) or []:
                label = step.get("kind") or step.get("id") or "unknown"
                per_stage[label].append(float(step.get("duration_ms") or 0.0))
        rows = [
            {
                "stage": stage,
                "n": len(vals),
                "p50": round(percentile(vals, 50), 1),
                "p95": round(percentile(vals, 95), 1),
                "max": round(max(vals), 1),
                "share": 0.0,
            }
            for stage, vals in per_stage.items()
        ]
        grand = sum(r["p50"] for r in rows) or 1.0
        for r in rows:
            r["share"] = round(100.0 * r["p50"] / grand, 1)
        rows.sort(key=lambda r: r["p95"], reverse=True)
        return rows, totals
    finally:
        session.close()
        engine.dispose()


def render(rows: list[dict], totals: list[float], db_url: str) -> str:
    L = ["# Latency budget — measured", ""]
    if not rows:
        L.append("No recorded runs found. Execute a few queries on the pipeline "
                 "dashboard (`/pipeline`) and re-run this script.")
        return "\n".join(L) + "\n"
    L.append(f"Computed from **{len(totals)} recorded pipeline runs** — real executions, "
             "not a synthetic benchmark. Every run stores a per-stage `duration_ms` in "
             "its trace artifact; this aggregates them.")
    L.append("")
    L.append(f"- **End-to-end p50:** {percentile(totals, 50):.0f} ms")
    L.append(f"- **End-to-end p95:** {percentile(totals, 95):.0f} ms")
    L.append("")
    L.append("| stage | n | p50 (ms) | p95 (ms) | max (ms) | share of p50 |")
    L.append("|---|:--:|:--:|:--:|:--:|:--:|")
    for r in rows:
        L.append(f"| {r['stage']} | {r['n']} | {r['p50']} | {r['p95']} | {r['max']} | {r['share']}% |")
    L.append("")
    L.append("### How to read this")
    L.append("")
    L.append("- **p95, not mean.** A RAG request is a chain of network calls; the mean "
             "hides the tail a user actually notices.")
    L.append("- **A stage whose p95 is many times its p50 is the latency budget's real "
             "cost centre** — that is where caching or a smaller model pays off, and it "
             "is usually an LLM call rather than retrieval.")
    L.append("- Regenerate with `python scripts/latency_report.py` after any change that "
             "could affect timing.")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-stage latency percentiles")
    ap.add_argument("--db", default=os.getenv("DATABASE_URL", "sqlite:///./data/app.db"))
    ap.add_argument("--out", default="eval/LATENCY.md")
    args = ap.parse_args()

    rows, totals = collect(args.db)
    md = render(rows, totals, args.db)
    print(md)
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
