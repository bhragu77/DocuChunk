#!/usr/bin/env python3
"""Benchmark every vector backend on the SAME fixture, ground truth and embeddings.

The point is not "which is fastest". Retrieval quality is expected to be nearly
identical — all three run the same embeddings through the same cosine metric — and
that expectation is itself the finding: **if quality is equal, the decision is made
on operational cost, not on MRR.** A benchmark that only reported latency would
invite the wrong conclusion (managed services always lose on network round-trip),
so this reports quality AND latency AND ops burden together.

Only the storage layer varies. Chunking, embeddings, ground truth and metrics are
byte-identical across runs, so any delta is attributable to the backend alone.

Run:
    python scripts/backend_benchmark.py                      # every reachable backend
    python scripts/backend_benchmark.py --backends pgvector,chroma
    python scripts/backend_benchmark.py --k 5 --out eval/BACKEND_COMPARISON.md
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


# ── backend factories ─────────────────────────────────────────────────────────

def _chroma_factory(work: Path):
    import chromadb
    return chromadb.PersistentClient(path=str(work / "chroma"))


def _pgvector_factory(work: Path):
    """Isolated schema-per-run would be ideal; instead we use a dedicated collection
    namespace and clear it, so a benchmark never disturbs application data."""
    from app.config import get_settings
    from app.pipeline.pgvector_store import PgVectorClient

    s = get_settings()
    dsn = s.pgvector_dsn or s.database_url
    if dsn.startswith("sqlite"):
        raise RuntimeError("pgvector backend needs a Postgres DATABASE_URL")
    return PgVectorClient(dsn)


def _pinecone_factory(work: Path):
    from app.config import get_settings
    from app.pipeline.pinecone_store import PineconeClient

    s = get_settings()
    if not s.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY not set")
    return PineconeClient(
        api_key=s.pinecone_api_key,
        index_name=s.pinecone_index,
        cloud=s.pinecone_cloud,
        region=s.pinecone_region,
    )


FACTORIES = {
    "chroma": (_chroma_factory, "self-hosted (embedded)"),
    "pgvector": (_pgvector_factory, "self-hosted (Postgres)"),
    "pinecone": (_pinecone_factory, "managed (serverless)"),
}


# ── timing ────────────────────────────────────────────────────────────────────

class TimedClient:
    """Wraps a vector client and records the wall time of every `query` call.

    Timing is taken around the collection call itself, so it excludes embedding and
    reranking — otherwise the model would dominate and the backends would look
    identical no matter what.
    """

    def __init__(self, inner):
        self._inner = inner
        self.query_ms: list[float] = []

    def _wrap(self, coll):
        outer = self

        class _TimedCollection:
            def __getattr__(self, item):
                return getattr(coll, item)

            def query(self, *a, **kw):
                t0 = time.perf_counter()
                try:
                    return coll.query(*a, **kw)
                finally:
                    outer.query_ms.append((time.perf_counter() - t0) * 1000)

        return _TimedCollection()

    def get_collection(self, name):
        return self._wrap(self._inner.get_collection(name))

    def get_or_create_collection(self, name, **kw):
        return self._wrap(self._inner.get_or_create_collection(name, **kw))

    def __getattr__(self, item):
        return getattr(self._inner, item)


def _pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    ordered = sorted(vals)
    idx = max(1, min(len(ordered), int(round(p / 100.0 * len(ordered) + 0.5))))
    return ordered[idx - 1]


# ── run ───────────────────────────────────────────────────────────────────────

def run_backend(name: str, k: int, eval_set: str, warmup: bool) -> dict:
    from app.eval.harness import run_eval

    factory, ops = FACTORIES[name]
    timed: dict = {}

    def wrapped_factory(work):
        client = TimedClient(factory(work))
        timed["client"] = client
        return client

    if warmup:
        # Serverless backends cold-start; an unwarmed first query inflates p95 and
        # would misrepresent steady-state latency.
        try:
            run_eval(k=k, eval_set_path=eval_set, client_factory=lambda w: factory(w))
        except Exception as exc:
            print(f"  warmup failed ({exc}) — continuing")

    t0 = time.perf_counter()
    report = run_eval(k=k, eval_set_path=eval_set, client_factory=wrapped_factory)
    wall = time.perf_counter() - t0

    q = timed.get("client").query_ms if timed.get("client") else []
    agg = report["aggregate"]
    return {
        "backend": name,
        "ops": ops,
        "mrr": agg["mrr"],
        "recall_at_k": agg["recall_at_k"],
        "precision_at_k": agg["precision_at_k"],
        "queries_timed": len(q),
        "p50_ms": round(statistics.median(q), 1) if q else None,
        "p95_ms": round(_pct(q, 95), 1) if q else None,
        "mean_ms": round(statistics.mean(q), 1) if q else None,
        "total_wall_s": round(wall, 1),
    }


def render(rows: list[dict], k: int, failures: dict[str, str]) -> str:
    L = ["# Vector backend comparison", ""]
    L.append(f"Identical fixture, ground truth, embeddings and metrics — **only the "
             f"storage layer varies**, so every delta is attributable to the backend. "
             f"k = {k}.")
    L.append("")
    L.append("| backend | ops burden | MRR | recall@k | precision@k | p50 query | p95 query |")
    L.append("|---|---|:--:|:--:|:--:|:--:|:--:|")
    for r in rows:
        p50 = f"{r['p50_ms']} ms" if r["p50_ms"] is not None else "—"
        p95 = f"{r['p95_ms']} ms" if r["p95_ms"] is not None else "—"
        L.append(f"| **{r['backend']}** | {r['ops']} | {r['mrr']:.3f} | "
                 f"{r['recall_at_k']:.3f} | {r['precision_at_k']:.3f} | {p50} | {p95} |")
    L.append("")

    if failures:
        L.append("### Not measured")
        L.append("")
        for name, why in failures.items():
            L.append(f"- **{name}** — {why}")
        L.append("")

    L.append("### How to read this")
    L.append("")
    L.append("- **Retrieval quality should be near-identical.** All backends score the "
             "same embeddings with the same cosine metric, so a large MRR gap would "
             "indicate an adapter bug, not a better database. Equal quality is the "
             "expected — and useful — result.")
    L.append("- **Latency is measured around the vector call only**, excluding "
             "embedding and reranking, which would otherwise dominate and mask the "
             "difference.")
    L.append("- **A managed service pays a network round-trip** that an embedded or "
             "same-host database does not. That is the cost of not operating it — "
             "read the latency column against the ops column, not on its own.")
    L.append("- **The decision:** when quality is equal, choose on operational burden, "
             "consistency guarantees and scale ceiling. pgvector keeps rows and "
             "vectors in one transaction boundary; Pinecone removes index operations "
             "entirely; Chroma is simplest to start with and hardest to scale.")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark vector backends on one fixture")
    ap.add_argument("--backends", default="chroma,pgvector,pinecone")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--eval-set", default=str(REPO / "tests/fixtures/eval_set.json"))
    ap.add_argument("--out", default="eval/BACKEND_COMPARISON.md")
    ap.add_argument("--out-json", default="eval/backend_comparison.json")
    ap.add_argument("--no-warmup", action="store_true")
    args = ap.parse_args()

    wanted = [b.strip() for b in args.backends.split(",") if b.strip()]
    rows, failures = [], {}
    for name in wanted:
        if name not in FACTORIES:
            failures[name] = "unknown backend"
            continue
        print(f"→ {name} ...", flush=True)
        try:
            row = run_backend(name, args.k, args.eval_set, warmup=not args.no_warmup)
            rows.append(row)
            print(f"  MRR={row['mrr']:.3f} recall={row['recall_at_k']:.3f} "
                  f"p50={row['p50_ms']}ms p95={row['p95_ms']}ms")
        except Exception as exc:
            failures[name] = str(exc)[:200]
            print(f"  SKIPPED: {str(exc)[:160]}")

    if not rows:
        print("no backend produced results")
        return 1

    md = render(rows, args.k, failures)
    print("\n" + md)
    (REPO / args.out).parent.mkdir(parents=True, exist_ok=True)
    (REPO / args.out).write_text(md)
    (REPO / args.out_json).write_text(json.dumps(
        {"k": args.k, "results": rows, "not_measured": failures}, indent=2))
    print(f"Wrote {args.out}")
    print(f"Wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
