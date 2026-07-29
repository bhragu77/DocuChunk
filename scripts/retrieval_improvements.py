#!/usr/bin/env python3
"""Measure MMR and multi-query expansion against the hybrid+rerank baseline.

Four configurations over the SAME fixture, ground truth and embeddings:

    baseline    hybrid + cross-encoder rerank          (eval/phase8_comparison.md)
    +rewrite    multi-query expansion, RRF-fused
    +mmr        MMR diversification over the reranked pool
    +both

**Per-category results are the point, not the aggregate.** Both techniques are
expected to help some categories and hurt others, and an aggregate average hides
exactly that. The stated hypotheses:

  * expansion should HURT `identifier` queries — "88-AZ-0097" is already the exact
    token BM25 needs, so every paraphrase is strictly worse and fusing bad lists
    with the good one demotes the correct hit;
  * expansion should HELP `ambiguous_entity` and `multi_hop`, where the user's
    wording and the document's wording differ;
  * MMR should HELP `multi_hop` coverage by breaking up blocks of near-duplicates,
    and slightly hurt precision on single-fact lookups.

A negative result is a valid outcome and is published as-is.

Run:
    python scripts/retrieval_improvements.py --out eval/RETRIEVAL_IMPROVEMENTS.md
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CONFIGS = ["baseline", "+rewrite", "+mmr", "+both"]

# Free-tier flash-lite allows 15 requests/minute; 4.2s spacing stays inside it.
RPM_INTERVAL_S = 4.2


def build_llm_fn():
    """Rewriter model. Falls back to a deterministic stub so the harness still runs
    (and is honestly labelled) with no API key."""
    from app.config import get_settings
    from app.generation.factory import build_gen_provider

    s = get_settings()
    try:
        prov = build_gen_provider(s)
        state = {"last": 0.0}

        def _fn(prompt: str) -> str:
            # Free-tier models cap REQUESTS PER MINUTE (15 on the flash-lite tier),
            # not only per day. Without pacing the harness 429s, rewrite_query
            # swallows the error and falls back to the original query -- so the
            # "+rewrite" config would silently BECOME the baseline and the whole
            # comparison would report a meaningless "no difference".
            for attempt in range(4):
                wait = RPM_INTERVAL_S - (time.monotonic() - state["last"])
                if wait > 0:
                    time.sleep(wait)
                state["last"] = time.monotonic()
                try:
                    return prov.generate(prompt, max_tokens=160, temperature=0.0) or ""
                except Exception as exc:
                    if "429" not in str(exc) or attempt == 3:
                        raise
                    time.sleep(RPM_INTERVAL_S * (attempt + 2))
            return ""

        _fn.label = f"{s.gen_provider}/{s.gen_model}"
        return _fn
    except Exception as exc:
        print(f"  (no generation provider: {str(exc)[:80]} — expansion disabled)")
        return None


def run(k: int, eval_set_path: str, out_md: str, out_json: str) -> int:
    import chromadb

    from app.eval.harness import (
        EVAL_USER_ID,
        _evaluate_with,
        _mode_block,
        get_eval_provider,
        ingest,
        load_eval_set,
        resolve_ground_truth,
    )
    from app.pipeline.bm25_index import BM25Index
    from app.pipeline.embedding_providers import embed_query
    from app.pipeline.mmr import mmr_rerank
    from app.pipeline.query_rewriter import fuse_ranked_lists, rewrite_query
    from app.pipeline.retrieval import hybrid_search, rerank
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from app.database import Base

    eval_set = load_eval_set(eval_set_path)
    provider, is_semantic = get_eval_provider()
    llm_fn = build_llm_fn()

    tmp = tempfile.TemporaryDirectory(prefix="docuchunk_improve_")
    work = Path(tmp.name)
    engine = create_engine(f"sqlite:///{work/'imp.db'}", connect_args={"check_same_thread": False})
    Session = sessionmaker(bind=engine)
    from app.models import user, document, job, chunk  # noqa: F401
    Base.metadata.create_all(bind=engine)
    chroma = chromadb.PersistentClient(path=str(work / "chroma"))
    bm25 = BM25Index(persist_dir=str(work / "bm25"))

    timings: dict[str, list[float]] = defaultdict(list)

    try:
        collection = ingest(eval_set, Session, chroma, provider, work, bm25=bm25)
        truth = resolve_ground_truth(collection, eval_set)

        def embed_fn(text: str):
            return embed_query(text, provider)

        pool = max(k * 10, 50)

        def _search(q: str):
            return hybrid_search(q, EVAL_USER_ID, chroma, bm25, embed_fn, top_n=pool)

        def make(cfg: str):
            use_rw = cfg in ("+rewrite", "+both") and llm_fn is not None
            use_mmr = cfg in ("+mmr", "+both")

            def _retrieve(q):
                t0 = time.perf_counter()
                query = q["query"]
                if use_rw:
                    variants = rewrite_query(query, llm_fn, n=3)
                    per_variant = [[c.chunk_id for c in _search(v)] for v in variants]
                    order = fuse_ranked_lists(per_variant)
                    by_id = {c.chunk_id: c for v in variants for c in _search(v)}
                    cands = [by_id[cid] for cid in order if cid in by_id]
                else:
                    cands = _search(query)

                ranked = rerank(query, cands, top_k=(pool if use_mmr else k))
                if use_mmr:
                    from app.config import get_settings
                    ranked = mmr_rerank(ranked, embed_fn, top_k=k,
                                        lambda_mult=get_settings().mmr_lambda)
                timings[cfg].append((time.perf_counter() - t0) * 1000)
                return [c.chunk_id for c in ranked], (ranked[0].doc_id if ranked else None)

            return _retrieve

        results = {}
        for cfg in CONFIGS:
            if cfg in ("+rewrite", "+both") and llm_fn is None:
                print(f"→ {cfg}: SKIPPED (no generation provider)")
                continue
            print(f"→ {cfg}")
            results[cfg] = _evaluate_with(make(cfg), eval_set, truth, k)
            agg = _mode_block(results[cfg])["aggregate"]
            print(f"   MRR={agg['mrr']:.3f} recall@{k}={agg['recall_at_k']:.3f}")
    finally:
        engine.dispose()
        tmp.cleanup()

    blocks = {cfg: _mode_block(rs) for cfg, rs in results.items()}
    md = render(blocks, timings, k, is_semantic, getattr(llm_fn, "label", None))
    print("\n" + md)
    (REPO / out_md).parent.mkdir(parents=True, exist_ok=True)
    (REPO / out_md).write_text(md)
    (REPO / out_json).write_text(json.dumps(
        {"k": k, "semantic": is_semantic, "configs": blocks}, indent=2, default=str))
    print(f"Wrote {out_md}")
    return 0


def _cat_table(blocks: dict, key: str, k: int) -> list[str]:
    cats = sorted({c for b in blocks.values() for c in (b.get("by_category") or {})})
    if not cats:
        return []
    L = [f"| category | " + " | ".join(blocks) + " |",
         "|---|" + "|".join([":--:"] * len(blocks)) + "|"]
    base = next(iter(blocks))
    for cat in cats:
        cells = []
        for cfg, b in blocks.items():
            v = (b.get("by_category") or {}).get(cat, {}).get(key)
            if v is None:
                cells.append("—")
            elif cfg == base:
                cells.append(f"{v:.3f}")
            else:
                bv = (blocks[base].get("by_category") or {}).get(cat, {}).get(key)
                d = v - bv if bv is not None else 0.0
                arrow = "▲" if d > 0.001 else ("▼" if d < -0.001 else "=")
                cells.append(f"{v:.3f} {arrow}{abs(d):.3f}" if arrow != "=" else f"{v:.3f} =")
        L.append(f"| {cat} | " + " | ".join(cells) + " |")
    return L


def render(blocks: dict, timings: dict, k: int, semantic: bool, llm_label) -> str:
    L = ["# Retrieval improvements — measured", ""]
    L.append(f"Same fixture, same ground truth, same embeddings; k = {k}. "
             f"Retrieval profile: **{'neural' if semantic else 'lexical surrogate'}**."
             + (f" Rewriter: `{llm_label}`." if llm_label else ""))
    L.append("")
    L.append("## Aggregate")
    L.append("")
    L.append("| config | MRR | recall@k | precision@k | p50 retrieval |")
    L.append("|---|:--:|:--:|:--:|:--:|")
    for cfg, b in blocks.items():
        a = b["aggregate"]
        ts = sorted(timings.get(cfg, []))
        p50 = f"{ts[len(ts)//2]:.0f} ms" if ts else "—"
        L.append(f"| **{cfg}** | {a['mrr']:.3f} | {a['recall_at_k']:.3f} | "
                 f"{a['precision_at_k']:.3f} | {p50} |")
    L.append("")
    L.append("## Per category — MRR")
    L.append("")
    L.append("Aggregates hide the trade. Both techniques are expected to help some "
             "categories and hurt others; this table is where the decision is made.")
    L.append("")
    L.extend(_cat_table(blocks, "mrr", k))
    L.append("")
    L.append("## Per category — recall@k")
    L.append("")
    L.extend(_cat_table(blocks, "recall_at_k", k))
    L.append("")
    L.append("## Verdict")
    L.append("")
    L.append("If every category sits at the ceiling in the baseline, a null result "
             "measures the FIXTURE, not the technique — an evaluation whose baseline "
             "saturates cannot detect an improvement. Check the per-category tables "
             "above before concluding anything from the aggregate.")
    L.append("")
    L.append("## Hypotheses under test")
    L.append("")
    L.append("- **Expansion should hurt `identifier`.** A part number is already the "
             "exact token BM25 needs; every paraphrase is strictly worse, and fusing "
             "bad lists with the good one demotes the correct hit.")
    L.append("- **Expansion should help `ambiguous_entity` / `multi_hop`,** where the "
             "user's wording and the document's wording differ.")
    L.append("- **MMR should help `multi_hop` coverage** by breaking up blocks of "
             "near-duplicates so a second fact fits in the budget, and slightly hurt "
             "precision on single-fact lookups.")
    L.append("")
    L.append("A negative result is a valid outcome and is published unchanged. Both "
             "features ship **disabled by default** (`QUERY_REWRITE_ENABLED`, "
             "`MMR_ENABLED`); the table above is what an operator would use to decide "
             "whether their query mix justifies the cost — expansion adds one LLM call "
             "per query, MMR adds one embedding per candidate.")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--eval-set", default=str(REPO / "tests/fixtures/eval_set.json"))
    ap.add_argument("--out", default="eval/RETRIEVAL_IMPROVEMENTS.md")
    ap.add_argument("--out-json", default="eval/retrieval_improvements.json")
    args = ap.parse_args()
    return run(args.k, args.eval_set, args.out, args.out_json)


if __name__ == "__main__":
    raise SystemExit(main())
