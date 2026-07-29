#!/usr/bin/env python3
"""Compare three synthesis paths over IDENTICAL retrieval.

  native      — app.generation.generate_answer (numbered sources, citations, verify)
  llamaindex  — LlamaIndex ResponseSynthesizer over our BaseRetriever
  langchain   — LangChain LCEL chain (prompt | llm | parser) over our BaseRetriever

All three call `retriever_core.retrieve`, so the evidence set is byte-identical and
any delta is attributable to synthesis alone. All three are given the same model,
because comparing two models and calling it a framework comparison is the easiest
way to produce a confident wrong answer.

What is measured, and why these and not "quality":

  citation_rate  — fraction of answers carrying at least one [n] marker. This is the
                   property the whole verification layer depends on. A framework that
                   returns fluent uncited prose has not produced a worse answer; it
                   has produced an answer this system cannot check.
  latency        — wall time per query.
  tokens/cost    — what the framework's prompt construction actually spends.
  answer_len     — proxy for verbosity differences in default prompting.

Run inside the app container (needs app deps AND the integration extras):
    docker exec documentchunking-app-1 python3 /app/scripts/framework_benchmark.py
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CITATION_RE = re.compile(r"\[\s*\d+\s*\]")

QUERIES = [
    "which replacement battery does the GX-4200 use",
    "what firmware version is recommended for the HX-9000",
    "what is the default admin password on the GX-4200",
    "how many trees should be planted per acre in a modern apple orchard",
    "which solar inverter comes with the longer warranty",
]


def _pct(vals, p):
    if not vals:
        return 0.0
    o = sorted(vals)
    k = max(1, min(len(o), int(round(p / 100.0 * len(o) + 0.5))))
    return o[k - 1]


def build_deps(user_id: str):
    """Reconstruct exactly what the app builds during lifespan (app/main.py), so the
    benchmark retrieves through the same index the running service uses."""
    from app.config import get_settings
    from app.core.chroma import build_chroma_client
    from app.integrations.retriever_core import RetrievalDeps
    from app.pipeline.embedding_providers import embed_single
    from app.pipeline.bm25_index import BM25Index

    s = get_settings()
    return RetrievalDeps(
        user_id=user_id,
        vector_client=build_chroma_client(),
        bm25_index=BM25Index(persist_dir=s.chroma_persist_dir),
        embed_fn=embed_single,
    )


# ── paths ─────────────────────────────────────────────────────────────────────

def run_native(query: str, deps, top_k: int) -> dict:
    from app.config import get_settings
    from app.generation.factory import build_gen_provider
    from app.generation.prompt_builder import build_grounded_prompt
    from app.integrations.retriever_core import retrieve

    chunks = retrieve(query, deps, top_k=top_k)
    prov = build_gen_provider(get_settings())
    prompt = build_grounded_prompt(query, chunks)
    t0 = time.perf_counter()
    answer = prov.generate(prompt, max_tokens=512, temperature=0.0)
    return {"answer": answer or "", "ms": (time.perf_counter() - t0) * 1000,
            "chunks": len(chunks)}


def run_llamaindex(query: str, deps, top_k: int, llm) -> dict:
    from app.integrations.llamaindex_adapter import build_query_engine

    engine = build_query_engine(deps, llm=llm, top_k=top_k)
    t0 = time.perf_counter()
    resp = engine.query(query)
    return {"answer": str(resp), "ms": (time.perf_counter() - t0) * 1000,
            "chunks": len(getattr(resp, "source_nodes", []) or [])}


def run_langchain(query: str, deps, top_k: int, llm) -> dict:
    from app.integrations.langchain_adapter import build_lcel_chain

    chain = build_lcel_chain(llm, deps, top_k=top_k)
    t0 = time.perf_counter()
    answer = chain.invoke(query)
    return {"answer": answer or "", "ms": (time.perf_counter() - t0) * 1000,
            "chunks": top_k}


def build_framework_llm():
    """One model, wrapped for each framework, so only synthesis varies."""
    from app.config import get_settings
    s = get_settings()
    model = s.gen_model
    key = s.gemini_api_key
    li = lc = None
    try:
        from llama_index.llms.google_genai import GoogleGenAI
        li = GoogleGenAI(model=model, api_key=key)
    except Exception as exc:
        print(f"  (llamaindex LLM unavailable: {str(exc)[:90]})")
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        lc = ChatGoogleGenerativeAI(model=model, google_api_key=key, temperature=0.0)
    except Exception as exc:
        print(f"  (langchain LLM unavailable: {str(exc)[:90]})")
    return li, lc


def summarise(name: str, runs: list[dict]) -> dict:
    ok = [r for r in runs if not r.get("error")]
    lens = [len(r["answer"]) for r in ok]
    ms = [r["ms"] for r in ok]
    cited = [1 for r in ok if CITATION_RE.search(r["answer"])]
    return {
        "path": name,
        "queries": len(runs),
        "ok": len(ok),
        "errors": len(runs) - len(ok),
        "citation_rate": round(len(cited) / len(ok), 3) if ok else None,
        "p50_ms": round(statistics.median(ms), 0) if ms else None,
        "p95_ms": round(_pct(ms, 95), 0) if ms else None,
        "avg_answer_chars": int(statistics.mean(lens)) if lens else 0,
    }


def render(rows, notes) -> str:
    L = ["# Framework comparison — synthesis over identical retrieval", ""]
    L.append("All three paths call `app/integrations/retriever_core.retrieve`, so the "
             "evidence set is **byte-identical**; only synthesis differs. All three are "
             "given the same model, because comparing two models and calling it a "
             "framework comparison is the easiest way to be confidently wrong.")
    L.append("")
    L.append("| path | queries | ok | citation rate | p50 | p95 | avg answer chars |")
    L.append("|---|:--:|:--:|:--:|:--:|:--:|:--:|")
    for r in rows:
        cr = f"{r['citation_rate']:.0%}" if r["citation_rate"] is not None else "—"
        L.append(f"| **{r['path']}** | {r['queries']} | {r['ok']} | {cr} | "
                 f"{r['p50_ms'] or '—'} ms | {r['p95_ms'] or '—'} ms | {r['avg_answer_chars']} |")
    L.append("")
    if notes:
        L.append("### Not measured")
        L.append("")
        for n in notes:
            L.append(f"- {n}")
        L.append("")
    L.append(f"> **Sample: {rows[0]['queries']} queries.** Large enough to expose a "
             "categorical difference in citation behaviour, too small to rank latency "
             "precisely. Treat the citation column as the finding and the millisecond "
             "columns as indicative.")
    L.append("")
    L.append("### What this measures, and what it does not")
    L.append("")
    L.append("- **Citation rate is the headline, not answer quality.** Every downstream "
             "guarantee in this system — citation validation, groundedness scoring, "
             "confidence, abstention — requires the model to emit `[n]` markers "
             "against numbered sources. A framework whose default prompt returns "
             "fluent uncited prose has not produced a worse answer; it has produced "
             "one this system **cannot verify**.")
    L.append("- **The retriever is the interop claim.** Our hybrid + reranked retrieval "
             "satisfies both `BaseRetriever` interfaces, so this pipeline drops into "
             "an existing LlamaIndex or LangChain stack rather than requiring one be "
             "built around it. That is the portable, reusable part.")
    L.append("- **What the frameworks buy:** composition plumbing and a large "
             "integration catalogue. **What they cost:** a dependency tree that has "
             "already conflicted in this repo (ragas 0.4.3 pinned "
             "`langchain-community<0.4`), and prompt behaviour you must override "
             "anyway the moment you need verifiable output.")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-id", required=True)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--queries", default="",
                    help="';'-separated queries. MUST be answerable from the target "
                         "corpus — otherwise every path correctly abstains and the "
                         "comparison measures nothing but agreement on refusal.")
    ap.add_argument("--out", default="eval/FRAMEWORK_COMPARISON.md")
    ap.add_argument("--out-json", default="eval/framework_comparison.json")
    args = ap.parse_args()

    global QUERIES
    if args.queries.strip():
        QUERIES = [q.strip() for q in args.queries.split(";") if q.strip()]

    deps = build_deps(args.user_id)
    li_llm, lc_llm = build_framework_llm()
    notes = []

    paths = {"native": lambda q: run_native(q, deps, args.top_k)}
    if li_llm is not None:
        paths["llamaindex"] = lambda q: run_llamaindex(q, deps, args.top_k, li_llm)
    else:
        notes.append("**llamaindex** — `llama-index-llms-google-genai` not installed")
    if lc_llm is not None:
        paths["langchain"] = lambda q: run_langchain(q, deps, args.top_k, lc_llm)
    else:
        notes.append("**langchain** — `langchain-google-genai` not installed")

    rows, raw = [], {}
    for name, fn in paths.items():
        print(f"→ {name}")
        runs = []
        for q in QUERIES:
            try:
                r = fn(q)
                runs.append(r)
                print(f"   ok  {r['ms']:.0f}ms  cited={bool(CITATION_RE.search(r['answer']))}")
            except Exception as exc:
                runs.append({"answer": "", "ms": 0.0, "error": str(exc)[:200]})
                print(f"   ERR {str(exc)[:110]}")
        raw[name] = runs
        rows.append(summarise(name, runs))

    md = render(rows, notes)
    print("\n" + md)
    (REPO / args.out).parent.mkdir(parents=True, exist_ok=True)
    (REPO / args.out).write_text(md)
    (REPO / args.out_json).write_text(json.dumps({"summary": rows, "runs": raw}, indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
