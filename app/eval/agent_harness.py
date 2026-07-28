"""
Phase D — Agent-vs-RAG comparison harness.

This answers the ONE question that justifies the agent's extra LLM calls: does
planned multi-hop retrieval actually produce better-grounded, more relevant,
higher-recall answers than the single-shot RAG path — and at what step cost?

It is a thin ORCHESTRATION layer over app/eval/gen_harness: same fixture, same
ingestion, same ground truth, same generator + judge + RAGAS-aligned metrics.
The ONLY thing that differs between the two arms is HOW the chunks are gathered:

  * rag arm    — hybrid_search + rerank once (exactly run_gen_eval's retrieval).
  * agent arm  — run_agent() plans multi-hop retrieval; score state.collected.

Both arms feed the identical generator/judge, so any metric delta is attributable
to retrieval strategy alone. Agent-trajectory stats (steps, tool calls, chunks
gathered, multi-hop rate) sit alongside the metric deltas.

BACKENDS (mirrors gen_harness exactly)
  offline (no llm_fn) → deterministic. The planner is a lexical DECOMPOSER
                        (make_offline_planner): it splits a compound question into
                        sub-queries and retrieves each, then finishes. No API, no
                        torch (LexicalReranker injected). This validates the
                        comparison plumbing and the multi-hop accumulation path,
                        and is what CI runs. Its metric numbers are surrogate.
  real (llm_fn given) → the REAL model plans (same llm_fn used to generate + judge).
                        Point at gemini for the definitive comparison.

Run:
  python -m app.eval.agent_harness                    # offline → eval/AGENT_BASELINE.md
  python -m app.eval.agent_harness --provider gemini  # real multi-hop planning + LLM judge
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import tempfile
from pathlib import Path
from dataclasses import asdict
from typing import Callable

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.eval.gen_harness import (
    FAILURE_CLASSES,
    FAILURE_FLOORS_NEURAL,
    FAILURE_FLOORS_SURROGATE,
    LexicalReranker,
    _METRIC_KEYS,
    _aggregate,
    _failure_breakdown,
    evaluate_gen_query,
    get_gen_backends,
)
from app.eval.harness import (
    EVAL_USER_ID,
    get_eval_provider,
    ingest,
    load_eval_set,
    resolve_ground_truth,
)
from app.generation.agent import AgentState, build_tools, run_agent
from app.pipeline.embedding_providers import embed_query

logger = logging.getLogger(__name__)

DEFAULT_EVAL_SET = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "eval_set.json"
DEFAULT_BASELINE_MD = Path(__file__).resolve().parents[2] / "eval" / "AGENT_BASELINE.md"
DEFAULT_BASELINE_JSON = Path(__file__).resolve().parents[2] / "eval" / "agent_baseline.json"


# ── Offline deterministic planner (the surrogate for a real LLM planner) ──────
# Splits a compound question into sub-queries and emits one retrieve_docs action
# per sub-query, then finish. On a simple question this is single-hop (≈ RAG); on
# a compound one it genuinely exercises the multi-hop accumulation + dedup path.

_DECOMPOSE_RE = re.compile(r"\band\b|\bversus\b|\bvs\.?\b|;|\?", re.IGNORECASE)


def decompose(question: str, max_sub: int = 3) -> list[str]:
    """Split a question into up to `max_sub` focused sub-queries (deterministic)."""
    parts = [p.strip(" ,.-") for p in _DECOMPOSE_RE.split(question or "")]
    subs: list[str] = []
    for p in parts:
        if len(p) > 3 and p.lower() not in (x.lower() for x in subs):
            subs.append(p)
    return subs[:max_sub] if subs else [(question or "").strip()]


def make_offline_planner(question: str, k: int, max_sub: int = 3) -> Callable[[str], str]:
    """A scripted planner: retrieve each sub-query (top_k=k), then finish.

    Ignores the prompt (it already knows the question) — the deterministic analog
    of an LLM reading the planner prompt and emitting JSON actions.
    """
    subs = decompose(question, max_sub)
    actions = [
        json.dumps({"tool": "retrieve_docs", "args": {"query": s, "top_k": k}})
        for s in subs
    ]
    actions.append(json.dumps({"tool": "finish", "args": {}}))
    state = {"i": 0}

    def _planner(_prompt: str) -> str:
        i = state["i"]
        state["i"] += 1
        return actions[i] if i < len(actions) else actions[-1]

    return _planner


# ── Comparison run ────────────────────────────────────────────────────────────

def _by_category(results: list) -> dict:
    """Per-category aggregates. The `multi_hop` slice is the one that matters here:
    a whole-fixture average dilutes the compound queries the agent exists to serve."""
    cats: dict[str, list] = {}
    for r in results:
        cats.setdefault(r.category, []).append(r)
    return {cat: {"n": len(rs), **_aggregate(rs)} for cat, rs in sorted(cats.items())}


def _side_summary(results: list) -> dict:
    counts, rates = _failure_breakdown(results)
    return {
        "aggregate": _aggregate(results),
        "by_category": _by_category(results),
        "num_abstained": sum(1 for r in results if r.abstained),
        "failure_breakdown": counts,
        "failure_rates": rates,
    }


def _trajectory_stats(traj: list[dict], max_steps: int) -> dict:
    n = len(traj) or 1
    return {
        "avg_steps": round(sum(t["steps"] for t in traj) / n, 3),
        "avg_tool_calls": round(sum(t["tool_calls"] for t in traj) / n, 3),
        "avg_chunks_gathered": round(sum(t["chunks_gathered"] for t in traj) / n, 3),
        "multi_hop_queries": sum(1 for t in traj if t["steps"] > 1),
        "fallback_queries": sum(1 for t in traj if t["used_fallback"]),
        "hit_max_steps": sum(1 for t in traj if t["steps"] >= max_steps),
    }


def run_agent_comparison(
    k: int = 5,
    llm_fn: Callable[[str], str] | None = None,
    judge_fn: Callable[[str], str] | None = None,
    gen_model: str = "llm",
    max_steps: int = 5,
    max_sub_queries: int = 3,
    eval_set_path: str | Path = DEFAULT_EVAL_SET,
    persist_dir: str | Path | None = None,
) -> dict:
    """Ingest the fixture once, then run BOTH retrieval strategies per query and
    score each with the same generator + judge. Returns a JSON-able comparison."""
    import chromadb

    from app.pipeline.bm25_index import BM25Index
    from app.pipeline.retrieval import hybrid_search, rerank

    eval_set = load_eval_set(eval_set_path)
    provider, is_semantic_embed = get_eval_provider()
    generator, judge, is_semantic_gen = get_gen_backends(llm_fn, judge_fn, gen_model)
    floors = FAILURE_FLOORS_NEURAL if is_semantic_gen else FAILURE_FLOORS_SURROGATE
    reranker = None if llm_fn is not None else LexicalReranker()

    tmp_ctx = tempfile.TemporaryDirectory(prefix="docuchunk_agenteval_")
    work = Path(persist_dir) if persist_dir else Path(tmp_ctx.name)
    work.mkdir(parents=True, exist_ok=True)

    engine = create_engine(
        f"sqlite:///{work/'agenteval.db'}", connect_args={"check_same_thread": False}
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

        tools = build_tools(EVAL_USER_ID, chroma, bm25, embed_fn, reranker=reranker)

        pool = max(k * 10, 50)
        rag_results, agent_results, traj = [], [], []
        for q in eval_set["queries"]:
            expected = truth[q["id"]]

            # ── RAG arm — single-shot hybrid + rerank ──────────────────────────
            cands = hybrid_search(q["query"], EVAL_USER_ID, chroma, bm25, embed_fn, top_n=pool)
            rag_ranked = rerank(q["query"], cands, top_k=k, reranker=reranker)
            rag_results.append(
                evaluate_gen_query(q, rag_ranked, expected, generator, judge, floors=floors)
            )

            # ── Agent arm — planned multi-hop retrieval ────────────────────────
            planner = (
                llm_fn if llm_fn is not None
                else make_offline_planner(q["query"], k, max_sub_queries)
            )
            state = AgentState(question=q["query"], user_id=EVAL_USER_ID, top_k=k)
            run_agent(state, tools, planner, max_steps=max_steps)
            agent_results.append(
                evaluate_gen_query(q, state.collected, expected, generator, judge, floors=floors)
            )
            ok_calls = sum(1 for h in state.history if h.status == "ok")
            traj.append({
                "id": q["id"],
                "steps": state.steps,
                "tool_calls": ok_calls,
                "chunks_gathered": len(state.collected),
                "used_fallback": any(
                    h.summary.startswith("[fallback]") for h in state.history
                ),
            })
    finally:
        engine.dispose()
        tmp_ctx.cleanup()

    rag = _side_summary(rag_results)
    agent = _side_summary(agent_results)
    deltas = {
        m: round(agent["aggregate"][m] - rag["aggregate"][m], 4) for m in _METRIC_KEYS
    }
    profile = "neural" if is_semantic_gen else "surrogate"
    return {
        "harness": "agent_vs_rag",
        "k": k,
        "max_steps": max_steps,
        "profile": profile,
        "generator": generator.name,
        "judge": judge.name,
        "planner": ("llm:" + gen_model) if llm_fn is not None else "offline_decomposer",
        "embedding_provider": {
            "name": provider.provider_name,
            "model": provider.model_name,
            "semantic": is_semantic_embed,
        },
        "note": (
            "Real multi-hop planning (the LLM chooses each retrieval) scored by an "
            "LLM judge; both arms share the generator + judge, so deltas isolate "
            "retrieval strategy."
            if is_semantic_gen else
            "OFFLINE SURROGATE: the planner is a deterministic lexical decomposer and "
            "the judge is lexical-overlap. The comparison PLUMBING and the multi-hop "
            "accumulation path are exercised and gated; the metric magnitudes are "
            "surrogate. Re-run with --provider gemini for the definitive comparison."
        ),
        "num_queries": len(rag_results),
        "rag": rag,
        "agent": agent,
        "deltas": deltas,
        "agent_trajectory": _trajectory_stats(traj, max_steps),
        "trajectory": traj,
        "rag_queries": [asdict(r) for r in rag_results],
        "agent_queries": [asdict(r) for r in agent_results],
    }


# ── Retrieval-budget sweep ────────────────────────────────────────────────────

def run_budget_sweep(
    ks: list[int],
    *,
    llm_fn=None,
    judge_fn=None,
    gen_model: str = "",
    max_steps: int = 5,
    eval_set_path: str | Path = DEFAULT_EVAL_SET,
) -> dict:
    """Run the agent-vs-RAG comparison across several retrieval budgets.

    This exists because a SINGLE k cannot answer "is the agent worth it?". Planned
    multi-hop retrieval buys coverage, and coverage only has value when the budget
    is scarce: if one top-k pass can already return every chunk the answer needs,
    decomposition adds LLM calls and dilutes precision for nothing. Sweeping k turns
    that from an opinion into a curve — the crossover point is the actual finding,
    and it is what tells you when to route a query to the agent instead of RAG.
    """
    points = []
    for k in sorted(set(ks)):
        rep = run_agent_comparison(
            k=k, llm_fn=llm_fn, judge_fn=judge_fn, gen_model=gen_model,
            max_steps=max_steps, eval_set_path=eval_set_path,
        )
        mh_rag = rep["rag"]["by_category"].get("multi_hop", {})
        mh_agent = rep["agent"]["by_category"].get("multi_hop", {})
        points.append({
            "k": k,
            "deltas": rep["deltas"],
            "rag": rep["rag"]["aggregate"],
            "agent": rep["agent"]["aggregate"],
            "multi_hop_deltas": {
                m: round(mh_agent.get(m, 0.0) - mh_rag.get(m, 0.0), 4)
                for m in _METRIC_KEYS
            } if mh_rag and mh_agent else {},
            "avg_steps": rep["agent_trajectory"]["avg_steps"],
            "avg_chunks_gathered": rep["agent_trajectory"]["avg_chunks_gathered"],
        })
    return {"harness": "agent_budget_sweep", "ks": sorted(set(ks)), "points": points}


def format_sweep_report(sweep: dict) -> str:
    L: list[str] = []
    L.append("## Retrieval-budget sweep — when is the agent worth it?")
    L.append("")
    L.append("Planned retrieval buys COVERAGE, and coverage is only worth paying for "
             "when the budget is scarce. Sweeping k makes the tradeoff explicit.")
    L.append("")
    L.append("| k | Δ context-recall | Δ context-precision | Δ answer-relevancy | "
             "avg steps | avg chunks (agent) |")
    L.append("|---|:---:|:---:|:---:|:---:|:---:|")
    for p in sweep["points"]:
        d = p["deltas"]
        L.append(
            f"| {p['k']} | {d['context_recall']:+.3f} | {d['context_precision']:+.3f} | "
            f"{d['answer_relevancy']:+.3f} | {p['avg_steps']} | {p['avg_chunks_gathered']} |"
        )
    L.append("")
    mh = [p for p in sweep["points"] if p.get("multi_hop_deltas")]
    if mh:
        L.append("### `multi_hop` queries only")
        L.append("")
        L.append("| k | Δ context-recall | Δ context-precision |")
        L.append("|---|:---:|:---:|")
        for p in mh:
            d = p["multi_hop_deltas"]
            L.append(f"| {p['k']} | {d['context_recall']:+.3f} | {d['context_precision']:+.3f} |")
        L.append("")
    return "\n".join(L)


# ── Reporting ─────────────────────────────────────────────────────────────────

_LABELS = {
    "faithfulness": "faithfulness",
    "answer_relevancy": "answer-relevancy",
    "context_precision": "context-precision",
    "context_recall": "context-recall",
}


def format_comparison_report(report: dict) -> str:
    rag = report["rag"]["aggregate"]
    agent = report["agent"]["aggregate"]
    d = report["deltas"]
    traj = report["agent_trajectory"]
    L: list[str] = []
    L.append("# Phase D — Agent vs RAG comparison")
    L.append("")
    L.append("Generated by `python -m app.eval.agent_harness`. Both arms share the "
             "same fixture, ground truth, generator and judge — only the RETRIEVAL "
             "strategy differs, so every delta is attributable to retrieval alone.")
    L.append("")
    L.append(f"- **Profile:** `{report['profile']}` (planner `{report['planner']}`, "
             f"generator `{report['generator']}`, judge `{report['judge']}`)")
    ep = report["embedding_provider"]
    L.append(f"- **Embeddings:** `{ep['name']}` — `{ep['model']}` (semantic={ep['semantic']})")
    L.append(f"- **Fixture:** {report['num_queries']} queries. "
             f"**k = {report['k']}**, **max_steps = {report['max_steps']}**.")
    L.append("")
    L.append("## Metric comparison")
    L.append("")
    L.append("| metric | RAG | Agent | Δ (agent − rag) | winner |")
    L.append("|--------|:---:|:-----:|:---------------:|:------:|")
    for m in _METRIC_KEYS:
        delta = d[m]
        if abs(delta) < 1e-9:
            winner = "tie"
        else:
            winner = "**agent**" if delta > 0 else "rag"
        sign = "+" if delta >= 0 else ""
        L.append(f"| {_LABELS[m]} | {rag[m]:.3f} | {agent[m]:.3f} | {sign}{delta:.3f} | {winner} |")
    for side, label in (("rag", "RAG"), ("agent", "Agent")):
        corr = report[side]["aggregate"].get("answer_correctness")
        if corr is not None:
            L.append(f"| answer-correctness ({label}) | — | {corr:.3f} | — | — |")
    L.append("")
    L.append("## Agent trajectory")
    L.append("")
    L.append(f"- Average steps per query: **{traj['avg_steps']}** "
             f"(tool calls: {traj['avg_tool_calls']})")
    L.append(f"- Average chunks gathered: **{traj['avg_chunks_gathered']}** "
             f"(RAG always gathers ≤ k = {report['k']})")
    L.append(f"- Multi-hop queries (>1 step): **{traj['multi_hop_queries']}** / "
             f"{report['num_queries']}")
    L.append(f"- Fallback-to-RAG invoked: {traj['fallback_queries']} · "
             f"hit max_steps: {traj['hit_max_steps']}")
    L.append("")
    L.append("## Failure breakdown (RAG → Agent)")
    L.append("")
    L.append("| class | RAG | Agent |")
    L.append("|-------|:---:|:-----:|")
    rc = report["rag"]["failure_breakdown"]
    ac = report["agent"]["failure_breakdown"]
    for cls in FAILURE_CLASSES:
        L.append(f"| {cls} | {rc.get(cls, 0)} | {ac.get(cls, 0)} |")
    L.append("")
    L.append("## How to read this")
    L.append("")
    L.append("- **The agent should never lose context-recall**: it gathers a superset "
             "of what a single retrieval finds, so recall is ≥ RAG. A drop signals a "
             "loop/guardrail bug.")
    L.append("- **context-precision can dip** for the agent — gathering more chunks over "
             "several hops can dilute precision. That is the real tradeoff this table "
             "surfaces; weigh it against the faithfulness / relevancy gains.")
    L.append("- **The `surrogate` profile validates plumbing, not magnitudes.** Re-run "
             "with `--provider gemini` for the definitive numbers, where the model does "
             "genuine semantic decomposition rather than lexical splitting.")
    L.append("")
    L.append(f"> {report['note']}")
    return "\n".join(L)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="DocuChunk agent-vs-RAG eval harness")
    parser.add_argument("--k", type=int, default=5, help="top-k per retrieval")
    parser.add_argument("--max-steps", type=int, default=5, help="agent step cap")
    parser.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET))
    parser.add_argument(
        "--provider", default="offline",
        help="'offline' (deterministic surrogate, default) | 'gemini' | 'openai_compat' | 'stub'",
    )
    parser.add_argument(
        "--rpm", type=int, default=0,
        help="pace live provider calls to this many requests/minute (e.g. 15 for the "
             "Gemini free tier); 0 disables pacing",
    )
    parser.add_argument("--out-md", default=str(DEFAULT_BASELINE_MD))
    parser.add_argument("--out-json", default=str(DEFAULT_BASELINE_JSON))
    parser.add_argument("--no-write", action="store_true", help="print only; do not write artifacts")
    parser.add_argument(
        "--sweep", default="",
        help="comma-separated k values, e.g. '1,2,3,5' — also emit the retrieval-budget "
             "sweep showing where planned retrieval stops paying for itself",
    )
    args = parser.parse_args()

    # Reuse gen_harness's provider construction (identical factory wiring).
    from app.eval.gen_harness import _build_provider_fns
    llm_fn, judge_fn, model_name = _build_provider_fns(args.provider, rpm=args.rpm)

    report = run_agent_comparison(
        k=args.k, llm_fn=llm_fn, judge_fn=judge_fn, gen_model=model_name,
        max_steps=args.max_steps, eval_set_path=args.eval_set,
    )
    md = format_comparison_report(report)

    if args.sweep:
        ks = [int(x) for x in args.sweep.split(",") if x.strip()]
        sweep = run_budget_sweep(
            ks, llm_fn=llm_fn, judge_fn=judge_fn, gen_model=model_name,
            max_steps=args.max_steps, eval_set_path=args.eval_set,
        )
        report["budget_sweep"] = sweep
        md = md.rstrip() + "\n\n" + format_sweep_report(sweep)

    print(md)

    if not args.no_write:
        out_md = Path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(md)
        Path(args.out_json).write_text(json.dumps(report, indent=2))
        print(f"\nWrote comparison → {out_md}")
        print(f"Wrote JSON       → {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
