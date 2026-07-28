#!/usr/bin/env python3
"""Validate our RAGAS-aligned metrics against the REAL RAGAS library.

Why this script exists separately from the harness
--------------------------------------------------
`app/eval/ragas_compat.py` states the claim we want to be able to make: our metric
definitions are "RAGAS-aligned (validated within ±ε)". A claim like that is only
worth making if someone actually ran the comparison — so this script runs it and
writes the deltas to an artifact.

RAGAS is deliberately NOT a project dependency (it drags in a LangChain stack that
conflicts with this repo's pins, and the CI gate must stay hermetic). So this runs
from its OWN virtualenv and imports `ragas_compat` by file path — that module has no
module-level imports beyond `logging`, precisely so it can be loaded like this.

Setup (one time):
    python3 -m venv .ragas-venv
    .ragas-venv/bin/pip install "ragas==0.4.3" datasets "langchain-community<0.4" \
        langchain-google-genai

Run:
    GEMINI_API_KEY=... .ragas-venv/bin/python scripts/ragas_crosscheck.py \
        --report eval/gen_baseline.json --out eval/RAGAS_CROSSCHECK.md
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def load_ragas_compat():
    """Import app/eval/ragas_compat.py without importing the `app` package."""
    path = REPO / "app" / "eval" / "ragas_compat.py"
    spec = importlib.util.spec_from_file_location("ragas_compat", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_gemini(model: str, embed_model: str, rpm: int):
    """RAGAS-compatible (llm, embeddings) backed by Gemini, paced for the free tier."""
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise SystemExit("GEMINI_API_KEY (or GOOGLE_API_KEY) must be set")
    os.environ.setdefault("GOOGLE_API_KEY", key)

    chat = ChatGoogleGenerativeAI(
        model=model, google_api_key=key, temperature=0.0, max_retries=6,
    )
    emb = GoogleGenerativeAIEmbeddings(model=embed_model, google_api_key=key)
    return LangchainLLMWrapper(chat), LangchainEmbeddingsWrapper(emb)


def run_config(rpm: int):
    """Serialize RAGAS and give it a generous timeout.

    RAGAS fans metric jobs out concurrently (max_workers defaults to 16). Against a
    free tier that instantly trips the per-minute cap, and every in-flight job then
    dies on a TimeoutError — the run reports nothing at all. max_workers=1 plus a
    long timeout makes the cross-check slow but actually completable. RunConfig has
    no rate knob of its own, so pacing comes from serialising plus tenacity retries.
    """
    from ragas.run_config import RunConfig
    return RunConfig(timeout=300, max_workers=1, max_retries=8, max_wait=120)


def format_report(cc: dict, report: dict, model: str) -> str:
    L = ["# RAGAS cross-check", ""]
    L.append("Our harness computes RAGAS-*aligned* metrics with its own implementations. "
             "This artifact scores the SAME run with the real `ragas` library and reports "
             "the gap, so \"RAGAS-aligned\" is a measured claim rather than a naming choice.")
    L.append("")
    L.append(f"- **Source report:** `{report.get('harness', '?')}`, "
             f"profile `{report.get('profile', '?')}`, {report.get('num_queries', '?')} queries")
    L.append(f"- **RAGAS judge model:** `{model}`")
    L.append("")

    if not cc.get("available"):
        L.append(f"> NOT RUN — {cc.get('note', 'ragas unavailable')}")
        return "\n".join(L) + "\n"
    if cc.get("error"):
        L.append(f"> RAGAS evaluate failed: `{cc['error']}`")
        return "\n".join(L) + "\n"

    ours, theirs, delta = cc.get("ours", {}), cc.get("ragas", {}), cc.get("delta_vs_ours", {})
    L.append("| metric | ours | RAGAS | |Δ| |")
    L.append("|---|:---:|:---:|:---:|")
    for m in sorted(set(ours) | set(theirs)):
        o = ours.get(m)
        r = theirs.get(m)
        d = delta.get(m)
        L.append(
            f"| {m} | {o if o is not None else '—'} | {r if r is not None else '—'} | "
            f"{d if d is not None else '—'} |"
        )
    L.append("")
    if cc.get("max_delta") is not None:
        L.append(f"**Max absolute deviation: {cc['max_delta']}**")
        L.append("")
    L.append("### How to read this")
    L.append("")
    L.append("- `context_precision` / `context_recall` are computed by us DETERMINISTICALLY "
             "from retrieval ground truth, while RAGAS infers relevance with an LLM — some "
             "gap here is expected and is not an error in either implementation.")
    L.append("- `faithfulness` / `answer_relevancy` are LLM-judged on both sides, so they "
             "carry the judge's own variance; agreement within a few points is the "
             "realistic bar, not equality.")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Cross-check our metrics against real RAGAS")
    ap.add_argument("--report", default="eval/gen_baseline.json",
                    help="a gen_harness JSON report (neural profile preferred)")
    ap.add_argument("--out", default="eval/RAGAS_CROSSCHECK.md")
    ap.add_argument("--out-json", default="eval/ragas_crosscheck.json")
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--embed-model", default="models/gemini-embedding-001")
    ap.add_argument("--rpm", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N rows (smoke-test without burning quota)")
    args = ap.parse_args()

    report = json.loads((REPO / args.report).read_text())
    if args.limit:
        report = dict(report)
        report["queries"] = report.get("queries", [])[: args.limit]
        report["num_queries"] = len(report["queries"])

    compat = load_ragas_compat()
    llm, emb = build_gemini(args.model, args.embed_model, args.rpm)
    cc = compat.run_ragas_crosscheck(
        report, llm=llm, embeddings=emb, run_config=run_config(args.rpm)
    )

    md = format_report(cc, report, args.model)
    print(md)
    (REPO / args.out).write_text(md)
    (REPO / args.out_json).write_text(json.dumps(cc, indent=2))
    print(f"Wrote {args.out}")
    print(f"Wrote {args.out_json}")
    return 0 if cc.get("available") and not cc.get("error") else 1


if __name__ == "__main__":
    sys.exit(main())
