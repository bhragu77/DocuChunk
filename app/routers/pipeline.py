"""
Pipeline dashboard — record-then-replay of the REAL RAG/agent pipeline.

POST /pipeline/runs runs one query through the agent pipeline ONCE, times every
stage, meters token/₹ cost on every LLM call, captures the retrieved evidence and
the groundedness verdict, and serializes all of it into a durable trace ARTIFACT
persisted to the `pipeline_runs` table. The dashboard then replays that artifact.

"Faithful per query" is guaranteed by construction: we serialize what executed, we
never synthesize. The run is wrapped in `span("pipeline.request")`, so when Langfuse
tracing is enabled the SAME run also lands in Langfuse (one source, two sinks) and
we keep its trace id for a deep-link.

Endpoints:
  POST   /pipeline/runs        — run + record; returns the artifact
  GET    /pipeline/runs        — gallery list (newest first, denormalized summary)
  GET    /pipeline/runs/{id}   — full artifact for replay
  DELETE /pipeline/runs/{id}   — permanent delete
  GET    /pipeline/examples    — curated "Try these" queries that exercise the graph
"""
from __future__ import annotations

import logging
import os
from time import perf_counter
from typing import Callable

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.dependencies import get_chroma, get_current_user
from app.core.timefmt import iso_utc
from app.database import get_db
from app.generation.agent import AgentState, agent_events, build_tools
from app.generation.base import GenerationError
from app.generation.query_classifier import classify_query, classify_task
from app.generation.usage import get_last_usage
from app.models.pipeline_run import PipelineRun
from app.models.user import User
from app.observability import current_trace_id, hash_user_id, span
from app.observability.pricing import cost_usd, to_inr
from app.pipeline.retrieval import ScoredChunk, generate_answer
from app.routers.agent import _AGENT_SCOPE, _citations
from app.routers.search import (
    GenerateAnswerRequest,
    _attach_verify_scores,
    _build_answer_response,
    _get_bm25,
    _get_embed_fn,
)

logger = logging.getLogger(__name__)

pipeline_router = APIRouter(prefix="/pipeline", tags=["pipeline"])


# ── curated "Try these" queries ─────────────────────────────────────────────────
# Chosen to EXERCISE the full graph — a boring query renders as one flat line. Each
# one is honest about what it should trigger; the dashboard shows these as chips.
EXAMPLE_QUERIES = [
    {
        "id": "multi_hop",
        "label": "Multi-hop compare",
        "query": "Compare the earliest and the latest key points in the document and explain how they differ.",
        "why": "Forces the agent to retrieve twice (two hops) and synthesize — the money shot.",
        "expect": "2+ retrieve_docs steps",
    },
    {
        "id": "whole_doc",
        "label": "Whole-document tool",
        "query": "Summarize the entire document end to end, covering every major section.",
        "why": "A whole-document task the agent answers with the fetch_document tool, not top-k search.",
        "expect": "fetch_document tool call",
    },
    {
        "id": "abstention",
        "label": "Abstention / low confidence",
        "query": "What is the author's personal phone number and home address?",
        "why": "Not in the corpus — a healthy pipeline abstains instead of hallucinating.",
        "expect": "abstain / low groundedness",
    },
]


# ── request / response shapes ───────────────────────────────────────────────────

class RunSummary(BaseModel):
    """One row in the run gallery (denormalized — no artifact parse needed)."""
    run_id: str
    query: str
    provider: str
    model_name: str
    status: str
    total_ms: float
    cost_inr: float | None
    step_count: int
    multi_hop: bool
    confidence: float | None
    created_at: str


# ── metered LLM wrapper (one source, two sinks) ─────────────────────────────────

class _Metered:
    """Wrap an llm_fn so every call's real token usage + ₹ cost is captured for the
    artifact — reading the SAME per-request usage contextvar the Langfuse seam reads
    (the provider writes it; make_llm_fn records it for Langfuse; we read it here)."""

    def __init__(self, fn: Callable[[str], str] | None, model_name: str | None):
        self._fn = fn
        self.model_name = model_name or ""
        self.calls: list[dict] = []

    def __call__(self, prompt: str) -> str:
        out = self._fn(prompt) if self._fn else ""
        usage = get_last_usage() or {}
        it, ot = usage.get("input_tokens"), usage.get("output_tokens")
        usd = cost_usd(self.model_name, it, ot)
        self.calls.append({
            "input_tokens": it, "output_tokens": ot,
            "cost_usd": usd, "cost_inr": to_inr(usd),
        })
        return out

    @staticmethod
    def totals(calls: list[dict]) -> dict:
        ti = sum(c["input_tokens"] or 0 for c in calls)
        to = sum(c["output_tokens"] or 0 for c in calls)
        usd_vals = [c["cost_usd"] for c in calls if c["cost_usd"] is not None]
        usd = round(sum(usd_vals), 8) if usd_vals else None
        return {
            "input_tokens": ti, "output_tokens": to,
            "cost_usd": usd, "cost_inr": to_inr(usd),
        }


def _stage(sid: str, kind: str, label: str, status_: str, duration_ms: float, detail: dict) -> dict:
    return {
        "id": sid, "kind": kind, "label": label, "status": status_,
        "duration_ms": round(duration_ms, 1), "detail": detail,
    }


def _chunk_view(c: ScoredChunk) -> dict:
    return {
        "chunk_id": c.chunk_id,
        "source": c.source,
        "page": c.page_number,
        "fused_score": round(getattr(c, "fused_score", 0.0) or 0.0, 4),
        "reranker_score": round(getattr(c, "reranker_score", 0.0) or 0.0, 4),
        "excerpt": (c.text or "")[:220],
    }


def _resolve_model(request: Request, model_key: str | None):
    """Pick the generation tier from the model picker registry, else the default
    seam. Returns (llm_fn, model_name, provider_key)."""
    registry = getattr(request.app.state, "gen_registry", None) or {}
    if model_key and model_key in registry:
        entry = registry[model_key]
        return entry["llm_fn"], entry["model_name"], model_key
    return (
        getattr(request.app.state, "llm_fn", None),
        getattr(request.app.state, "gen_model_name", None) or "unknown",
        "default",
    )


def _langfuse_link() -> dict | None:
    """A Langfuse deep-link only when tracing is actually on (else the trace id is
    seeded locally but was never sent anywhere — a dead link)."""
    if os.getenv("TRACING_ENABLED", "").strip().lower() != "true":
        return None
    host = os.getenv("LANGFUSE_HOST", "").strip() or "https://cloud.langfuse.com"
    tid = current_trace_id()
    if not tid:
        return None
    return {"trace_id": tid, "host": host.rstrip("/")}


def _user_documents(db: Session, user_id: str, scope_doc_id: str | None) -> list[dict]:
    """The user's ready documents [{doc_id, name}] — shown to the agent planner so it
    calls fetch_document with a real id. Scoped to one doc when the run is."""
    from app.models.document import Document, DocumentStatus
    q = db.query(Document).filter(
        Document.user_id == user_id, Document.status == DocumentStatus.ready
    )
    if scope_doc_id:
        q = q.filter(Document.id == scope_doc_id)
    docs = q.order_by(Document.created_at.desc()).limit(25).all()
    return [{"doc_id": d.id, "name": d.original_filename} for d in docs]


def _record_run(
    request: Request, current_user: User, chroma, bm25, embed_fn,
    query: str, doc_id: str | None, top_k: int, model_key: str | None,
    available_docs: list[dict] | None = None,
) -> dict:
    """Run the pipeline once and build the trace artifact. This is the 'record' half
    of record-then-replay — every number in the artifact comes from this execution."""
    settings = get_settings()
    llm_fn, model_name, provider_key = _resolve_model(request, model_key)
    if llm_fn is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail="generation_not_configured — set GEN_PROVIDER (or enable a model tier).",
        )
    verify_model = getattr(request.app.state, "verify_model_name", None) or model_name
    verify_fn = getattr(request.app.state, "verify_fn", None) or llm_fn

    metered = _Metered(llm_fn, model_name)
    verify_metered = _Metered(verify_fn, verify_model)

    allowed = set(settings.agent_allowed_tools_list) or None
    docs = available_docs or []
    tools = build_tools(
        current_user.id, chroma, bm25, embed_fn, doc_id=doc_id, available_docs=docs,
    )

    steps: list[dict] = []
    t0 = perf_counter()

    with span(
        "pipeline.request",
        user=hash_user_id(current_user.id), doc_id=doc_id, top_k=top_k,
        provider=provider_key, model=model_name,
    ) as root:
        # ── stage: query classification (the router) ──────────────────────────
        s = perf_counter()
        query_type = classify_query(query)
        answer_task = classify_task(query)
        steps.append(_stage(
            "router", "router", "Query classified", "ok",
            (perf_counter() - s) * 1000,
            {"query_type": query_type.value, "answer_task": answer_task.value},
        ))

        # ── stage(s): the agent loop, one timeline row per executed step ───────
        state = AgentState(
            question=query, user_id=current_user.id, doc_id=doc_id, top_k=top_k,
            available_docs=docs,
        )
        gen = agent_events(
            state, tools, metered,
            max_steps=settings.agent_max_steps, allowed_tools=allowed,
        )
        prev = 0
        with span("agent", question_len=len(query)) as asp:
            while True:
                s = perf_counter()
                try:
                    rec = next(gen)
                except StopIteration:
                    break
                except GenerationError as exc:
                    # A planner LLM call failed (e.g. Gemini 504 / rate limit). The
                    # agent fires several sequential model calls, so a transient
                    # failure on any of them must NOT 500 the run — record it as an
                    # error stage and stop planning with whatever we've gathered.
                    dur = (perf_counter() - s) * 1000
                    logger.warning("pipeline: agent planning failed: %s", exc)
                    steps.append(_stage(
                        f"agent-{state.steps + 1}", "agent", "Agent planning failed",
                        "error", dur, {"error": str(exc)[:300]},
                    ))
                    break
                dur = (perf_counter() - s) * 1000
                step_calls = metered.calls[prev:]
                prev = len(metered.calls)
                is_fallback = str(rec.summary or "").startswith("[fallback]")
                steps.append(_stage(
                    f"agent-{rec.n}", "fallback" if is_fallback else "agent",
                    f"Step {rec.n}: {rec.tool}", rec.status, dur,
                    {
                        "n": rec.n, "tool": rec.tool, "args": rec.args,
                        "summary": rec.summary, "observation": (rec.observation or "")[:600],
                        "new_chunks": rec.data.get("_new_chunks"),
                        "error": rec.error,
                        "cost": _Metered.totals(step_calls),
                    },
                ))
            asp.update(steps=state.steps, collected=len(state.collected))

        tool_ok = sum(1 for st in steps if st["kind"] in ("agent", "fallback") and st["status"] == "ok")
        citations = _citations(state.collected)

        answer_draft = None
        gen_calls: list[dict] = []
        verdict_status = "ok"

        # ── stage: generation (only if the agent gathered evidence) ───────────
        if state.collected:
            thin = len(state.collected) < 2
            gen_idx = len(metered.calls)
            s = perf_counter()
            try:
                with span("generate", model=model_name) as gsp:
                    answer_draft = generate_answer(
                        query, state.collected, llm_fn=metered, doc_chunks_fetcher=None,
                        query_type=query_type, thin_context=thin,
                        answer_task=answer_task, model_name=model_name,
                    )
                    gsp.update(collected=len(state.collected))
                gdur = (perf_counter() - s) * 1000
                gen_calls = metered.calls[gen_idx:]
                gtot = _Metered.totals(gen_calls)
                steps.append(_stage(
                    "generate", "generate", "Generate grounded answer", "ok", gdur,
                    {"model": model_name, "provider": provider_key,
                     "input_tokens": gtot["input_tokens"], "output_tokens": gtot["output_tokens"],
                     "cost_inr": gtot["cost_inr"]},
                ))
            except GenerationError as exc:
                gdur = (perf_counter() - s) * 1000
                verdict_status = "error"
                steps.append(_stage(
                    "generate", "generate", "Generate grounded answer", "error", gdur,
                    {"model": model_name, "error": str(exc)[:300]},
                ))
        else:
            verdict_status = "error"
            steps.append(_stage(
                "generate", "generate", "Generate grounded answer", "error", 0.0,
                {"error": "no_relevant_content — the agent gathered no evidence"},
            ))

        # ── stage: verification (validation + groundedness + confidence) ──────
        base = None
        if answer_draft is not None:
            s = perf_counter()
            with span("verify") as vsp:
                base = _build_answer_response(
                    query, state.collected, citations, answer_draft, verify_metered,
                    query_type.value, _AGENT_SCOPE, answer_task,
                )
                _attach_verify_scores(vsp, base)
            vdur = (perf_counter() - s) * 1000
            if base.abstained:
                verdict_status = "abstained"
            steps.append(_stage(
                "verify", "verify", "Verify groundedness", "ok", vdur,
                {"grounded": bool(base.grounded), "confidence": round(base.confidence or 0.0, 3),
                 "verified": bool(base.verified), "abstained": bool(base.abstained),
                 "unsupported_claims": base.unsupported_claims or []},
            ))
            root.update(abstained=bool(base.abstained), steps=state.steps)

        total_ms = (perf_counter() - t0) * 1000
        langfuse = _langfuse_link()

    # ── assemble the artifact ───────────────────────────────────────────────────
    all_gen_calls = metered.calls
    grand = _Metered.totals(all_gen_calls + verify_metered.calls)
    verdict = {
        "status": verdict_status,
        "grounded": bool(base.grounded) if base else False,
        "confidence": round(base.confidence, 3) if base else 0.0,
        "verified": bool(base.verified) if base else False,
        "abstained": bool(base.abstained) if base else False,
        "signals": (base.confidence_signals or {}) if base else {},
    }
    artifact = {
        "query": query,
        "doc_id": doc_id,
        "provider": provider_key,
        "model_name": model_name,
        "provenance": f"recorded · {provider_key}",
        "totals": {
            "latency_ms": round(total_ms, 1),
            "input_tokens": grand["input_tokens"],
            "output_tokens": grand["output_tokens"],
            "cost_usd": grand["cost_usd"],
            "cost_inr": grand["cost_inr"],
            "steps": tool_ok,
            "tool_calls": tool_ok,
            "chunks": len(state.collected),
            "multi_hop": tool_ok > 1,
        },
        "verdict": verdict,
        "answer": base.answer if base else None,
        "citations": [
            {"index": i + 1, "source": c.source, "page": c.page_number,
             "excerpt": c.text_excerpt}
            for i, c in enumerate(base.citations)
        ] if base else [],
        "chunks": [_chunk_view(c) for c in state.collected],
        "steps": steps,
        "langfuse": langfuse,
    }
    return artifact


# ── endpoints ────────────────────────────────────────────────────────────────

@pipeline_router.get("/examples")
def list_examples():
    """The curated 'Try these' queries (chips) that exercise the full graph."""
    return {"examples": EXAMPLE_QUERIES}


@pipeline_router.post("/runs")
def create_run(
    payload: GenerateAnswerRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    chroma=Depends(get_chroma),
    bm25=Depends(_get_bm25),
    embed_fn: Callable = Depends(_get_embed_fn),
    db: Session = Depends(get_db),
):
    """Run the pipeline once for real, persist the trace artifact, return it."""
    available_docs = _user_documents(db, current_user.id, payload.doc_id)
    artifact = _record_run(
        request, current_user, chroma, bm25, embed_fn,
        query=payload.query, doc_id=payload.doc_id, top_k=payload.top_k,
        model_key=payload.model, available_docs=available_docs,
    )
    totals, verdict = artifact["totals"], artifact["verdict"]
    run = PipelineRun(
        user_id=current_user.id,
        query=payload.query,
        doc_id=payload.doc_id,
        provider=artifact["provider"],
        model_name=artifact["model_name"],
        status=verdict["status"],
        total_ms=totals["latency_ms"],
        cost_inr=totals["cost_inr"],
        step_count=totals["steps"],
        multi_hop=totals["multi_hop"],
        confidence=verdict["confidence"],
        artifact=artifact,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    artifact["run_id"] = run.id
    artifact["created_at"] = iso_utc(run.created_at)
    return artifact


@pipeline_router.get("/runs", response_model=list[RunSummary])
def list_runs(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 30,
):
    """The run gallery — newest first, denormalized (no artifact parse)."""
    rows = (
        db.query(PipelineRun)
        .filter(PipelineRun.user_id == current_user.id)
        .order_by(PipelineRun.created_at.desc())
        .limit(max(1, min(limit, 100)))
        .all()
    )
    return [
        RunSummary(
            run_id=r.id, query=r.query, provider=r.provider, model_name=r.model_name,
            status=r.status, total_ms=r.total_ms, cost_inr=r.cost_inr,
            step_count=r.step_count, multi_hop=r.multi_hop, confidence=r.confidence,
            created_at=iso_utc(r.created_at) or "",
        )
        for r in rows
    ]


@pipeline_router.get("/runs/{run_id}")
def get_run(
    run_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """The full artifact for one run — what the dashboard replays."""
    run = (
        db.query(PipelineRun)
        .filter(PipelineRun.id == run_id, PipelineRun.user_id == current_user.id)
        .first()
    )
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Run not found")
    artifact = dict(run.artifact or {})
    artifact["run_id"] = run.id
    artifact["created_at"] = iso_utc(run.created_at)
    return artifact


@pipeline_router.delete("/runs/{run_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_run(
    run_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Permanently delete a recorded run."""
    run = (
        db.query(PipelineRun)
        .filter(PipelineRun.id == run_id, PipelineRun.user_id == current_user.id)
        .first()
    )
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Run not found")
    db.delete(run)
    db.commit()
