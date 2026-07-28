"""
Pipeline dashboard — record-then-replay trace artifacts.

Hermetic: the agent loop, retrieval, generation and verification are monkeypatched
so the pipeline runs deterministically with NO live model, Chroma, or Ollama. What
we assert is the RECORDER's contract: it serializes exactly what executed —
per-stage timings, metered token/₹ cost, evidence chunks, groundedness verdict and
provider provenance — and persists it as a replayable artifact.
"""
import tempfile
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database import Base, get_db
from app.core.dependencies import get_current_user
from app.generation.agent.state import StepRecord
from app.generation.usage import set_last_usage
from app.pipeline.bm25_index import BM25Index
from app.pipeline.retrieval import ScoredChunk
from app.routers.search import GenerateAnswerResponse

_engine = create_engine("sqlite:///./test_pipeline_dash.db",
                        connect_args={"check_same_thread": False})
_TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


class _FakeUser:
    id = "u_pipe"
    email = "pipe@example.com"


def _override_get_db():
    db = _TestingSession()
    try:
        yield db
    finally:
        db.close()


def _cand(cid, text, doc_id="docp"):
    return ScoredChunk(
        chunk_id=cid, text=text, doc_id=doc_id, source="report.pdf", page_number=3,
        fused_score=0.7, dense_rank=1, bm25_rank=1, reranker_score=0.9,
        metadata={"source": "report.pdf", "page_number": 3},
    )


@pytest.fixture
def client(monkeypatch):
    from app.models import user, document, job, pipeline_run  # noqa: F401 — register tables
    Base.metadata.create_all(bind=_engine)
    app.dependency_overrides[get_current_user] = lambda: _FakeUser()
    app.dependency_overrides[get_db] = _override_get_db
    saved = {k: getattr(app.state, k, None) for k in
             ("bm25", "embed_fn", "llm_fn", "verify_fn", "gen_model_name",
              "verify_model_name", "gen_registry")}
    app.state.bm25 = BM25Index(persist_dir=tempfile.mkdtemp(prefix="bm25_pipe_"))
    app.state.embed_fn = lambda text: [0.1] * 384

    def _metering_llm(prompt):
        set_last_usage({"input_tokens": 20, "output_tokens": 5})
        return "planner/gen output"

    app.state.llm_fn = _metering_llm
    app.state.verify_fn = _metering_llm
    app.state.gen_model_name = "stub-1"       # priced 0.0 → cost is a real 0.0
    app.state.verify_model_name = "stub-verify"
    app.state.gen_registry = {}

    # ── fake pipeline internals (module-level names the router calls) ──
    monkeypatch.setattr("app.routers.pipeline.build_tools", lambda *a, **k: {"retrieve_docs": object()})

    def fake_agent_events(state, tools, llm_fn, *, max_steps=5, allowed_tools=None):
        llm_fn("plan step 1")                     # metered planner call
        state.add_chunks([_cand("c1", "First fact."), _cand("c2", "Second fact.")])
        yield StepRecord(n=1, tool="retrieve_docs", args={"query": "first"},
                         status="ok", summary="retrieve_docs (+2 new)", data={"_new_chunks": 2})
        llm_fn("plan step 2")                      # a second hop → multi_hop
        state.add_chunks([_cand("c3", "Third fact.")])
        yield StepRecord(n=2, tool="retrieve_docs", args={"query": "second"},
                         status="ok", summary="retrieve_docs (+1 new)", data={"_new_chunks": 1})
    monkeypatch.setattr("app.routers.pipeline.agent_events", fake_agent_events)

    def fake_generate(query, chunks, llm_fn=None, **kw):
        llm_fn("generate the answer")              # metered generation call
        return "The grounded answer [1]."
    monkeypatch.setattr("app.routers.pipeline.generate_answer", fake_generate)

    def fake_build_response(query, chunks, citations, draft, verify_fn, qt, scope, task):
        verify_fn("verify")                        # metered verification call
        return GenerateAnswerResponse(
            query=query, answer=draft, citations=citations, cited_sources=citations,
            all_sources=citations, abstained=False, grounded=True, confidence=0.86,
            unsupported_claims=[], verified=True,
            confidence_signals={"groundedness": 0.9, "verified": True},
            query_type=qt, retrieval_scope=scope, answer_task=task,
        )
    monkeypatch.setattr("app.routers.pipeline._build_answer_response", fake_build_response)

    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_db, None)
        for k, v in saved.items():
            setattr(app.state, k, v)
        Base.metadata.drop_all(bind=_engine)


# ── examples ──────────────────────────────────────────────────────────────────

def test_examples_endpoint(client):
    ex = client.get("/pipeline/examples").json()["examples"]
    ids = {e["id"] for e in ex}
    assert ids == {"multi_hop", "whole_doc", "abstention"}
    assert all("query" in e and "why" in e and "expect" in e for e in ex)


# ── record: the artifact contract ───────────────────────────────────────────────

def test_created_at_is_timezone_aware_utc(client):
    """created_at must carry an explicit UTC offset.

    It is stored naive-UTC, and a bare "2026-07-28T08:26:03" is parsed by JavaScript
    as LOCAL time — so a browser at UTC+5:30 rendered a 6-minute-old run as "5h ago".
    """
    from datetime import timezone

    art = client.post("/pipeline/runs", json={"query": "compare A and B"}).json()
    parsed = datetime.fromisoformat(art["created_at"])
    assert parsed.tzinfo is not None, "created_at must not be timezone-naive"
    assert parsed.utcoffset() == timedelta(0)

    # …and the same guarantee on the gallery list route the sidebar actually reads.
    row = client.get("/pipeline/runs").json()[0]
    listed = datetime.fromisoformat(row["created_at"])
    assert listed.tzinfo is not None
    # Sanity: the run just happened, so its age is seconds — not hours.
    age = abs((datetime.now(timezone.utc) - listed).total_seconds())
    assert age < 300, f"created_at is skewed by {age}s"


def test_run_records_faithful_artifact(client):
    art = client.post("/pipeline/runs", json={"query": "compare A and B"}).json()

    # provenance + identity
    assert art["provider"] == "default"
    assert art["provenance"] == "recorded · default"
    assert art["run_id"] and art["created_at"]

    # the timeline has the real stages in order
    kinds = [s["kind"] for s in art["steps"]]
    assert kinds == ["router", "agent", "agent", "generate", "verify"]
    assert all("duration_ms" in s for s in art["steps"])

    # totals reflect what executed
    t = art["totals"]
    assert t["chunks"] == 3 and t["steps"] == 2 and t["multi_hop"] is True
    # 4 metered calls (2 planner + 1 generate + 1 verify), each 20→5 tokens
    assert t["input_tokens"] == 80 and t["output_tokens"] == 20
    assert t["cost_inr"] == 0.0            # stub priced at 0 → a real, correct 0

    # verdict + answer + evidence
    assert art["verdict"]["grounded"] is True and art["verdict"]["confidence"] == 0.86
    assert art["answer"] == "The grounded answer [1]."
    assert len(art["chunks"]) == 3
    assert art["chunks"][0]["reranker_score"] == 0.9 and art["chunks"][0]["source"] == "report.pdf"


def test_generate_step_carries_cost(client):
    art = client.post("/pipeline/runs", json={"query": "q"}).json()
    gen = next(s for s in art["steps"] if s["kind"] == "generate")
    assert gen["detail"]["input_tokens"] == 20 and gen["detail"]["output_tokens"] == 5
    assert gen["detail"]["cost_inr"] == 0.0


# ── gallery: list / get / delete ────────────────────────────────────────────────

def test_gallery_list_and_get_and_delete(client):
    run_id = client.post("/pipeline/runs", json={"query": "hello"}).json()["run_id"]

    rows = client.get("/pipeline/runs").json()
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == run_id and row["query"] == "hello"
    assert row["multi_hop"] is True and row["step_count"] == 2 and row["status"] == "ok"

    full = client.get(f"/pipeline/runs/{run_id}").json()
    assert full["run_id"] == run_id and full["answer"] == "The grounded answer [1]."

    assert client.delete(f"/pipeline/runs/{run_id}").status_code == 204
    assert client.get("/pipeline/runs").json() == []
    assert client.get(f"/pipeline/runs/{run_id}").status_code == 404


def test_get_missing_run_404(client):
    assert client.get("/pipeline/runs/nope").status_code == 404


# ── model routing / provenance ──────────────────────────────────────────────────

def test_run_uses_selected_model_provenance(client):
    def _reg_llm(prompt):
        set_last_usage({"input_tokens": 10, "output_tokens": 2})
        return "gemini output"
    app.state.gen_registry = {
        "gemini": {"llm_fn": _reg_llm, "llm_stream_fn": None,
                   "model_name": "gemini-3.1-flash-lite", "provider_name": "gemini"},
    }
    art = client.post("/pipeline/runs", json={"query": "q", "model": "gemini"}).json()
    assert art["provider"] == "gemini"
    assert art["model_name"] == "gemini-3.1-flash-lite"
    assert art["provenance"] == "recorded · gemini"


def test_unknown_model_falls_back_to_default(client):
    art = client.post("/pipeline/runs", json={"query": "q", "model": "nope"}).json()
    assert art["provider"] == "default"


def test_run_requires_a_generation_backend(client):
    app.state.llm_fn = None
    app.state.gen_registry = {}
    resp = client.post("/pipeline/runs", json={"query": "q"})
    assert resp.status_code == 400
    assert "generation_not_configured" in resp.json()["detail"]


def test_planner_failure_is_recorded_not_500(client, monkeypatch):
    """A transient LLM failure during agent PLANNING (e.g. Gemini 504) must be
    recorded as an error stage, not bubble up as a 500 — the dashboard fires several
    sequential model calls, so any one timing out cannot crash the whole run."""
    from app.generation.base import GenerationError

    def boom(state, tools, llm_fn, *, max_steps=5, allowed_tools=None):
        raise GenerationError("Gemini 504 DEADLINE_EXCEEDED")
        yield  # noqa — makes this a generator function
    monkeypatch.setattr("app.routers.pipeline.agent_events", boom)

    resp = client.post("/pipeline/runs", json={"query": "q"})
    assert resp.status_code == 200, resp.text
    art = resp.json()
    assert any(s["status"] == "error" for s in art["steps"])   # the failure is visible
    assert art["verdict"]["status"] == "error"                  # no evidence → honest error
    assert art["run_id"]                                        # still persisted for the gallery
