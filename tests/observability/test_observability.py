"""
Phase 9A — observability SEAM tests.

Covers the acceptance criteria:
  * NullTracer is the default when TRACING_ENABLED is unset.
  * A deliberately broken tracer changes no endpoint's response / status code.
  * The exact query span tree is produced for one query (recording tracer).
  * Two concurrent async requests do not interleave their span stacks.
  * trace_id survives the enqueue -> worker boundary.
"""
from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

import app.observability as obs
from app.main import app
from app.core.dependencies import get_current_user, get_chroma
from app.pipeline.bm25_index import BM25Index  # noqa: F401 (import parity w/ prod deps)
from tests.observability.recording import BrokenTracer, RecordingSpan, RecordingTracer


# ── Tracer reset + clean env around every test ────────────────────────────────

@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    monkeypatch.delenv("TRACING_ENABLED", raising=False)
    monkeypatch.delenv("TRACE_CAPTURE", raising=False)
    obs.reset_tracer()
    yield
    obs.reset_tracer()


# ── Mock retrieval backends (real hybrid_search + real rerank run over these) ──

class _FakeUser:
    id = "u_obs"
    email = "obs@example.com"


def _meta(i: int) -> dict:
    return {
        "doc_id": "docA",
        "source": "m.pdf",
        "page_number": 1,
        "chunk_index": i,
        "char_start": i * 100,
        "char_end": i * 100 + 90,
    }


class _MockCollection:
    def __init__(self, ids, texts):
        self._ids = ids
        self._texts = texts
        self._metas = [_meta(i) for i in range(len(ids))]

    def query(self, query_embeddings, n_results, where=None, include=None):
        return {
            "ids": [self._ids],
            "documents": [self._texts],
            "metadatas": [self._metas],
            "distances": [[0.1 * i for i in range(len(self._ids))]],
        }

    def get(self, ids=None, where=None, include=None):
        if ids is None:
            sel = list(range(len(self._ids)))
        else:
            sel = [self._ids.index(i) for i in ids if i in self._ids]
        return {
            "ids": [self._ids[i] for i in sel],
            "documents": [self._texts[i] for i in sel],
            "metadatas": [self._metas[i] for i in sel],
        }


class _MockChroma:
    def __init__(self, coll):
        self._coll = coll

    def get_collection(self, name):
        return self._coll


class _MockBM25:
    def __init__(self, ids):
        self._ids = ids

    def query(self, query, user_id=None, doc_id=None, top_n=50):
        return [(cid, 1.0 / (r + 1)) for r, cid in enumerate(self._ids)]


class _FakeCrossEncoder:
    def predict(self, pairs):
        # Descending scores, deterministic.
        return [1.0 - 0.1 * i for i in range(len(pairs))]


def _install_backends(monkeypatch, llm=lambda p: "This is a stub answer."):
    ids = ["c0", "c1", "c2", "c3"]
    texts = [
        "Apple Inc. reported annual revenue of $394 billion in fiscal 2023.",
        "The company's services segment continued to grow.",
        "iPhone remained the largest product line by revenue.",
        "Operating margin was strong across regions.",
    ]
    coll = _MockCollection(ids, texts)
    app.dependency_overrides[get_current_user] = lambda: _FakeUser()
    app.dependency_overrides[get_chroma] = lambda: _MockChroma(coll)
    app.state.bm25 = _MockBM25(ids)
    app.state.embed_fn = lambda text: [0.1] * 384
    app.state.llm_fn = llm
    app.state.verify_fn = None
    app.state.answer_cache = None
    app.state.gen_model_name = "test-model-1"
    # Real rerank runs, but over a fake cross-encoder (no model download / network).
    monkeypatch.setattr("app.pipeline.retrieval._get_cross_encoder", lambda: _FakeCrossEncoder())


def _clear_backends():
    app.dependency_overrides.pop(get_current_user, None)
    app.dependency_overrides.pop(get_chroma, None)


# ── 1. NullTracer is the default when TRACING_ENABLED is unset ─────────────────

def test_null_tracer_is_default_when_unset():
    assert isinstance(obs.get_tracer(), obs.NullTracer)


def test_tracing_enabled_true_still_no_op_in_phase_9a(monkeypatch):
    # Phase 9A ships only the no-op tracer even when explicitly enabled.
    monkeypatch.setenv("TRACING_ENABLED", "true")
    obs.reset_tracer()
    assert isinstance(obs.get_tracer(), obs.NullTracer)


def test_set_and_reset_tracer():
    rec = RecordingTracer()
    obs.set_tracer(rec)
    assert obs.get_tracer() is rec
    obs.reset_tracer()
    assert isinstance(obs.get_tracer(), obs.NullTracer)


def test_hash_user_id_is_stable_16char_and_not_raw(monkeypatch):
    monkeypatch.setenv("TRACE_SALT", "pepper")
    h1 = obs.hash_user_id("user-123")
    h2 = obs.hash_user_id("user-123")
    assert h1 == h2
    assert len(h1) == 16
    assert "user-123" not in h1
    assert obs.hash_user_id("other") != h1
    assert obs.hash_user_id("") == ""


def test_span_always_yields_usable_span_even_when_broken():
    obs.set_tracer(BrokenTracer())
    with obs.span("x") as sp:
        assert sp is not None
        # These must not raise despite the broken backend.
        sp.update(a=1)
        sp.score("s", 0.5)


# ── 2. A broken tracer changes no endpoint response / status ───────────────────

def test_broken_tracer_does_not_change_endpoint(monkeypatch):
    _install_backends(monkeypatch)
    try:
        client = TestClient(app)
        payload = {"query": "What is Apple's revenue?"}

        obs.reset_tracer()  # NullTracer baseline
        base = client.post("/generate/answer?stream=false", json=payload)

        obs.set_tracer(BrokenTracer())
        broken = client.post("/generate/answer?stream=false", json=payload)

        assert base.status_code == broken.status_code == 200
        assert base.json() == broken.json()
    finally:
        _clear_backends()


def test_broken_tracer_streaming_still_streams(monkeypatch):
    _install_backends(monkeypatch)
    try:
        obs.set_tracer(BrokenTracer())
        client = TestClient(app)
        with client.stream("POST", "/generate/answer?stream=true", json={"query": "revenue?"}) as r:
            assert r.status_code == 200
            body = "".join(r.iter_text())
        assert "verification" in body  # the two-phase stream completed
    finally:
        _clear_backends()


# ── 3. Exact query span tree for one query ─────────────────────────────────────

def test_exact_query_span_tree(monkeypatch):
    rec = RecordingTracer()
    obs.set_tracer(rec)
    _install_backends(monkeypatch)
    try:
        client = TestClient(app)
        resp = client.post("/generate/answer?stream=false", json={"query": "What is Apple's revenue?"})
        assert resp.status_code == 200, resp.text
    finally:
        _clear_backends()

    # Exactly one root: rag.request
    roots = rec.roots()
    assert [r.name for r in roots] == ["rag.request"]
    root = roots[0]

    assert root.child_names() == ["retrieve", "rerank", "prompt_build", "generate", "verify"]

    retrieve = root.children[0]
    assert retrieve.child_names() == ["embed_query", "vector_search", "bm25_search", "rrf_fuse"]

    # rerank / generate carry model labels; verify carries the three scores.
    rerank = rec.only("rerank")
    assert rerank.fields.get("model")
    assert rerank.fields.get("candidates_in") == 4

    generate = rec.only("generate")
    assert generate.fields.get("model") == "test-model-1"

    verify = rec.only("verify")
    assert set(verify.scores) == {"groundedness", "citation_validity", "confidence"}

    # abstained boolean recorded on the ROOT span.
    assert "abstained" in root.fields

    # Every span shares one trace id.
    trace_ids = {s.trace_id for s in rec.spans}
    assert len(trace_ids) == 1 and next(iter(trace_ids))


def test_streaming_query_span_tree(monkeypatch):
    """The streaming path emits the same tree; generate/verify run in the SSE
    generator (a copied context) and re-adopt the root without leaking tokens."""
    rec = RecordingTracer()
    obs.set_tracer(rec)
    _install_backends(monkeypatch)
    try:
        client = TestClient(app)
        with client.stream("POST", "/generate/answer?stream=true", json={"query": "What is Apple's revenue?"}) as r:
            assert r.status_code == 200
            body = "".join(r.iter_text())
        assert "verification" in body
    finally:
        _clear_backends()

    root = rec.only("rag.request")
    assert root.child_names() == ["retrieve", "rerank", "prompt_build", "generate", "verify"]
    assert root.children[0].child_names() == ["embed_query", "vector_search", "bm25_search", "rrf_fuse"]
    # ttft is measured on the streaming generate span.
    assert "ttft_ms" in rec.only("generate").fields
    assert len({s.trace_id for s in rec.spans}) == 1


# ── 4. Concurrent async requests do not interleave span stacks ─────────────────

@pytest.mark.asyncio
async def test_concurrent_requests_do_not_interleave():
    rec = RecordingTracer()
    obs.set_tracer(rec)

    async def one(tag: str):
        with obs.span("root", tag=tag):
            await asyncio.sleep(0)
            with obs.span("mid", tag=tag):
                await asyncio.sleep(0)
                with obs.span("leaf", tag=tag):
                    await asyncio.sleep(0)

    await asyncio.gather(one("A"), one("B"), one("A2"), one("B2"))

    # Each non-root span's parent belongs to the SAME logical request (same tag),
    # proving no cross-task interleaving of the contextvar stack.
    for sp in rec.spans:
        if sp.parent is not None:
            assert sp.fields["tag"] == sp.parent.fields["tag"]

    # Each request minted its own trace id; roots are all distinct.
    root_traces = [r.trace_id for r in rec.roots()]
    assert len(root_traces) == len(set(root_traces)) == 4

    # A leaf's whole ancestry stays within its own tag.
    for leaf in rec.by_name("leaf"):
        node = leaf
        while node.parent is not None:
            assert node.parent.fields["tag"] == leaf.fields["tag"]
            node = node.parent


# ── 5. trace_id survives the enqueue -> worker boundary ────────────────────────

@pytest.mark.asyncio
async def test_trace_id_survives_enqueue_to_worker(monkeypatch):
    import app.workers.tasks as tasks

    rec = RecordingTracer()
    obs.set_tracer(rec)

    # Isolate the boundary: stub the heavy orchestrator + provider build.
    monkeypatch.setattr(tasks, "build_provider", lambda ctx: None)
    monkeypatch.setattr(tasks, "_run_pipeline_sync", lambda *a, **k: "ready")

    # The enqueuer passes trace_id in the (pickled, primitive-only) job kwargs; the
    # worker seeds the ingest root under it.
    trace_id = obs.new_trace_id()
    result = await tasks.run_pipeline({}, "doc-boundary", trace_id=trace_id)

    assert result == {"doc_id": "doc-boundary", "status": "ready"}
    ingest = rec.only("ingest.document")
    assert ingest.trace_id == trace_id
    assert ingest.fields.get("doc_id") == "doc-boundary"
    assert ingest.fields.get("status") == "ready"
    assert rec.flush_count == 1  # end-of-job flush ran
