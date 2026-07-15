"""
Generation-quality fix — endpoint integration (Parts 4 + 6).

  * thin-result retry: a 1-chunk fixture triggers a broadened re-retrieval that
    appends chunks; a 5-chunk fixture does NOT; THIN_RESULT_RETRY=false disables it.
  * response shape: query_type, verbatim_ratio, quality_warning are present and
    correct on both the sync JSON body and the streaming verification event.
  * all additive — existing fields are untouched (covered by the Story 2-7 suites).

Hermetic: StubProvider + patched retrieval only. No live API.
"""
import json
import tempfile
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.core.dependencies import get_current_user
from app.generation.stub import StubProvider
from app.main import app
from app.pipeline.bm25_index import BM25Index
from app.pipeline.retrieval import ScoredChunk


class _FakeUser:
    id = "u_gq"
    email = "gq@example.com"


def _cand(cid, text, doc_id="docA", source="m.pdf", page=1):
    return ScoredChunk(
        chunk_id=cid, text=text, doc_id=doc_id, source=source, page_number=page,
        fused_score=0.5, dense_rank=1, bm25_rank=1, reranker_score=1.0,
        metadata={"source": source, "page_number": page},
    )


@pytest.fixture
def endpoint_client():
    app.dependency_overrides[get_current_user] = lambda: _FakeUser()
    prev = {k: getattr(app.state, k, None) for k in ("bm25", "embed_fn", "llm_fn", "llm_stream_fn")}
    app.state.bm25 = BM25Index(persist_dir=tempfile.mkdtemp(prefix="bm25_gq_"))
    app.state.embed_fn = lambda text: [0.1] * 384
    app.state.llm_fn = None
    app.state.llm_stream_fn = None
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        for k, v in prev.items():
            setattr(app.state, k, v)


def _wire_stub(response_text: str) -> StubProvider:
    stub = StubProvider()
    stub.set_response(response_text)
    app.state.llm_fn = lambda prompt: stub.generate(prompt)
    return stub


# ── Part 4a — thin-result retry ───────────────────────────────────────────────

def test_thin_result_triggers_broadened_retry(endpoint_client, monkeypatch):
    """A DEFINITIONAL query with 1 reranked chunk re-runs retrieval broadened and
    APPENDS the deduped hits (2 → 3 total)."""
    calls = {"hybrid": 0}

    def fake_hybrid(**kw):
        calls["hybrid"] += 1
        if calls["hybrid"] == 1:
            return [_cand("c1", "The GX-4200 samples the thermistor.")]  # thin (1)
        # broadened retry returns fresh chunks (plus a duplicate to prove dedup)
        return [
            _cand("c1", "The GX-4200 samples the thermistor."),  # dup — must dedup
            _cand("c2", "It regulates temperature via feedback."),
            _cand("c3", "The controller is rated for indoor use."),
        ]

    monkeypatch.setattr("app.routers.search.hybrid_search", fake_hybrid)
    monkeypatch.setattr("app.routers.search.rerank", lambda q, c, top_k: c[:top_k])
    _wire_stub("The GX-4200 regulates temperature [1][2][3].")

    resp = endpoint_client.post("/generate/answer?stream=false", json={"query": "What is the GX-4200?"})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert calls["hybrid"] == 2, "expected exactly one broadened retry"
    all_ids = [c["chunk_id"] for c in body["all_sources"]]
    assert all_ids == ["c1", "c2", "c3"]  # original + deduped retry hits


def test_five_chunks_does_not_trigger_retry(endpoint_client, monkeypatch):
    """A healthy 5-chunk result (>= THIN_RESULT_THRESHOLD) never retries."""
    calls = {"hybrid": 0}

    def fake_hybrid(**kw):
        calls["hybrid"] += 1
        return [_cand(f"c{i}", f"Content {i}.") for i in range(5)]

    monkeypatch.setattr("app.routers.search.hybrid_search", fake_hybrid)
    monkeypatch.setattr("app.routers.search.rerank", lambda q, c, top_k: c[:top_k])
    _wire_stub("An answer [1].")

    resp = endpoint_client.post("/generate/answer?stream=false", json={"query": "What is the GX-4200?"})
    assert resp.status_code == 200, resp.text
    assert calls["hybrid"] == 1  # no retry


def test_thin_result_retry_disabled_by_config(endpoint_client, monkeypatch):
    """THIN_RESULT_RETRY=false disables the fallback entirely — 1 chunk, no retry."""
    calls = {"hybrid": 0}

    def fake_hybrid(**kw):
        calls["hybrid"] += 1
        return [_cand("c1", "The GX-4200 samples the thermistor.")]

    monkeypatch.setattr("app.routers.search.hybrid_search", fake_hybrid)
    monkeypatch.setattr("app.routers.search.rerank", lambda q, c, top_k: c[:top_k])

    fake_settings = SimpleNamespace(
        stream_enabled=True,
        validate_use_llm=False,
        thin_result_retry=False,
        thin_result_threshold=3,
    )
    monkeypatch.setattr("app.routers.search.get_settings", lambda: fake_settings)
    _wire_stub("An answer [1].")

    resp = endpoint_client.post("/generate/answer?stream=false", json={"query": "What is the GX-4200?"})
    assert resp.status_code == 200, resp.text
    assert calls["hybrid"] == 1  # retry disabled → single retrieval


# ── Part 6 — response shape additions ─────────────────────────────────────────

def test_response_includes_new_quality_fields(endpoint_client, monkeypatch):
    """Sync JSON body carries query_type, verbatim_ratio, quality_warning."""
    monkeypatch.setattr("app.routers.search.hybrid_search",
                        lambda **kw: [_cand("c1", "Apple revenue was $394 billion.")])
    monkeypatch.setattr("app.routers.search.rerank", lambda q, c, top_k: c[:top_k])
    _wire_stub("Apple's revenue reached roughly $394 billion that year [1].")

    resp = endpoint_client.post("/generate/answer?stream=false", json={"query": "What is Apple's revenue?"})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["query_type"] == "definitional"   # "What is ..." → DEFINITIONAL
    assert "verbatim_ratio" in body and isinstance(body["verbatim_ratio"], float)
    assert "quality_warning" in body  # null here (rephrased)


def test_verbatim_answer_sets_high_verbatim_warning(endpoint_client, monkeypatch):
    """An answer copied verbatim from the chunk → verbatim_ratio high + warning."""
    source = (
        "The GX-4200 industrial controller regulates internal temperature using a "
        "closed-loop feedback system that samples the thermistor every 200 milliseconds."
    )
    monkeypatch.setattr("app.routers.search.hybrid_search",
                        lambda **kw: [_cand("c1", source)])
    monkeypatch.setattr("app.routers.search.rerank", lambda q, c, top_k: c[:top_k])
    _wire_stub(source + " [1]")

    resp = endpoint_client.post("/generate/answer?stream=false", json={"query": "How does the GX-4200 work?"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["verbatim_ratio"] == 1.0
    assert body["quality_warning"] == "high_verbatim"


def test_streaming_verification_includes_new_fields(endpoint_client, monkeypatch):
    """The SSE verification event carries the new fields too (same _build path)."""
    monkeypatch.setattr("app.routers.search.hybrid_search",
                        lambda **kw: [_cand("c1", "Apple revenue was $394 billion.")])
    monkeypatch.setattr("app.routers.search.rerank", lambda q, c, top_k: c[:top_k])
    stub = StubProvider()
    stub.set_response("Apple earned around $394 billion in revenue [1].")
    app.state.llm_fn = lambda prompt: stub.generate(prompt)
    app.state.llm_stream_fn = lambda prompt: stub.generate_stream(prompt)

    resp = endpoint_client.post("/generate/answer?stream=true", json={"query": "What is Apple's revenue?"})
    assert resp.status_code == 200, resp.text

    vdata = None
    for block in resp.text.split("\n\n"):
        if "event: verification" in block:
            for line in block.split("\n"):
                if line.startswith("data:"):
                    vdata = json.loads(line[len("data:"):].strip())
    assert vdata is not None
    assert vdata["query_type"] == "definitional"
    assert "verbatim_ratio" in vdata
    assert "quality_warning" in vdata


def test_error_shape_carries_query_type(endpoint_client, monkeypatch):
    """generation_not_configured (llm_fn None) still reports the classified type."""
    monkeypatch.setattr("app.routers.search.hybrid_search",
                        lambda **kw: [_cand("c1", "x"), _cand("c2", "y"), _cand("c3", "z")])
    monkeypatch.setattr("app.routers.search.rerank", lambda q, c, top_k: c[:top_k])
    # llm_fn stays None (fixture default)

    resp = endpoint_client.post("/generate/answer?stream=false", json={"query": "Compare A and B"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["error"] == "generation_not_configured"
    assert body["query_type"] == "analytical"
