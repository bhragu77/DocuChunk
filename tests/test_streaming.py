"""
Story 6 — Streaming & Async tests.

Covers the two-phase SSE contract for POST /generate/answer:
  * StubProvider.generate_stream yields the answer one word at a time.
  * ?stream=true streams 'token' events (done=false … done=true) followed by a
    single 'verification' event whose payload is schema-identical to the
    ?stream=false JSON body.
  * abstention over the stream → verification.abstained=true, confidence=0.0.
  * generate_stream failing on the first chunk falls back to generate() (one big
    token event) — graceful degradation, not a crash.
  * STREAM_ENABLED=false forces JSON even with ?stream=true (global kill switch).
  * /search/semantic is pure retrieval — never streams, never emits SSE.

All hermetic: StubProvider only, retrieval internals patched. No live API.
"""
import json
import tempfile

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from app.main import app
from app.core.dependencies import get_current_user
from app.generation.prompt_builder import ABSTENTION_PHRASE
from app.generation.stub import StubProvider
from app.pipeline.bm25_index import BM25Index
from app.pipeline.retrieval import ScoredChunk


# ── 1. StubProvider.generate_stream (unit) ────────────────────────────────────

def test_stub_generate_stream_yields_one_word_at_a_time():
    """The stub streams word-by-word and concatenation reconstructs generate()."""
    stub = StubProvider()
    stub.set_response("Apple Inc. revenue was $394 billion.")

    chunks = list(stub.generate_stream("prompt"))

    assert len(chunks) == 6  # six words → six chunks
    assert "".join(chunks) == "Apple Inc. revenue was $394 billion."
    assert "".join(chunks) == stub.generate("prompt")


def test_stub_generate_stream_empty_response_yields_nothing():
    """An empty (refusal) response streams nothing — a valid refusal."""
    stub = StubProvider()
    stub.set_response("")
    assert list(stub.generate_stream("prompt")) == []


def test_stub_generate_stream_raises_on_error():
    """Configured error propagates (used to exercise the endpoint fallback)."""
    stub = StubProvider()
    stub.set_error(RuntimeError("stream boom"))
    with pytest.raises(RuntimeError):
        list(stub.generate_stream("prompt"))


# ── Endpoint fixtures / helpers ───────────────────────────────────────────────

class _FakeUser:
    id = "u_story6"
    email = "story6@example.com"


def _cand(cid, text, doc_id="docA", source="m.pdf", page=1):
    return ScoredChunk(
        chunk_id=cid, text=text, doc_id=doc_id, source=source, page_number=page,
        fused_score=0.5, dense_rank=1, bm25_rank=1, reranker_score=1.0,
        metadata={"source": source, "page_number": page},
    )


@pytest.fixture
def endpoint_client():
    app.dependency_overrides[get_current_user] = lambda: _FakeUser()
    prev_bm25 = getattr(app.state, "bm25", None)
    prev_embed = getattr(app.state, "embed_fn", None)
    prev_llm = getattr(app.state, "llm_fn", None)
    prev_stream = getattr(app.state, "llm_stream_fn", None)
    app.state.bm25 = BM25Index(persist_dir=tempfile.mkdtemp(prefix="bm25_s6_"))
    app.state.embed_fn = lambda text: [0.1] * 384
    app.state.llm_fn = None
    app.state.llm_stream_fn = None
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.state.bm25 = prev_bm25
        app.state.embed_fn = prev_embed
        app.state.llm_fn = prev_llm
        app.state.llm_stream_fn = prev_stream


def _parse_sse(text: str) -> list[dict]:
    """Parse an SSE body into [{'event': name, 'data': parsed_json}, ...]."""
    events = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        ev = {"event": None, "data": None}
        for line in block.split("\n"):
            if line.startswith("event:"):
                ev["event"] = line[len("event:"):].strip()
            elif line.startswith("data:"):
                ev["data"] = json.loads(line[len("data:"):].strip())
        events.append(ev)
    return events


def _wire_stub(stub: StubProvider):
    app.state.llm_fn = lambda prompt: stub.generate(prompt)
    app.state.llm_stream_fn = lambda prompt: stub.generate_stream(prompt)


# ── 2. SSE integration: token stream + verification ───────────────────────────

def test_sse_streams_tokens_then_verification(endpoint_client, monkeypatch):
    """?stream=true → text/event-stream: multiple token(done=false), one
    token(done=true) carrying full_answer, one verification with confidence and
    cited_sources."""
    cands = [_cand("c1", "Apple Inc. revenue was $394 billion.")]
    monkeypatch.setattr("app.routers.search.hybrid_search", lambda **kw: cands)
    monkeypatch.setattr("app.routers.search.rerank", lambda q, c, top_k: c[:top_k])

    stub = StubProvider()
    stub.set_response("Apple Inc. revenue was $394 billion [1].")
    _wire_stub(stub)

    resp = endpoint_client.post("/generate/answer?stream=true", json={"query": "Apple revenue?"})
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(resp.text)
    token_events = [e for e in events if e["event"] == "token"]
    verification = [e for e in events if e["event"] == "verification"]

    # Multiple streaming token events, all done=false.
    streaming = [e for e in token_events if e["data"]["done"] is False]
    assert len(streaming) > 1
    assert all(e["data"]["done"] is False for e in streaming)

    # Exactly one terminal token event carrying the assembled answer.
    done = [e for e in token_events if e["data"]["done"] is True]
    assert len(done) == 1
    assert done[0]["data"]["text"] == ""
    assert done[0]["data"]["full_answer"] == "Apple Inc. revenue was $394 billion [1]."
    # The concatenated stream reconstructs the same answer the client can trust.
    assembled = "".join(e["data"]["text"] for e in streaming).strip()
    assert assembled == done[0]["data"]["full_answer"]

    # Exactly one verification event, carrying confidence + cited_sources.
    assert len(verification) == 1
    vdata = verification[0]["data"]
    assert "confidence" in vdata
    assert [c["chunk_id"] for c in vdata["cited_sources"]] == ["c1"]
    assert vdata["abstained"] is False


def test_verification_event_matches_nonstreaming_json(endpoint_client, monkeypatch):
    """The verification event is schema-compatible with the ?stream=false JSON body:
    same fields, same types, same values for the same inputs."""
    cands = [_cand("c1", "Apple Inc. revenue was $394 billion.")]
    monkeypatch.setattr("app.routers.search.hybrid_search", lambda **kw: cands)
    monkeypatch.setattr("app.routers.search.rerank", lambda q, c, top_k: c[:top_k])

    stub = StubProvider()
    stub.set_response("Apple Inc. revenue was $394 billion [1].")
    _wire_stub(stub)

    # Non-streaming JSON body.
    json_resp = endpoint_client.post("/generate/answer?stream=false", json={"query": "Apple revenue?"})
    assert json_resp.status_code == 200, json_resp.text
    json_body = json_resp.json()

    # Streaming verification event.
    sse_resp = endpoint_client.post("/generate/answer?stream=true", json={"query": "Apple revenue?"})
    events = _parse_sse(sse_resp.text)
    vdata = next(e["data"] for e in events if e["event"] == "verification")

    # Identical field sets and identical values (same stub, same retrieval).
    assert set(vdata.keys()) == set(json_body.keys())
    assert vdata == json_body


def test_sse_abstention_confidence_zero(endpoint_client, monkeypatch):
    """Stub returns the abstention phrase → verification abstained=true, confidence=0.0."""
    cands = [_cand("c1", "Some content."), _cand("c2", "Other content.")]
    monkeypatch.setattr("app.routers.search.hybrid_search", lambda **kw: cands)
    monkeypatch.setattr("app.routers.search.rerank", lambda q, c, top_k: c[:top_k])

    stub = StubProvider()
    stub.set_response(ABSTENTION_PHRASE)
    _wire_stub(stub)

    resp = endpoint_client.post("/generate/answer?stream=true", json={"query": "unanswerable?"})
    events = _parse_sse(resp.text)
    vdata = next(e["data"] for e in events if e["event"] == "verification")

    assert vdata["abstained"] is True
    assert vdata["confidence"] == 0.0
    assert vdata["cited_sources"] == []


# ── 3. Fallback: generate_stream fails → generate() one big chunk ─────────────

def test_sse_falls_back_to_generate_when_stream_raises(endpoint_client, monkeypatch):
    """If generate_stream raises on the first chunk, the endpoint falls back to
    generate() and yields ONE big token event + verification — no crash."""
    cands = [_cand("c1", "Apple Inc. revenue was $394 billion.")]
    monkeypatch.setattr("app.routers.search.hybrid_search", lambda **kw: cands)
    monkeypatch.setattr("app.routers.search.rerank", lambda q, c, top_k: c[:top_k])

    def boom_stream(prompt):
        raise RuntimeError("no streaming here")
        yield  # pragma: no cover — makes this a generator function

    stub = StubProvider()
    stub.set_response("Apple Inc. revenue was $394 billion [1].")
    app.state.llm_fn = lambda prompt: stub.generate(prompt)
    app.state.llm_stream_fn = boom_stream

    resp = endpoint_client.post("/generate/answer?stream=true", json={"query": "Apple revenue?"})
    assert resp.status_code == 200, resp.text
    events = _parse_sse(resp.text)

    token_events = [e for e in events if e["event"] == "token"]
    streaming = [e for e in token_events if e["data"]["done"] is False]
    done = [e for e in token_events if e["data"]["done"] is True]

    # Fallback yields the WHOLE answer as a single chunk.
    assert len(streaming) == 1
    assert streaming[0]["data"]["text"] == "Apple Inc. revenue was $394 billion [1]."
    assert len(done) == 1
    assert done[0]["data"]["full_answer"] == "Apple Inc. revenue was $394 billion [1]."

    # Verification still runs on the fallback answer.
    vdata = next(e["data"] for e in events if e["event"] == "verification")
    assert [c["chunk_id"] for c in vdata["cited_sources"]] == ["c1"]


def test_sse_falls_back_when_no_stream_fn_wired(endpoint_client, monkeypatch):
    """No llm_stream_fn at all → fall back to llm_fn (one big token) + verification."""
    cands = [_cand("c1", "Apple Inc. revenue was $394 billion.")]
    monkeypatch.setattr("app.routers.search.hybrid_search", lambda **kw: cands)
    monkeypatch.setattr("app.routers.search.rerank", lambda q, c, top_k: c[:top_k])

    stub = StubProvider()
    stub.set_response("Apple Inc. revenue was $394 billion [1].")
    app.state.llm_fn = lambda prompt: stub.generate(prompt)
    app.state.llm_stream_fn = None  # no streaming seam

    resp = endpoint_client.post("/generate/answer?stream=true", json={"query": "Apple revenue?"})
    events = _parse_sse(resp.text)
    streaming = [e for e in events if e["event"] == "token" and e["data"]["done"] is False]
    assert len(streaming) == 1
    assert streaming[0]["data"]["text"] == "Apple Inc. revenue was $394 billion [1]."


# ── 4. STREAM_ENABLED=false global kill switch ────────────────────────────────

def test_stream_enabled_false_forces_json(endpoint_client, monkeypatch):
    """With STREAM_ENABLED=false, even ?stream=true returns the synchronous JSON
    body (not an SSE stream)."""
    cands = [_cand("c1", "Apple Inc. revenue was $394 billion.")]
    monkeypatch.setattr("app.routers.search.hybrid_search", lambda **kw: cands)
    monkeypatch.setattr("app.routers.search.rerank", lambda q, c, top_k: c[:top_k])

    stub = StubProvider()
    stub.set_response("Apple Inc. revenue was $394 billion [1].")
    _wire_stub(stub)

    from app.config import get_settings
    real = get_settings()

    class _KillSwitchSettings:
        stream_enabled = False
        validate_use_llm = real.validate_use_llm

    monkeypatch.setattr("app.routers.search.get_settings", lambda: _KillSwitchSettings())

    resp = endpoint_client.post("/generate/answer?stream=true", json={"query": "Apple revenue?"})
    assert resp.status_code == 200, resp.text
    # JSON, not an event-stream.
    assert resp.headers["content-type"].startswith("application/json")
    body = resp.json()
    assert body["answer"] == "Apple Inc. revenue was $394 billion [1]."
    assert body["cited_sources"][0]["chunk_id"] == "c1"


# ── 5. /search/semantic is unaffected by streaming ────────────────────────────

def test_semantic_search_never_streams(endpoint_client, monkeypatch):
    """/search/semantic is pure retrieval: JSON response, no SSE, no LLM/stream."""
    cands = [_cand("c1", "Apple Inc. revenue was $394 billion.")]
    monkeypatch.setattr("app.routers.search.hybrid_search", lambda **kw: cands)
    monkeypatch.setattr("app.routers.search.rerank", lambda q, c, top_k: c[:top_k])

    llm = MagicMock()
    stream = MagicMock()
    app.state.llm_fn = llm
    app.state.llm_stream_fn = stream

    resp = endpoint_client.post("/search/semantic", json={"query": "Apple revenue", "top_k": 5})
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("application/json")
    assert "event:" not in resp.text  # no SSE frames
    body = resp.json()
    assert body["results"][0]["chunk_id"] == "c1"
    llm.assert_not_called()
    stream.assert_not_called()
