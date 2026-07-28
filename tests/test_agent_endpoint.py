"""
POST /generate/agent — integration tests (hermetic).

StubbedLLM plays two roles off the SAME callable, routed by prompt content:
  * planner prompts (contain "RETRIEVAL PLANNING agent") → a queued JSON action
  * the grounded-answer prompt                            → the answer prose
Retrieval is monkeypatched in the agent tools module, so no real Chroma/BM25 is
touched. Asserts the agent path reuses /generate/answer's schema (+ agent_steps)
and that guardrails surface through the endpoint.
"""
import json
import tempfile

import pytest
from fastapi.testclient import TestClient

from app.core.dependencies import get_current_user, get_chroma
from app.main import app
from app.pipeline.retrieval import ScoredChunk


class _FakeUser:
    id = "u_agent"
    email = "agent@example.com"


def _chunk(cid, text, doc="docA", page=1):
    return ScoredChunk(
        chunk_id=cid, text=text, doc_id=doc, source="a.pdf",
        page_number=page, fused_score=1.0, reranker_score=0.9,
        metadata={"source": "a.pdf", "page_number": page, "doc_id": doc},
    )


@pytest.fixture
def client(monkeypatch):
    app.dependency_overrides[get_current_user] = lambda: _FakeUser()
    app.dependency_overrides[get_chroma] = lambda: "FAKE_CHROMA"
    prev = {k: getattr(app.state, k, None)
            for k in ("bm25", "embed_fn", "llm_fn", "llm_stream_fn",
                      "verify_fn", "answer_cache", "gen_model_name")}
    app.state.bm25 = "FAKE_BM25"
    app.state.embed_fn = lambda text: [0.1] * 8
    app.state.llm_fn = None
    app.state.llm_stream_fn = None
    app.state.verify_fn = lambda prompt: "none"   # groundedness: all supported
    app.state.answer_cache = None
    app.state.gen_model_name = "stub-1"

    # Retrieve returns two chunks per query; rerank is identity-truncate.
    monkeypatch.setattr(
        "app.generation.agent.tools.hybrid_search",
        lambda **kw: [_chunk("c1", "Flo is the accountant."),
                      _chunk("c2", "Leonardo is the artist.")],
    )
    monkeypatch.setattr(
        "app.generation.agent.tools.rerank",
        lambda query, candidates, top_k=8, reranker=None: candidates[:top_k],
    )
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_chroma, None)
        for k, v in prev.items():
            setattr(app.state, k, v)


def _wire_llm(*actions, answer="Flo is the accountant [1]."):
    """An llm_fn that returns queued JSON actions to the planner and prose to gen."""
    queue = list(actions)
    state = {"i": 0}

    def fn(prompt: str) -> str:
        if "RETRIEVAL PLANNING agent" in prompt:
            i = state["i"]
            state["i"] += 1
            return queue[i] if i < len(queue) else json.dumps({"tool": "finish", "args": {}})
        return answer

    app.state.llm_fn = fn
    return fn


def _act(tool, **args):
    return json.dumps({"tool": tool, "args": args})


def _finish():
    return json.dumps({"tool": "finish", "args": {}})


def _parse_sse(text: str):
    """Return list of (event, data-dict) from an SSE body."""
    events = []
    event = None
    for line in text.splitlines():
        if line.startswith("event: "):
            event = line[len("event: "):].strip()
        elif line.startswith("data: "):
            events.append((event, json.loads(line[len("data: "):])))
    return events


# ── Non-streaming: multi-hop, schema parity, agent_steps ──────────────────────

def test_agent_multi_hop_non_streaming(client):
    _wire_llm(_act("retrieve_docs", query="who is the accountant"),
              _act("retrieve_docs", query="who is the artist"),
              _finish())

    resp = client.post("/generate/agent?stream=false",
                       json={"query": "List the crew and roles", "doc_id": "docA"})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    # Reuses /generate/answer's schema…
    assert body["answer"] == "Flo is the accountant [1]."
    assert "confidence" in body and "confidence_signals" in body
    assert [c["chunk_id"] for c in body["all_sources"]] == ["c1", "c2"]
    # …plus the executed trajectory.
    assert len(body["agent_steps"]) == 2
    assert all(s["tool"] == "retrieve_docs" for s in body["agent_steps"])
    assert body["retrieval_scope"] == "agent"


# ── Fallback: planner finishes immediately → one fallback retrieval still runs ─

def test_agent_fallback_when_planner_gives_up(client):
    _wire_llm(_finish())

    resp = client.post("/generate/agent?stream=false",
                       json={"query": "who is the accountant", "doc_id": "docA"})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["answer"]  # still produced an answer from fallback evidence
    assert len(body["agent_steps"]) == 1
    assert body["agent_steps"][0]["summary"].startswith("[fallback]")


# ── generation_not_configured: no llm_fn wired ────────────────────────────────

def test_agent_generation_not_configured(client):
    app.state.llm_fn = None  # explicit

    resp = client.post("/generate/agent?stream=false",
                       json={"query": "anything", "doc_id": "docA"})
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["error"] == "generation_not_configured"
    assert body["answer"] is None
    assert body["agent_steps"] == []


# ── Streaming: live step events, then tokens, then verification ───────────────

def test_agent_streaming_emits_steps_then_verification(client):
    _wire_llm(_act("retrieve_docs", query="who is the accountant"), _finish())

    resp = client.post("/generate/agent?stream=true",
                       json={"query": "who is the accountant", "doc_id": "docA"})
    assert resp.status_code == 200, resp.text
    events = _parse_sse(resp.text)
    kinds = [e for e, _ in events]

    assert "step" in kinds
    assert "token" in kinds
    assert kinds[-1] == "verification"

    step_events = [d for e, d in events if e == "step"]
    assert step_events[0]["tool"] == "retrieve_docs"

    verification = events[-1][1]
    assert verification["answer"].strip() == "Flo is the accountant [1]."
    assert len(verification["agent_steps"]) == 1
