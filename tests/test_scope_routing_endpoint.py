"""
Scope-Aware Retrieval Routing — integration + prompt/groundedness wiring.

  * A GLOBAL query with a doc_id feeds the WHOLE document (fetch_whole_document)
    and BYPASSES hybrid_search / rerank.
  * A LOCAL query still runs the existing top-k hybrid path.
  * The response echoes retrieval_scope + answer_task (additive fields).
  * The prompt builder injects a TASK INSTRUCTIONS block for non-ANSWER tasks.
  * groundedness_check relaxes its verifier prompt for INFER answers.

Hermetic: StubProvider + monkeypatched retrieval. No live API, no real Chroma.
"""
import tempfile

import pytest
from fastapi.testclient import TestClient

from app.core.dependencies import get_current_user
from app.generation.prompt_builder import build_grounded_prompt
from app.generation.query_classifier import AnswerTask
from app.generation.stub import StubProvider
from app.main import app
from app.pipeline.bm25_index import BM25Index
from app.pipeline.retrieval import ScoredChunk, groundedness_check


class _FakeUser:
    id = "u_scope"
    email = "scope@example.com"


def _cand(cid, text, doc_id="docK", source="keebler.pdf", page=1, idx=0):
    return ScoredChunk(
        chunk_id=cid, text=text, doc_id=doc_id, source=source, page_number=page,
        fused_score=1.0, metadata={"source": source, "page_number": page, "chunk_index": idx,
                                   "doc_id": doc_id},
    )


@pytest.fixture
def endpoint_client():
    app.dependency_overrides[get_current_user] = lambda: _FakeUser()
    prev = {k: getattr(app.state, k, None)
            for k in ("bm25", "embed_fn", "llm_fn", "llm_stream_fn", "verify_fn", "answer_cache")}
    app.state.bm25 = BM25Index(persist_dir=tempfile.mkdtemp(prefix="bm25_scope_"))
    app.state.embed_fn = lambda text: [0.1] * 384
    app.state.llm_fn = None
    app.state.llm_stream_fn = None
    app.state.verify_fn = None
    app.state.answer_cache = None
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        for k, v in prev.items():
            setattr(app.state, k, v)


def _wire_stub(response_text: str):
    stub = StubProvider()
    stub.set_response(response_text)
    app.state.llm_fn = lambda prompt: stub.generate(prompt)
    app.state.verify_fn = lambda prompt: "none"  # groundedness: all supported
    return stub


# ── GLOBAL query feeds the whole doc and bypasses hybrid/rerank ────────────────

def test_global_query_uses_whole_document_not_topk(endpoint_client, monkeypatch):
    calls = {"hybrid": 0, "whole": 0}

    # Whole document = all 4 elf chunks (top-k would only surface a few).
    whole_doc = [
        _cand("k1", "Flo was the accountant.", idx=0),
        _cand("k2", "Leonardo was the artist.", idx=1),
        _cand("k3", "Sam was the peanut butter baker.", idx=2),
        _cand("k4", "Fast Eddie wrapped the products.", idx=3),
    ]

    def fake_whole(chroma, user_id, doc_id):
        calls["whole"] += 1
        return whole_doc

    def fake_hybrid(**kw):
        calls["hybrid"] += 1
        return [_cand("k1", "Flo was the accountant.")]

    monkeypatch.setattr("app.routers.search.fetch_whole_document", fake_whole)
    monkeypatch.setattr("app.routers.search.hybrid_search", fake_hybrid)
    _wire_stub("Flo [1], Leonardo [2], Sam [3], Fast Eddie [4].")

    resp = endpoint_client.post(
        "/generate/answer?stream=false",
        json={"query": "List all elves along with their roles", "doc_id": "docK"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert calls["whole"] == 1, "GLOBAL query must fetch the whole document"
    assert calls["hybrid"] == 0, "GLOBAL query must NOT run top-k hybrid search"
    assert body["retrieval_scope"] == "global"
    assert body["answer_task"] == "enumerate"
    # All four chunks reached the model as citations (nothing dropped by top-k).
    assert [c["chunk_id"] for c in body["all_sources"]] == ["k1", "k2", "k3", "k4"]


# ── LOCAL query keeps the existing top-k path ─────────────────────────────────

def test_local_query_still_uses_hybrid_topk(endpoint_client, monkeypatch):
    calls = {"hybrid": 0, "whole": 0}

    monkeypatch.setattr(
        "app.routers.search.fetch_whole_document",
        lambda *a, **k: (calls.__setitem__("whole", calls["whole"] + 1), [])[1],
    )

    def fake_hybrid(**kw):
        calls["hybrid"] += 1
        return [_cand("k1", "Flo was the accountant.")]

    monkeypatch.setattr("app.routers.search.hybrid_search", fake_hybrid)
    monkeypatch.setattr("app.routers.search.rerank", lambda q, c, top_k: c[:top_k])
    _wire_stub("Flo is the accountant [1].")

    resp = endpoint_client.post(
        "/generate/answer?stream=false",
        json={"query": "Who is the accountant?", "doc_id": "docK"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert calls["hybrid"] >= 1
    assert calls["whole"] == 0, "LOCAL query must not fetch the whole document"
    assert body["retrieval_scope"] == "local"
    assert body["answer_task"] == "answer"


# ── Empty whole-doc fetch falls back to the LOCAL path (never 500) ────────────

def test_global_query_falls_back_when_whole_doc_empty(endpoint_client, monkeypatch):
    calls = {"hybrid": 0}
    monkeypatch.setattr("app.routers.search.fetch_whole_document", lambda *a, **k: [])

    def fake_hybrid(**kw):
        calls["hybrid"] += 1
        return [_cand("k1", "Flo was the accountant.")]

    monkeypatch.setattr("app.routers.search.hybrid_search", fake_hybrid)
    monkeypatch.setattr("app.routers.search.rerank", lambda q, c, top_k: c[:top_k])
    _wire_stub("Flo [1].")

    resp = endpoint_client.post(
        "/generate/answer?stream=false",
        json={"query": "Summarize the document", "doc_id": "docK"},
    )
    assert resp.status_code == 200, resp.text
    # Whole-doc was empty → fell through to top-k hybrid, still answered.
    assert calls["hybrid"] >= 1


# ── SCOPE_ROUTING_ENABLED=false forces LOCAL for everyone ─────────────────────

def test_scope_routing_disabled_forces_local(endpoint_client, monkeypatch):
    from types import SimpleNamespace
    from app.config import get_settings

    base = get_settings()
    disabled = SimpleNamespace(**{**base.__dict__, "scope_routing_enabled": False})
    monkeypatch.setattr("app.routers.search.get_settings", lambda: disabled)

    calls = {"hybrid": 0, "whole": 0}
    monkeypatch.setattr(
        "app.routers.search.fetch_whole_document",
        lambda *a, **k: (calls.__setitem__("whole", calls["whole"] + 1), [])[1],
    )
    monkeypatch.setattr(
        "app.routers.search.hybrid_search",
        lambda **kw: (calls.__setitem__("hybrid", calls["hybrid"] + 1),
                      [_cand("k1", "Flo was the accountant.")])[1],
    )
    monkeypatch.setattr("app.routers.search.rerank", lambda q, c, top_k: c[:top_k])
    _wire_stub("Flo [1].")

    resp = endpoint_client.post(
        "/generate/answer?stream=false",
        json={"query": "List all elves and their roles", "doc_id": "docK"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert calls["whole"] == 0, "routing disabled → no whole-doc fetch"
    assert calls["hybrid"] >= 1
    assert body["retrieval_scope"] == "local"


# ── Prompt builder injects the TASK INSTRUCTIONS block ────────────────────────

def test_prompt_builder_adds_task_block_for_table():
    chunks = [_cand("k1", "Flo was the accountant.")]
    prompt = build_grounded_prompt("map each elf to their role", chunks,
                                   answer_task=AnswerTask.TABLE)
    assert "TASK INSTRUCTIONS:" in prompt
    assert "Markdown table" in prompt


def test_prompt_builder_no_task_block_for_plain_answer():
    chunks = [_cand("k1", "Flo was the accountant.")]
    prompt = build_grounded_prompt("who is the accountant", chunks, answer_task=None)
    assert "TASK INSTRUCTIONS:" not in prompt
    prompt2 = build_grounded_prompt("who is the accountant", chunks,
                                    answer_task=AnswerTask.ANSWER)
    assert "TASK INSTRUCTIONS:" not in prompt2


# ── Groundedness relaxes its prompt for INFER answers ─────────────────────────

def test_groundedness_prompt_relaxes_for_infer():
    seen = {}

    def spy_llm(prompt):
        seen["prompt"] = prompt
        return "none"

    chunks = [_cand("k1", "Flo was the accountant. Leonardo was the artist.")]

    # Strict default (no task): the classic "NOT directly supported" wording.
    groundedness_check("q", "A logistics elf would manage shipping.", chunks, llm_fn=spy_llm)
    assert "directly supported" in seen["prompt"].lower()

    # INFER: reasonable inference is explicitly allowed.
    groundedness_check("q", "A logistics elf would manage shipping.", chunks,
                       llm_fn=spy_llm, answer_task=AnswerTask.INFER)
    assert "reasonable inference" in seen["prompt"].lower()
