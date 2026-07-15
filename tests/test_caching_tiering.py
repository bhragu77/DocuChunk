"""
Story 7 — Caching + Model Tiering tests.

Three areas:
  * MODEL TIERING — build_verify_provider (separate tier vs reuse-gen fallback);
    make_verify_fn/make_llm_fn bind the right output caps; the endpoint routes
    generation→llm_fn and validation+groundedness→verify_fn.
  * CACHING — deterministic version-aware key; InMemory LRU + TTL; backend
    selection; endpoint hit/miss; streaming hit fast path; the full
    miss→hit→version-bump→miss integration arc.
  * TOKEN BUDGETING — over-budget prompts trim lowest-ranked sources first.

Hermetic: StubProvider / recording fakes only, retrieval patched, a throwaway
SQLite DB for doc_version. No live API, no Redis.
"""
import json
import tempfile
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from unittest.mock import MagicMock

from app.main import app
from app.database import Base, get_db
from app.core.dependencies import get_current_user
from app.generation.cache import (
    InMemoryAnswerCache,
    RedisAnswerCache,
    build_cache,
    compute_cache_key,
    normalize_query,
)
from app.generation.factory import (
    build_gen_provider,
    build_verify_provider,
    make_llm_fn,
    make_verify_fn,
)
from app.generation.prompt_builder import build_grounded_prompt
from app.generation.stub import StubProvider
from app.pipeline.bm25_index import BM25Index
from app.pipeline.retrieval import ScoredChunk


# ══ MODEL TIERING ══════════════════════════════════════════════════════════════

def _tier_settings(**over):
    base = dict(
        gen_provider="stub",
        verify_provider="",
        verify_model="",
        verify_api_key="",
        gemini_api_key="",
        gen_model="gemini-2.0-flash",
        gen_timeout_s=30.0,
        verify_max_tokens=64,
        verify_temperature=0.1,
        gen_max_tokens=1024,
        gen_temperature=0.3,
        openai_compat_base_url="",
        openai_compat_api_key="",
        openai_compat_model="",
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_build_verify_provider_falls_back_to_gen_when_unset():
    """VERIFY_PROVIDER unset → reuse the generation provider (backward compatible)."""
    s = _tier_settings(verify_provider="")
    gen = build_gen_provider(s)
    # Passing the gen provider reuses the exact instance (no duplicate client).
    assert build_verify_provider(s, gen_provider=gen) is gen
    # Without one, it builds a generation provider of the same kind.
    assert isinstance(build_verify_provider(s), StubProvider)


def test_build_verify_provider_returns_separate_provider_when_set():
    """VERIFY_PROVIDER set → a distinct verification provider (its own model)."""
    s = _tier_settings(verify_provider="stub", verify_model="cheap-verify")
    verify = build_verify_provider(s)
    gen = build_gen_provider(s)
    assert isinstance(verify, StubProvider)
    assert verify.model_name == "cheap-verify"
    assert verify is not gen


def test_make_fns_bind_the_right_output_caps():
    """make_verify_fn uses VERIFY_MAX_TOKENS; make_llm_fn uses GEN_MAX_TOKENS."""
    class _Recorder:
        provider_name = "rec"
        model_name = "rec-1"
        def __init__(self):
            self.calls = []
        def generate(self, prompt, *, max_tokens, temperature):
            self.calls.append((max_tokens, temperature))
            return "ok"

    s = _tier_settings()
    rec = _Recorder()

    make_verify_fn(rec, s)("verify this")
    assert rec.calls[-1] == (64, 0.1)          # VERIFY_MAX_TOKENS / VERIFY_TEMPERATURE

    make_llm_fn(rec, s)("generate this")
    assert rec.calls[-1] == (1024, 0.3)        # GEN_MAX_TOKENS / GEN_TEMPERATURE


# ══ CACHING — unit ═════════════════════════════════════════════════════════════

def test_cache_key_deterministic():
    a = compute_cache_key("u1", "d1", 1, "What is X?")
    b = compute_cache_key("u1", "d1", 1, "What is X?")
    assert a == b


def test_cache_key_changes_with_doc_version():
    v1 = compute_cache_key("u1", "d1", 1, "What is X?")
    v2 = compute_cache_key("u1", "d1", 2, "What is X?")
    assert v1 != v2  # a re-upload (version bump) busts the cache


def test_cache_key_normalizes_case_and_whitespace():
    """Case/whitespace-only differences collapse to the SAME key (normalization);
    a genuinely different query yields a different key."""
    assert normalize_query("  What Is X? ") == "what is x?"
    same = compute_cache_key("u1", "d1", 1, "  What Is X? ")
    also = compute_cache_key("u1", "d1", 1, "what is x?")
    assert same == also
    different = compute_cache_key("u1", "d1", 1, "What is Y?")
    assert different != same


def test_inmemory_lru_evicts_oldest():
    cache = InMemoryAnswerCache(max_entries=2)
    cache.set("k1", {"a": 1}, ttl=3600)
    cache.set("k2", {"a": 2}, ttl=3600)
    cache.set("k3", {"a": 3}, ttl=3600)  # overflow → evict oldest (k1)
    assert cache.get("k1") is None
    assert cache.get("k2") == {"a": 2}
    assert cache.get("k3") == {"a": 3}


def test_inmemory_lru_touch_on_get_protects_entry():
    """Reading k1 marks it most-recently-used, so the next insert evicts k2."""
    cache = InMemoryAnswerCache(max_entries=2)
    cache.set("k1", {"a": 1}, ttl=3600)
    cache.set("k2", {"a": 2}, ttl=3600)
    assert cache.get("k1") == {"a": 1}   # touch k1
    cache.set("k3", {"a": 3}, ttl=3600)  # evict LRU = k2
    assert cache.get("k2") is None
    assert cache.get("k1") == {"a": 1}


def test_inmemory_ttl_expiry(monkeypatch):
    import app.generation.cache as cache_mod
    clock = {"t": 1000.0}
    monkeypatch.setattr(cache_mod.time, "time", lambda: clock["t"])

    cache = InMemoryAnswerCache()
    cache.set("k", {"a": 1}, ttl=10)
    assert cache.get("k") == {"a": 1}       # within TTL
    clock["t"] = 1011.0                       # 11s later, past the 10s TTL
    assert cache.get("k") is None            # expired → miss


def test_build_cache_backends():
    assert build_cache(SimpleNamespace(cache_backend="none")) is None
    mem = build_cache(SimpleNamespace(cache_backend="memory", cache_max_memory_entries=5))
    assert isinstance(mem, InMemoryAnswerCache)
    assert build_cache(SimpleNamespace(cache_backend="bogus")) is None  # unknown → disabled


def test_redis_cache_roundtrip_with_fake_client():
    """RedisAnswerCache stores JSON with a TTL and reads it back."""
    store = {}

    class _FakeRedis:
        def get(self, key):
            return store.get(key)
        def set(self, key, value, ex=None):
            store[key] = value

    cache = RedisAnswerCache(_FakeRedis())
    cache.set("k", {"answer": "hi", "confidence": 0.9}, ttl=3600)
    assert cache.get("k") == {"answer": "hi", "confidence": 0.9}
    # Stored value is JSON text under a namespaced key.
    assert any(k.endswith("k") for k in store)
    assert json.loads(next(iter(store.values())))["answer"] == "hi"
    assert cache.get("missing") is None


# ══ TOKEN BUDGETING ════════════════════════════════════════════════════════════

def _sc(cid, text, doc_id="docA"):
    return ScoredChunk(
        chunk_id=cid, text=text, doc_id=doc_id, source="m.pdf", page_number=1,
        fused_score=0.5, dense_rank=1, bm25_rank=1, reranker_score=1.0,
    )


def test_input_budget_trims_lowest_ranked_first():
    """Over-budget prompt: the highest-ranked chunk is preserved whole; the
    lowest-ranked is truncated first."""
    top = "ALPHA " * 120   # index 0 — must survive intact
    low = "OMEGA " * 120    # index 1 — trimmed first
    chunks = [_sc("c0", top), _sc("c1", low)]

    # Budget must leave room for the top chunk (≈180 tok) fully plus a PARTIAL low
    # chunk, on top of the fixed prompt overhead (≈405 tok after the generation-
    # quality prompt rewrite grew the system instructions). 700 fits that window.
    prompt = build_grounded_prompt(
        "question", chunks, settings=SimpleNamespace(gen_max_input_tokens=700)
    )

    assert prompt.count("ALPHA") == 120           # top-ranked fully preserved
    assert 0 < prompt.count("OMEGA") < 120        # lowest-ranked truncated


def test_input_budget_noop_when_under_cap():
    """A comfortably-sized prompt is unchanged by the budget (both chunks whole)."""
    chunks = [_sc("c0", "small top text"), _sc("c1", "small low text")]
    prompt = build_grounded_prompt(
        "question", chunks, settings=SimpleNamespace(gen_max_input_tokens=4000)
    )
    assert "small top text" in prompt
    assert "small low text" in prompt


# ══ ENDPOINT — tiering + caching ═══════════════════════════════════════════════

from app.models.document import Document
from app.routers.search import Citation, GenerateAnswerResponse

_TEST_DB_URL = "sqlite:///./test_story7.db"
_engine = create_engine(_TEST_DB_URL, connect_args={"check_same_thread": False})
_TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


class _FakeUser:
    id = "u_story7"
    email = "story7@example.com"


def _override_get_db():
    db = _TestingSession()
    try:
        yield db
    finally:
        db.close()


def _insert_doc(doc_id="doc7", user_id="u_story7", version=1):
    db = _TestingSession()
    try:
        existing = db.query(Document).filter(Document.id == doc_id).first()
        if existing:
            existing.version = version
        else:
            db.add(Document(
                id=doc_id, user_id=user_id, filename="f.pdf", original_filename="f.pdf",
                file_path="/tmp/f.pdf", file_size=1, mime_type="application/pdf", version=version,
            ))
        db.commit()
    finally:
        db.close()


def _cand(cid, text, doc_id="doc7"):
    return ScoredChunk(
        chunk_id=cid, text=text, doc_id=doc_id, source="m.pdf", page_number=1,
        fused_score=0.5, dense_rank=1, bm25_rank=1, reranker_score=1.0,
        metadata={"source": "m.pdf", "page_number": 1},
    )


@pytest.fixture
def endpoint_client():
    from app.models import user, document, job  # noqa: F401 — register tables
    Base.metadata.create_all(bind=_engine)
    app.dependency_overrides[get_current_user] = lambda: _FakeUser()
    app.dependency_overrides[get_db] = _override_get_db
    saved = {k: getattr(app.state, k, None)
             for k in ("bm25", "embed_fn", "llm_fn", "verify_fn", "llm_stream_fn", "answer_cache")}
    app.state.bm25 = BM25Index(persist_dir=tempfile.mkdtemp(prefix="bm25_s7_"))
    app.state.embed_fn = lambda text: [0.1] * 384
    app.state.llm_fn = None
    app.state.verify_fn = None
    app.state.llm_stream_fn = None
    app.state.answer_cache = None
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_db, None)
        for k, v in saved.items():
            setattr(app.state, k, v)
        Base.metadata.drop_all(bind=_engine)


def _patch_retrieval(monkeypatch, cands):
    monkeypatch.setattr("app.routers.search.hybrid_search", lambda **kw: cands)
    monkeypatch.setattr("app.routers.search.rerank", lambda q, c, top_k: c[:top_k])


def _parse_sse(text: str) -> list[dict]:
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


def test_endpoint_routes_generate_to_llm_and_verify_to_verify_fn(endpoint_client, monkeypatch):
    """Tiering: generate_answer receives llm_fn; validate_citations and
    groundedness_check receive verify_fn — distinct objects."""
    from app.generation.citation_validator import ValidationResult

    _patch_retrieval(monkeypatch, [_cand("c1", "Apple revenue was $394 billion.")])

    gen_sentinel = lambda p: "generated"      # noqa: E731
    verify_sentinel = lambda p: "verified"    # noqa: E731
    app.state.llm_fn = gen_sentinel
    app.state.verify_fn = verify_sentinel

    seen = {}

    def gen_spy(query, chunks, llm_fn=None, doc_chunks_fetcher=None, **kwargs):
        seen["generate"] = llm_fn
        return "The answer [1]."

    def validate_spy(answer, indices, chunks, llm_fn=None, use_llm=True):
        seen["validate"] = llm_fn
        return ValidationResult()

    def ground_spy(query, draft, chunks, llm_fn=None, answer_task=None):
        seen["ground"] = llm_fn
        return {"grounded": True, "confidence": 0.9, "unsupported_claims": [], "verified": True}

    monkeypatch.setattr("app.routers.search.generate_answer", gen_spy)
    monkeypatch.setattr("app.routers.search.validate_citations", validate_spy)
    monkeypatch.setattr("app.routers.search.groundedness_check", ground_spy)

    resp = endpoint_client.post("/generate/answer?stream=false", json={"query": "Apple revenue?"})
    assert resp.status_code == 200, resp.text
    assert seen["generate"] is gen_sentinel          # strong model generates
    assert seen["validate"] is verify_sentinel       # cheap model validates
    assert seen["ground"] is verify_sentinel         # cheap model verifies


def test_verify_fn_falls_back_to_llm_fn_when_unset(endpoint_client, monkeypatch):
    """No verify_fn wired → verification reuses llm_fn (backward compatible)."""
    from app.generation.citation_validator import ValidationResult

    _patch_retrieval(monkeypatch, [_cand("c1", "Apple revenue was $394 billion.")])
    only_llm = lambda p: "x"  # noqa: E731
    app.state.llm_fn = only_llm
    app.state.verify_fn = None  # not wired

    seen = {}
    monkeypatch.setattr("app.routers.search.generate_answer",
                        lambda query, chunks, llm_fn=None, doc_chunks_fetcher=None, **kw: (seen.__setitem__("gen", llm_fn), "A [1].")[1])
    monkeypatch.setattr("app.routers.search.validate_citations",
                        lambda a, i, c, llm_fn=None, use_llm=True: (seen.__setitem__("val", llm_fn), ValidationResult())[1])
    monkeypatch.setattr("app.routers.search.groundedness_check",
                        lambda q, d, c, llm_fn=None, answer_task=None: (seen.__setitem__("gr", llm_fn), {"grounded": True, "confidence": 1.0, "unsupported_claims": [], "verified": True})[1])

    endpoint_client.post("/generate/answer?stream=false", json={"query": "q"})
    assert seen["gen"] is only_llm
    assert seen["val"] is only_llm   # fell back
    assert seen["gr"] is only_llm    # fell back


def test_cache_miss_then_hit_skips_llm(endpoint_client, monkeypatch):
    """First request (miss) calls the LLM and caches; second identical request
    (hit) returns the same body with ZERO further LLM calls."""
    _patch_retrieval(monkeypatch, [_cand("c1", "Apple Inc. revenue was $394 billion.")])
    _insert_doc(version=1)

    stub = StubProvider()
    stub.set_response("Apple Inc. revenue was $394 billion [1].")
    app.state.llm_fn = lambda p: stub.generate(p)
    app.state.answer_cache = InMemoryAnswerCache()

    body = {"query": "What is Apple's revenue?", "doc_id": "doc7"}

    r1 = endpoint_client.post("/generate/answer?stream=false", json=body)
    assert r1.status_code == 200, r1.text
    assert len(stub.calls) >= 1
    calls_after_first = len(stub.calls)

    r2 = endpoint_client.post("/generate/answer?stream=false", json=body)
    assert r2.status_code == 200, r2.text
    assert len(stub.calls) == calls_after_first      # NO new LLM calls on the hit
    assert r2.json() == r1.json()                    # identical response


def test_cache_hit_streaming_fast_path(endpoint_client, monkeypatch):
    """A pre-populated cache serves a streaming request as one token event (full
    answer) + one verification event — no generation, no LLM call."""
    _insert_doc(version=1)
    cache = InMemoryAnswerCache()
    app.state.answer_cache = cache
    app.state.llm_fn = MagicMock(side_effect=AssertionError("llm must not be called on a cache hit"))

    cached = GenerateAnswerResponse(
        query="cached q",
        answer="Cached answer [1].",
        citations=[Citation(chunk_id="c1", source="m.pdf", page_number=1, text_excerpt="x")],
        cited_sources=[Citation(chunk_id="c1", source="m.pdf", page_number=1, text_excerpt="x")],
        grounded=True, confidence=0.9, unsupported_claims=[],
    ).model_dump()
    key = compute_cache_key("u_story7", "doc7", 1, "cached q")
    cache.set(key, cached, ttl=3600)

    resp = endpoint_client.post("/generate/answer?stream=true",
                                json={"query": "cached q", "doc_id": "doc7"})
    assert resp.status_code == 200, resp.text
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(resp.text)

    tokens = [e for e in events if e["event"] == "token"]
    verifications = [e for e in events if e["event"] == "verification"]
    assert len(tokens) == 1
    assert tokens[0]["data"]["done"] is True
    assert tokens[0]["data"]["full_answer"] == "Cached answer [1]."
    assert len(verifications) == 1
    assert verifications[0]["data"] == cached


def test_cache_backend_none_calls_llm_every_time(endpoint_client, monkeypatch):
    """CACHE_BACKEND=none (answer_cache is None) → no caching; the LLM runs on
    every identical request."""
    _patch_retrieval(monkeypatch, [_cand("c1", "Apple Inc. revenue was $394 billion.")])
    _insert_doc(version=1)

    stub = StubProvider()
    stub.set_response("Apple Inc. revenue was $394 billion [1].")
    app.state.llm_fn = lambda p: stub.generate(p)
    app.state.answer_cache = None  # caching disabled

    body = {"query": "What is Apple's revenue?", "doc_id": "doc7"}
    endpoint_client.post("/generate/answer?stream=false", json=body)
    after_first = len(stub.calls)
    endpoint_client.post("/generate/answer?stream=false", json=body)
    assert len(stub.calls) > after_first  # called again — nothing cached


def test_integration_miss_hit_versionbump_miss(endpoint_client, monkeypatch):
    """Full arc: miss → hit (zero LLM) → re-upload (version++) → miss again
    (cache busted by the version-aware key)."""
    _patch_retrieval(monkeypatch, [_cand("c1", "Apple Inc. revenue was $394 billion.")])
    _insert_doc(version=1)

    stub = StubProvider()
    stub.set_response("Apple Inc. revenue was $394 billion [1].")
    app.state.llm_fn = lambda p: stub.generate(p)
    app.state.answer_cache = InMemoryAnswerCache()

    body = {"query": "What is Apple's revenue?", "doc_id": "doc7"}

    endpoint_client.post("/generate/answer?stream=false", json=body)  # miss
    n1 = len(stub.calls)
    assert n1 >= 1

    endpoint_client.post("/generate/answer?stream=false", json=body)  # hit
    assert len(stub.calls) == n1                                       # no new calls

    _insert_doc(version=2)                                             # re-upload bumps version
    endpoint_client.post("/generate/answer?stream=false", json=body)  # miss again
    assert len(stub.calls) > n1                                        # cache busted → LLM ran
