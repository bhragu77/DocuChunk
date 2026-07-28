"""
Chat model picker — the per-request "gemini" | "offline" generation tier.

All tests are hermetic: no live API, no google-genai SDK, no Ollama. The registry
is exercised with fake providers / a monkeypatched reachability probe, and the
endpoint is driven with the same in-memory SQLite + dependency-override harness the
Story-7 tiering tests use.

Covers the acceptance points from the plan:
  * build_gen_registry offers "offline" ONLY when enabled AND Ollama is reachable;
  * an unconfigured tier is simply absent (never a hard error);
  * GET /generate/models reports the wired tiers + a default;
  * POST /generate/answer routes generation to the SELECTED tier's llm_fn, falls
    back to the default seam for an unknown/absent model, and keys the answer cache
    per model so the two tiers never serve each other's cached answers.
"""
import tempfile
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database import Base, get_db
from app.core.dependencies import get_current_user
from app.generation import factory
from app.generation.factory import build_gen_registry
from app.generation.cache import compute_cache_key
from app.models.document import Document
from app.pipeline.bm25_index import BM25Index
from app.pipeline.retrieval import ScoredChunk


# ── settings + fake provider helpers ────────────────────────────────────────────

def _settings(**overrides):
    base = dict(
        gemini_api_key="",
        gen_model="gemini-3.1-flash-lite",
        gen_temperature=0.3,
        gen_max_tokens=1024,
        gen_timeout_s=30.0,
        offline_enabled=False,
        offline_base_url="http://ollama:11434/v1",
        offline_api_key="ollama",
        offline_model="qwen2.5:1.5b",
        offline_timeout_s=180.0,   # local CPU inference needs its own, larger budget
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class _FakeProvider:
    """Minimal provider — build_gen_registry only reads model_name/provider_name at
    build time and wraps generate() into llm_fn (called later, if at all)."""

    def __init__(self, provider_name, model_name):
        self.provider_name = provider_name
        self.model_name = model_name

    def generate(self, prompt, **kw):
        return f"{self.provider_name} answer"

    def generate_stream(self, prompt, **kw):
        yield f"{self.provider_name} answer"


# ── build_gen_registry ──────────────────────────────────────────────────────────

def test_registry_empty_when_nothing_configured():
    assert build_gen_registry(_settings()) == {}


def test_registry_offline_present_when_enabled_and_reachable(monkeypatch):
    monkeypatch.setattr(factory, "_offline_reachable", lambda url, timeout_s=2.0: True)
    reg = build_gen_registry(_settings(offline_enabled=True))
    assert "offline" in reg
    assert reg["offline"]["model_name"] == "qwen2.5:1.5b"
    assert callable(reg["offline"]["llm_fn"])
    assert callable(reg["offline"]["llm_stream_fn"])
    assert "gemini" not in reg  # no key configured


def test_registry_offline_absent_when_unreachable(monkeypatch):
    monkeypatch.setattr(factory, "_offline_reachable", lambda url, timeout_s=2.0: False)
    reg = build_gen_registry(_settings(offline_enabled=True))
    assert "offline" not in reg  # probe failed → tier hidden, not errored


def test_registry_offline_absent_when_disabled(monkeypatch):
    # Probe would pass, but the flag is off → the probe must not even be consulted.
    monkeypatch.setattr(factory, "_offline_reachable",
                        lambda *a, **k: pytest.fail("probe must not run when disabled"))
    assert "offline" not in build_gen_registry(_settings(offline_enabled=False))


def test_registry_gemini_present_when_key_set(monkeypatch):
    monkeypatch.setattr("app.generation.gemini.GeminiProvider",
                        lambda **kw: _FakeProvider("gemini", kw.get("model_name")))
    reg = build_gen_registry(_settings(gemini_api_key="k"))
    assert "gemini" in reg
    assert reg["gemini"]["model_name"] == "gemini-3.1-flash-lite"


def test_registry_one_bad_tier_does_not_sink_the_other(monkeypatch):
    # gemini construction blows up, offline is fine → offline still available.
    def _boom(**kw):
        raise RuntimeError("sdk missing")
    monkeypatch.setattr("app.generation.gemini.GeminiProvider", _boom)
    monkeypatch.setattr(factory, "_offline_reachable", lambda *a, **k: True)
    reg = build_gen_registry(_settings(gemini_api_key="k", offline_enabled=True))
    assert "gemini" not in reg and "offline" in reg


# ── cache key varies by model ───────────────────────────────────────────────────

def test_cache_key_varies_by_model_but_default_is_stable():
    base = compute_cache_key("u", "d", 1, "q")
    gem = compute_cache_key("u", "d", 1, "q", model="gemini")
    off = compute_cache_key("u", "d", 1, "q", model="offline")
    assert base != gem != off and gem != off
    # Omitting model reproduces the exact pre-picker key (no cache migration).
    assert compute_cache_key("u", "d", 1, "q", model=None) == base


# ══ ENDPOINT harness ════════════════════════════════════════════════════════════

_engine = create_engine("sqlite:///./test_model_picker.db",
                        connect_args={"check_same_thread": False})
_TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=_engine)


class _FakeUser:
    id = "u_picker"
    email = "picker@example.com"


def _override_get_db():
    db = _TestingSession()
    try:
        yield db
    finally:
        db.close()


def _insert_doc(doc_id="docp", version=1):
    db = _TestingSession()
    try:
        if not db.query(Document).filter(Document.id == doc_id).first():
            db.add(Document(
                id=doc_id, user_id="u_picker", filename="f.pdf", original_filename="f.pdf",
                file_path="/tmp/f.pdf", file_size=1, mime_type="application/pdf", version=version,
            ))
            db.commit()
    finally:
        db.close()


def _cand(cid, text, doc_id="docp"):
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
    saved = {k: getattr(app.state, k, None) for k in
             ("bm25", "embed_fn", "llm_fn", "verify_fn", "llm_stream_fn",
              "gen_model_name", "answer_cache", "gen_registry")}
    app.state.bm25 = BM25Index(persist_dir=tempfile.mkdtemp(prefix="bm25_picker_"))
    app.state.embed_fn = lambda text: [0.1] * 384
    app.state.llm_fn = lambda p: "default answer"
    app.state.verify_fn = lambda p: "verified"
    app.state.llm_stream_fn = None
    app.state.gen_model_name = "default-model"
    app.state.answer_cache = None
    app.state.gen_registry = {}
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_db, None)
        for k, v in saved.items():
            setattr(app.state, k, v)
        Base.metadata.drop_all(bind=_engine)


def _entry(name):
    return {"llm_fn": lambda p: f"{name} answer", "llm_stream_fn": None,
            "model_name": f"{name}-model", "provider_name": name}


def test_models_endpoint_lists_tiers_and_default(endpoint_client):
    app.state.gen_registry = {"gemini": _entry("gemini"), "offline": _entry("offline")}
    data = endpoint_client.get("/generate/models").json()
    keys = {m["key"] for m in data["models"]}
    assert keys == {"gemini", "offline"}
    assert data["default"] == "gemini"          # gemini preferred as default
    labels = {m["key"]: m["label"] for m in data["models"]}
    # Labels name the trade-off the user is choosing between, not the plumbing.
    assert "accurate" in labels["gemini"] and "free" in labels["offline"]


def test_models_endpoint_empty_registry(endpoint_client):
    data = endpoint_client.get("/generate/models").json()
    assert data == {"models": [], "default": None}


def _patch_pipeline(monkeypatch, seen):
    from app.generation.citation_validator import ValidationResult
    monkeypatch.setattr("app.routers.search.hybrid_search",
                        lambda **kw: [_cand("c1", "Grounded fact here.")])
    monkeypatch.setattr("app.routers.search.rerank", lambda q, c, top_k: c[:top_k])

    def gen_spy(query, chunks, llm_fn=None, **kwargs):
        seen["llm_fn"] = llm_fn
        seen["model_name"] = kwargs.get("model_name")
        return "The answer [1]."
    monkeypatch.setattr("app.routers.search.generate_answer", gen_spy)
    monkeypatch.setattr("app.routers.search.validate_citations",
                        lambda *a, **k: ValidationResult())
    monkeypatch.setattr("app.routers.search.groundedness_check",
                        lambda *a, **k: {"grounded": True, "confidence": 0.9,
                                         "unsupported_claims": [], "verified": True})


def test_answer_routes_to_selected_model(endpoint_client, monkeypatch):
    offline_fn = lambda p: "offline answer"      # noqa: E731
    app.state.gen_registry = {
        "gemini": {"llm_fn": lambda p: "gem", "llm_stream_fn": None,
                   "model_name": "gemini-3.1-flash-lite", "provider_name": "gemini"},
        "offline": {"llm_fn": offline_fn, "llm_stream_fn": None,
                    "model_name": "qwen2.5:1.5b", "provider_name": "offline"},
    }
    seen = {}
    _patch_pipeline(monkeypatch, seen)
    resp = endpoint_client.post("/generate/answer?stream=false",
                                json={"query": "q?", "doc_id": "docp", "model": "offline"})
    assert resp.status_code == 200, resp.text
    assert seen["llm_fn"] is offline_fn              # selected tier generated
    assert seen["model_name"] == "qwen2.5:1.5b"      # and its model name labels the span


def test_answer_falls_back_to_default_for_unknown_model(endpoint_client, monkeypatch):
    default_fn = app.state.llm_fn                     # the fixture's default seam
    app.state.gen_registry = {"gemini": _entry("gemini")}
    seen = {}
    _patch_pipeline(monkeypatch, seen)
    resp = endpoint_client.post("/generate/answer?stream=false",
                                json={"query": "q?", "doc_id": "docp", "model": "does-not-exist"})
    assert resp.status_code == 200, resp.text
    assert seen["llm_fn"] is default_fn              # unknown → default, never an error
    assert seen["model_name"] == "default-model"


def test_answer_uses_default_when_no_model_given(endpoint_client, monkeypatch):
    default_fn = app.state.llm_fn
    app.state.gen_registry = {"offline": _entry("offline")}
    seen = {}
    _patch_pipeline(monkeypatch, seen)
    resp = endpoint_client.post("/generate/answer?stream=false",
                                json={"query": "q?", "doc_id": "docp"})
    assert resp.status_code == 200, resp.text
    assert seen["llm_fn"] is default_fn              # no model field → unchanged behavior
