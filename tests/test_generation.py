"""
Story 1 — GenerationProvider seam tests.

All tests are hermetic: StubProvider or an injected fake client only, no live API
and no google-genai SDK required (the SDK import in gemini.py is guarded).
"""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.generation.base import GenerationError, GenerationProvider
from app.generation.factory import build_gen_provider
from app.generation.stub import StubProvider, DEFAULT_STUB_RESPONSE
from app.generation.gemini import GeminiProvider
from app.generation.openai_compat import OpenAICompatProvider


# ── Config helper ─────────────────────────────────────────────────────────────

def _settings(**overrides):
    base = dict(
        gen_provider="stub",
        gemini_api_key="",
        gen_model="gemini-2.0-flash",
        gen_temperature=0.3,
        gen_max_tokens=1024,
        gen_timeout_s=30.0,
        openai_compat_base_url="",
        openai_compat_api_key="",
        openai_compat_model="",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


# ── StubProvider ──────────────────────────────────────────────────────────────

def test_stub_provider_returns_default_string():
    p = StubProvider()
    assert p.provider_name == "stub"
    assert p.generate("any prompt") == DEFAULT_STUB_RESPONSE
    assert isinstance(p, GenerationProvider)  # satisfies the Protocol


def test_stub_provider_is_controllable_per_test():
    p = StubProvider()
    p.set_response("[1] grounded answer with citation")
    assert p.generate("q") == "[1] grounded answer with citation"
    # refusal is an empty string, not an error
    p.set_response("")
    assert p.generate("q") == ""
    # records prompts for introspection
    assert len(p.calls) == 2


def test_stub_provider_can_simulate_error():
    p = StubProvider()
    p.set_error(GenerationError("simulated backend failure"))
    with pytest.raises(GenerationError):
        p.generate("q")


# ── Factory routing ───────────────────────────────────────────────────────────

def test_factory_returns_stub_by_default():
    p = build_gen_provider(_settings(gen_provider="stub"))
    assert isinstance(p, StubProvider)
    assert p.provider_name == "stub"


def test_factory_returns_openai_compat():
    p = build_gen_provider(_settings(
        gen_provider="openai_compat",
        openai_compat_base_url="http://localhost:11434/v1",
        openai_compat_api_key="sk-local",
        openai_compat_model="llama3",
    ))
    assert isinstance(p, OpenAICompatProvider)
    assert p.provider_name == "openai_compat"
    assert p.model_name == "llama3"


def test_factory_routes_to_gemini_or_reports_missing_sdk():
    """With a key, the factory routes to GeminiProvider. In an env without the
    google-genai SDK installed, construction raises a clear GenerationError about the
    missing SDK — either way, routing reached the Gemini branch."""
    cfg = _settings(gen_provider="gemini", gemini_api_key="fake-key")
    try:
        p = build_gen_provider(cfg)
        assert p.provider_name == "gemini"
    except GenerationError as exc:
        assert "google-genai" in str(exc)


def test_factory_raises_on_missing_gemini_key():
    with pytest.raises(GenerationError):
        build_gen_provider(_settings(gen_provider="gemini", gemini_api_key=""))


def test_factory_raises_on_missing_openai_compat_key():
    with pytest.raises(GenerationError):
        build_gen_provider(_settings(
            gen_provider="openai_compat",
            openai_compat_base_url="http://x/v1",
            openai_compat_api_key="",
        ))


def test_factory_raises_on_unknown_provider():
    with pytest.raises(GenerationError):
        build_gen_provider(_settings(gen_provider="banana"))


# ── GeminiProvider with an injected fake client (no SDK needed) ────────────────

def test_gemini_provider_generate_with_fake_client():
    class _Resp:
        text = "Apple's revenue was $394 billion. [1]"

    class _Models:
        def __init__(self):
            self.last = None

        def generate_content(self, model, contents, config):
            self.last = SimpleNamespace(model=model, contents=contents, config=config)
            return _Resp()

    class _Client:
        def __init__(self):
            self.models = _Models()

    fake = _Client()
    p = GeminiProvider(api_key="unused", model_name="gemini-2.0-flash", client=fake,
                       max_tokens=512, temperature=0.1)
    out = p.generate("What is Apple's revenue?")
    assert out == "Apple's revenue was $394 billion. [1]"
    # bound defaults from construction are passed through
    assert fake.models.last.config["max_output_tokens"] == 512
    assert fake.models.last.config["temperature"] == 0.1


def test_gemini_provider_empty_text_is_refusal_not_error():
    class _Resp:
        text = None  # safety block / empty candidate

    class _Client:
        class models:
            @staticmethod
            def generate_content(model, contents, config):
                return _Resp()

    p = GeminiProvider(api_key="unused", client=_Client())
    assert p.generate("q") == ""  # refusal → "", never raises


def test_gemini_provider_call_failure_raises_generation_error():
    class _Client:
        class models:
            @staticmethod
            def generate_content(model, contents, config):
                raise TimeoutError("deadline exceeded")

    p = GeminiProvider(api_key="unused", client=_Client())
    with pytest.raises(GenerationError):
        p.generate("q")


# ── Dead code is gone ─────────────────────────────────────────────────────────

def test_call_hf_api_is_gone_from_codebase():
    """Story 1 deletes the dead HF path entirely — no reference may remain in app/."""
    app_dir = Path(__file__).resolve().parent.parent / "app"
    offenders = [
        str(py) for py in app_dir.rglob("*.py")
        if "_call_hf_api" in py.read_text(encoding="utf-8")
    ]
    assert not offenders, f"_call_hf_api still referenced in: {offenders}"


# ── Generation is a WEB-process concern: the worker must NOT build one ─────────

def test_worker_ctx_has_no_gen_provider(monkeypatch):
    """on_startup must not put a generation provider on ctx — generation lives in the
    web process (interactive/latency-bound), the worker does batch, resumable work."""
    from app import worker as worker_mod

    # Keep on_startup light + side-effect-free: no DB write, no real Chroma, no ST model.
    monkeypatch.setattr(worker_mod, "init_db", lambda: None)
    monkeypatch.setattr(worker_mod, "build_chroma_client", lambda: object())
    monkeypatch.setattr(worker_mod.settings, "embed_provider", "hf_api")  # skip ST load

    ctx: dict = {}
    asyncio.run(worker_mod.on_startup(ctx))

    assert "llm_fn" not in ctx, "worker ctx must not carry a generation provider"
    # sanity: on_startup actually ran and wired its real (batch) resources
    assert "chroma" in ctx and "bm25" in ctx and "session_factory" in ctx
