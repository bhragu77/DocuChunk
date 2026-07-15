"""
build_gen_provider(settings) — the single, config-driven construction site for the
generation backend. Mirrors embedding_providers.build_provider().

GEN_PROVIDER decides the concrete class:
  stub          → StubProvider (default; offline, safe — no live API until configured)
  gemini        → GeminiProvider (requires GEMINI_API_KEY)
  openai_compat → OpenAICompatProvider (requires OPENAI_COMPAT_BASE_URL + _API_KEY)

Raises GenerationError on a missing key/URL for a non-stub provider. The lifespan
wiring catches that and leaves app.state.llm_fn = None, so /generate/answer returns
Story 0's honest "generation_not_configured" response instead of crashing startup.
"""
from __future__ import annotations

import logging
from typing import Callable, Iterator

from app.generation.base import GenerationError, GenerationProvider
from app.generation.stub import StubProvider

logger = logging.getLogger(__name__)


def _est_tokens(text: str) -> int:
    """Rough token estimate (len//4), matching the rest of the codebase."""
    return len(text) // 4


def build_gen_provider(settings) -> GenerationProvider:
    provider = (settings.gen_provider or "stub").lower()

    if provider == "stub":
        logger.info("Generation provider: stub (offline)")
        return StubProvider()

    if provider == "gemini":
        if not settings.gemini_api_key:
            raise GenerationError("GEN_PROVIDER=gemini but GEMINI_API_KEY is not set")
        from app.generation.gemini import GeminiProvider
        return GeminiProvider(
            api_key=settings.gemini_api_key,
            model_name=settings.gen_model,
            timeout_s=settings.gen_timeout_s,
            max_tokens=settings.gen_max_tokens,
            temperature=settings.gen_temperature,
        )

    if provider == "openai_compat":
        if not settings.openai_compat_api_key:
            raise GenerationError("GEN_PROVIDER=openai_compat but OPENAI_COMPAT_API_KEY is not set")
        if not settings.openai_compat_base_url:
            raise GenerationError("GEN_PROVIDER=openai_compat but OPENAI_COMPAT_BASE_URL is not set")
        from app.generation.openai_compat import OpenAICompatProvider
        return OpenAICompatProvider(
            base_url=settings.openai_compat_base_url,
            api_key=settings.openai_compat_api_key,
            model_name=settings.openai_compat_model,
            timeout_s=settings.gen_timeout_s,
            max_tokens=settings.gen_max_tokens,
            temperature=settings.gen_temperature,
        )

    raise GenerationError(
        f"Unknown GEN_PROVIDER={settings.gen_provider!r}; expected 'stub' | 'gemini' | 'openai_compat'."
    )


def build_verify_provider(settings, gen_provider: GenerationProvider | None = None) -> GenerationProvider:
    """
    Story 7 — construct the VERIFICATION-tier provider (citation Tier-2 judge +
    groundedness). Same factory pattern as build_gen_provider but reads VERIFY_*.

    When VERIFY_PROVIDER is unset (the default), fall back to the GENERATION
    provider so a single model still generates AND verifies — exactly the
    pre-Story-7 behavior. Pass gen_provider to reuse the already-built instance
    (avoids a second SDK client / duplicate key check); when it's None we build a
    generation provider on demand (used by unit tests).
    """
    provider = (settings.verify_provider or "").lower()

    if not provider:
        logger.info("Verification provider: reusing generation provider (VERIFY_PROVIDER unset)")
        return gen_provider if gen_provider is not None else build_gen_provider(settings)

    if provider == "stub":
        logger.info("Verification provider: stub (offline)")
        return StubProvider(model_name=settings.verify_model or "stub-verify")

    if provider == "gemini":
        api_key = settings.verify_api_key or settings.gemini_api_key
        if not api_key:
            raise GenerationError("VERIFY_PROVIDER=gemini but neither VERIFY_API_KEY nor GEMINI_API_KEY is set")
        from app.generation.gemini import GeminiProvider
        return GeminiProvider(
            api_key=api_key,
            model_name=settings.verify_model or settings.gen_model,
            timeout_s=settings.gen_timeout_s,
            max_tokens=settings.verify_max_tokens,
            temperature=settings.verify_temperature,
        )

    if provider == "openai_compat":
        api_key = settings.verify_api_key or settings.openai_compat_api_key
        if not api_key:
            raise GenerationError("VERIFY_PROVIDER=openai_compat but no VERIFY_API_KEY / OPENAI_COMPAT_API_KEY is set")
        if not settings.openai_compat_base_url:
            raise GenerationError("VERIFY_PROVIDER=openai_compat but OPENAI_COMPAT_BASE_URL is not set")
        from app.generation.openai_compat import OpenAICompatProvider
        return OpenAICompatProvider(
            base_url=settings.openai_compat_base_url,
            api_key=api_key,
            model_name=settings.verify_model or settings.openai_compat_model,
            timeout_s=settings.gen_timeout_s,
            max_tokens=settings.verify_max_tokens,
            temperature=settings.verify_temperature,
        )

    raise GenerationError(
        f"Unknown VERIFY_PROVIDER={settings.verify_provider!r}; expected '' | 'stub' | 'gemini' | 'openai_compat'."
    )


# ── Role-bound callables (the seam the endpoint reads off app.state) ──────────
# Each wrapper binds the tier's config (max_tokens/temperature) so callers just
# pass a prompt, and logs one `llm_call` line per call for observability (Story 7
# #5): enough to later compute cache hit rate, cost/answer, and verify accuracy.

def make_llm_fn(provider: GenerationProvider, settings) -> Callable[[str], str]:
    """Generation callable — strong model, GEN_* caps, role=generate."""
    def _fn(prompt: str) -> str:
        result = provider.generate(
            prompt, max_tokens=settings.gen_max_tokens, temperature=settings.gen_temperature
        )
        logger.info(
            "llm_call provider=%s model=%s role=generate tokens_in≈%d tokens_out≈%d",
            provider.provider_name, provider.model_name, _est_tokens(prompt), _est_tokens(result),
        )
        return result
    return _fn


def make_llm_stream_fn(provider: GenerationProvider, settings) -> Callable[[str], Iterator[str]]:
    """Streaming generation callable — strong model, GEN_* caps, role=generate(stream)."""
    def _fn(prompt: str) -> Iterator[str]:
        logger.info(
            "llm_call provider=%s model=%s role=generate_stream tokens_in≈%d tokens_out≈(streamed)",
            provider.provider_name, provider.model_name, _est_tokens(prompt),
        )
        return provider.generate_stream(
            prompt, max_tokens=settings.gen_max_tokens, temperature=settings.gen_temperature
        )
    return _fn


def make_verify_fn(provider: GenerationProvider, settings) -> Callable[[str], str]:
    """Verification callable — cheap model, VERIFY_* caps (hard-capped output), role=verify."""
    def _fn(prompt: str) -> str:
        result = provider.generate(
            prompt, max_tokens=settings.verify_max_tokens, temperature=settings.verify_temperature
        )
        logger.info(
            "llm_call provider=%s model=%s role=verify tokens_in≈%d tokens_out≈%d",
            provider.provider_name, provider.model_name, _est_tokens(prompt), _est_tokens(result),
        )
        return result
    return _fn
