"""
GeminiProvider — generation via the google-genai SDK (pip install google-genai).

SDK import is GUARDED (deferred into __init__): this module must import even when
google-genai is NOT installed, so the test suite — which only uses StubProvider or an
injected fake client — never crashes on import. The SDK is loaded only when a real
GeminiProvider is constructed (client=None, i.e. GEN_PROVIDER=gemini). A missing SDK
or a call failure raises GenerationError, which the /generate/answer endpoint catches
and turns into Story 0's honest error shape (never a 500).

The SDK accepts plain dicts for both http_options and generation config
(HttpOptionsOrDict / GenerateContentConfigOrDict), so we pass dicts and never import
`google.genai.types` — that keeps generate() usable with an injected fake client in
tests without the SDK present.
"""
from __future__ import annotations

import logging
from typing import Iterator

from app.generation.base import GenerationError

logger = logging.getLogger(__name__)


class GeminiProvider:
    provider_name = "gemini"

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.0-flash",
        *,
        timeout_s: float = 30.0,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        client=None,  # injectable for tests; a real SDK client is built when None
    ):
        self.model_name = model_name
        self._timeout_s = timeout_s
        self._default_max_tokens = max_tokens
        self._default_temperature = temperature

        if client is not None:
            self._client = client
            return

        if not api_key:
            raise GenerationError("GeminiProvider requires an API key (GEMINI_API_KEY)")

        try:
            from google import genai
        except ImportError as exc:  # SDK optional — only needed for GEN_PROVIDER=gemini
            raise GenerationError(
                "google-genai is not installed. Run `pip install google-genai` "
                "or set GEN_PROVIDER=stub / openai_compat."
            ) from exc

        # http_options.timeout is in MILLISECONDS in the google-genai SDK; a dict is accepted.
        self._client = genai.Client(
            api_key=api_key,
            http_options={"timeout": int(self._timeout_s * 1000)},
        )

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        max_tokens = self._default_max_tokens if max_tokens is None else max_tokens
        temperature = self._default_temperature if temperature is None else temperature

        try:
            resp = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={"max_output_tokens": max_tokens, "temperature": temperature},
            )
        except Exception as exc:  # timeout, transport, API error → unrecoverable
            logger.warning("Gemini generate_content failed: %s", exc)
            raise GenerationError(f"Gemini generation failed: {exc}") from exc

        # resp.text is None on a safety block / empty candidate → treat as refusal ("").
        return (getattr(resp, "text", None) or "").strip()

    def generate_stream(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """
        Story 6 — stream via the google-genai SDK's streaming mode
        (models.generate_content_stream). Each yielded response chunk carries a
        .text delta; we forward the non-empty deltas. A safety block / empty
        candidate simply yields nothing (a refusal). Any transport/API error
        raises GenerationError — the endpoint then falls back to generate().
        """
        max_tokens = self._default_max_tokens if max_tokens is None else max_tokens
        temperature = self._default_temperature if temperature is None else temperature

        try:
            stream = self._client.models.generate_content_stream(
                model=self.model_name,
                contents=prompt,
                config={"max_output_tokens": max_tokens, "temperature": temperature},
            )
            for chunk in stream:
                text = getattr(chunk, "text", None)
                if text:
                    yield text
        except Exception as exc:  # timeout, transport, API error → unrecoverable
            logger.warning("Gemini generate_content_stream failed: %s", exc)
            raise GenerationError(f"Gemini streaming failed: {exc}") from exc
