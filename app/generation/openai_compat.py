"""
OpenAICompatProvider — generation via any OpenAI-compatible /chat/completions
endpoint (DeepSeek, Groq, Together, a local Ollama, etc.). The fallback provider
for local dev without a Gemini key, or when Gemini isn't available.

Uses httpx (already a project dependency). Base URL + API key + model come from
config. Any transport/HTTP error raises GenerationError; the endpoint turns that
into Story 0's honest error shape. A missing choice/content is treated as a refusal
("") rather than an error.
"""
from __future__ import annotations

import json
import logging
from typing import Iterator

import httpx

from app.generation.base import GenerationError

logger = logging.getLogger(__name__)


class OpenAICompatProvider:
    provider_name = "openai_compat"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        *,
        timeout_s: float = 30.0,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        client=None,  # injectable object with .post(...) for tests
    ):
        if not base_url:
            raise GenerationError("OpenAICompatProvider requires OPENAI_COMPAT_BASE_URL")
        self.model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._default_max_tokens = max_tokens
        self._default_temperature = temperature
        self._client = client  # None → use module-level httpx.post

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        max_tokens = self._default_max_tokens if max_tokens is None else max_tokens
        temperature = self._default_temperature if temperature is None else temperature

        url = f"{self._base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            post = self._client.post if self._client is not None else httpx.post
            resp = post(url, json=payload, headers=headers, timeout=self._timeout_s)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # transport / HTTP / decode → unrecoverable
            logger.warning("OpenAI-compat chat/completions failed: %s", exc)
            raise GenerationError(f"OpenAI-compatible generation failed: {exc}") from exc

        choices = (data or {}).get("choices") or []
        if not choices:
            return ""  # no candidate → treat as refusal, not an error
        return (choices[0].get("message", {}).get("content") or "").strip()

    def generate_stream(
        self,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[str]:
        """
        Story 6 — stream via the OpenAI-compatible streaming mode (stream=true).
        The server emits SSE lines ("data: {json}\\n"); each JSON carries a
        choices[0].delta.content fragment. We forward the non-empty fragments and
        stop on the "[DONE]" sentinel. Any transport/HTTP error raises
        GenerationError — the endpoint falls back to generate().

        Uses the injected client's .stream(...) when present (tests), else
        httpx.stream. The stream is a context manager yielding a response whose
        .iter_lines() produces the SSE lines.
        """
        max_tokens = self._default_max_tokens if max_tokens is None else max_tokens
        temperature = self._default_temperature if temperature is None else temperature

        url = f"{self._base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        stream_fn = (
            self._client.stream
            if self._client is not None and hasattr(self._client, "stream")
            else httpx.stream
        )

        try:
            with stream_fn("POST", url, json=payload, headers=headers, timeout=self._timeout_s) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    # httpx.iter_lines() yields str; tolerate bytes from a fake client.
                    if isinstance(line, bytes):
                        line = line.decode("utf-8")
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        obj = json.loads(data_str)
                    except ValueError:
                        continue  # keep-alive / malformed line → skip
                    choices = (obj or {}).get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    fragment = delta.get("content")
                    if fragment:
                        yield fragment
        except Exception as exc:  # transport / HTTP / decode → unrecoverable
            logger.warning("OpenAI-compat streaming chat/completions failed: %s", exc)
            raise GenerationError(f"OpenAI-compatible streaming failed: {exc}") from exc
