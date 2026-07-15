"""
StubProvider — the deterministic, offline generation backend used in ALL tests and
the default GEN_PROVIDER. It never calls a live API, so CI is fast and hermetic.

Controllable per-test (needed for Stories 2-3):
  * default: returns a fixed string ("This is a stub answer.").
  * set_response(text) / .response = text → change what generate() returns, e.g. to
    simulate a specific citation pattern or a refusal ("" ).
  * set_error(exc) / .error = exc → make generate() RAISE, to exercise the endpoint's
    GenerationError handling (Story 0 honest error shape).
"""
from __future__ import annotations

import logging
from typing import Iterator

logger = logging.getLogger(__name__)

DEFAULT_STUB_RESPONSE = "This is a stub answer."


class StubProvider:
    provider_name = "stub"

    def __init__(self, response: str = DEFAULT_STUB_RESPONSE, model_name: str = "stub-1"):
        self.model_name = model_name
        self.response = response
        self.error: Exception | None = None
        self.calls: list[str] = []  # prompts seen, for test introspection

    def set_response(self, response: str) -> None:
        self.response = response

    def set_error(self, error: Exception | None) -> None:
        self.error = error

    def generate(self, prompt: str, *, max_tokens: int = 1024, temperature: float = 0.3) -> str:
        self.calls.append(prompt)
        if self.error is not None:
            raise self.error
        return self.response

    def generate_stream(
        self, prompt: str, *, max_tokens: int = 1024, temperature: float = 0.3
    ) -> Iterator[str]:
        """
        Story 6 — simulate streaming by yielding the configured response one word
        at a time. Every word except the last carries a trailing space, so
        "".join(generate_stream(...)) == generate(...) exactly (the endpoint just
        concatenates chunks into full_answer).

        Respects .error (raises, to exercise the endpoint's stream→generate
        fallback) and an empty response (yields nothing, i.e. a refusal).
        """
        self.calls.append(prompt)
        if self.error is not None:
            raise self.error
        words = self.response.split()
        for i, word in enumerate(words):
            yield word if i == len(words) - 1 else word + " "
