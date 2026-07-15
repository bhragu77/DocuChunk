"""
GenerationProvider protocol + the shared error type.

Contract (mirrors EmbeddingProvider in embedding_providers.py):
  * generate() returns the generated text.
  * An EMPTY STRING is a valid result — it means the model refused / produced
    nothing usable. It is NOT an error and must not raise.
  * Any UNRECOVERABLE failure (timeout, transport error, bad API key at call time,
    SDK not installed) raises GenerationError. The caller (the /generate/answer
    endpoint) catches it and returns Story 0's honest error shape, never a 500.

Concrete providers bind their per-call defaults (max_tokens / temperature / timeout)
from config at construction time via the factory, so `provider.generate(prompt)`
already uses the configured values. The keyword args let callers override per call.
"""
from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable


class GenerationError(Exception):
    """Raised by a provider on an unrecoverable generation failure (never for a refusal)."""


@runtime_checkable
class GenerationProvider(Protocol):
    """A provider turns a single prompt into generated text."""

    model_name: str
    provider_name: str

    def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> str:
        """
        Generate text for `prompt`. Returns the completion (possibly "" on refusal).
        Raises GenerationError on an unrecoverable error.
        """
        ...

    def generate_stream(
        self,
        prompt: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> Iterator[str]:
        """
        Story 6 — stream the completion as it arrives, yielding text chunks.

        Yields the answer in incremental pieces (a token, a word, a model-chunk)
        so the /generate/answer SSE endpoint can forward each piece to the client
        as it is produced. Concatenating every yielded chunk reproduces the full
        answer that generate() would have returned.

        An empty stream (yields nothing) is a valid refusal — same contract as
        generate() returning "". Any UNRECOVERABLE failure raises GenerationError;
        the streaming endpoint falls back to generate() on such a failure so a
        provider without real streaming still works (one big chunk).
        """
        ...
