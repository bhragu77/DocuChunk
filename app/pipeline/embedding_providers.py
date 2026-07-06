"""
Embedding providers — the pluggable backends that turn text into vectors.

IMPORTANT — no model import at module load:
  This module must import WITHOUT `sentence-transformers` installed. LocalSTProvider
  receives an ALREADY-LOADED model object from the worker ctx (loaded once in worker
  on_startup), it never imports or loads a model itself. That keeps the model a
  worker-lifetime singleton and keeps tests fast (they pass a fake model).

Normalization contract:
  For all-MiniLM-L6-v2 with cosine collections, normalize=True → L2-normalize each
  vector to unit length. On unit vectors cosine similarity == dot product, so the
  stored document vectors and Phase 8 query vectors are directly comparable. The
  provider applies normalization itself (client-side) so it is observable and
  testable independent of the model.

Query embedding — single entrypoint:
  embed_query(text, provider) routes through provider.embed() with the SAME model and
  SAME normalization as document embedding. Phase 8 MUST call embed_query and MUST NOT
  reimplement query embedding, or query and document vectors could silently diverge.
"""
from __future__ import annotations

import logging
import time
from typing import Protocol, runtime_checkable

import numpy as np

from app.config import get_settings

logger = logging.getLogger(__name__)

EMBED_DIM = 384  # all-MiniLM-L6-v2 dimensionality


# ── Normalization ─────────────────────────────────────────────────────────────

def _l2_normalize(vec) -> list[float]:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0 or not np.isfinite(norm):
        return arr.astype(np.float32).tolist()
    return (arr / norm).astype(np.float32).tolist()


def _rows_to_vectors(raw, expected_n: int, normalize: bool) -> list[list[float]] | None:
    """
    Turn a provider's raw output into a list of `expected_n` python-float vectors,
    or None if the shape can't be aligned (length mismatch → caller marks all None).
    """
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] != expected_n:
        logger.error("embedding length mismatch: expected %d rows, got %s", expected_n, arr.shape)
        return None
    if normalize:
        return [_l2_normalize(row) for row in arr]
    return [row.astype(np.float32).tolist() for row in arr]


# ── Protocol ──────────────────────────────────────────────────────────────────

@runtime_checkable
class EmbeddingProvider(Protocol):
    """
    A provider turns a batch of texts into a batch of vectors.

    Contract:
      * embed() returns one entry per input text, positionally aligned.
      * A per-entry None marks that one text failed; the rest are still valid.
      * embed() NEVER raises for a whole batch — on total failure it returns
        [None] * len(texts). Callers rely on this to keep vectors aligned to chunks.
    """
    normalize: bool
    model_name: str
    provider_name: str

    def embed(self, texts: list[str]) -> list[list[float] | None]: ...


# ── LocalSTProvider (default) ─────────────────────────────────────────────────

class LocalSTProvider:
    """
    In-process sentence-transformers provider. The model is injected (from worker
    ctx), never loaded here. normalize is applied client-side so behavior is
    identical regardless of the model's own settings.
    """
    provider_name = "local"

    def __init__(self, model, model_name: str, normalize: bool = True):
        self._model = model
        self.model_name = model_name
        self.normalize = normalize

    def embed(self, texts: list[str]) -> list[list[float] | None]:
        if not texts:
            return []
        try:
            raw = self._model.encode(texts)
        except Exception:
            # Never raise for a whole batch — mark every entry failed.
            logger.exception("LocalSTProvider.encode failed for a batch of %d", len(texts))
            return [None] * len(texts)

        vectors = _rows_to_vectors(raw, expected_n=len(texts), normalize=self.normalize)
        if vectors is None:
            return [None] * len(texts)  # never misalign vectors to the wrong chunks
        return list(vectors)


# ── HFAPIProvider ─────────────────────────────────────────────────────────────

def _status_and_retry_after(exc) -> tuple[int | None, float | None]:
    """Best-effort extraction of an HTTP status + Retry-After from an HF error."""
    resp = getattr(exc, "response", None)
    status = getattr(resp, "status_code", None)
    retry_after = None
    headers = getattr(resp, "headers", None)
    if headers is not None and hasattr(headers, "get"):
        ra = headers.get("Retry-After") or headers.get("retry-after")
        if ra is not None:
            try:
                retry_after = float(ra)
            except (TypeError, ValueError):
                retry_after = None
    return status, retry_after


def _is_network_error(exc) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    name = type(exc).__name__.lower()
    return "timeout" in name or "connection" in name


class HFAPIProvider:
    """
    HuggingFace Inference API provider via huggingface_hub.InferenceClient.

    Uses the client's feature_extraction (NOT hand-built api-inference URLs — that
    domain is dead and the router URL is unstable; the client absorbs routing).
    Retries 429/503/network with exponential backoff, honoring Retry-After. On an
    exhausted retry budget it returns [None] * n rather than raising. Output/input
    length mismatch → all None (never misalign). normalize applied client-side.
    """
    provider_name = "hf_api"

    def __init__(
        self,
        model_name: str,
        token: str | None = None,
        normalize: bool = True,
        client=None,
        base_delay: float = 1.0,
        max_retries: int | None = None,
    ):
        self.model_name = model_name
        self.normalize = normalize
        self.base_delay = base_delay
        s = get_settings()
        self.max_retries = s.embed_max_retries if max_retries is None else max_retries
        if client is not None:
            self._client = client
        else:
            from huggingface_hub import InferenceClient
            self._client = InferenceClient(model=model_name, token=token, provider="hf-inference")

    def _call_with_retry(self, texts: list[str]):
        attempt = 0
        while True:
            try:
                return self._client.feature_extraction(texts)
            except Exception as exc:
                status, retry_after = _status_and_retry_after(exc)
                retryable = status in (429, 503) or _is_network_error(exc)
                if not retryable or attempt >= self.max_retries:
                    logger.warning(
                        "HF embed failed (attempt %d/%d, status=%s): %s",
                        attempt, self.max_retries, status, exc,
                    )
                    return None
                delay = retry_after if retry_after is not None else self.base_delay * (2 ** attempt)
                logger.info("HF embed retry in %.2fs (attempt %d, status=%s)", delay, attempt, status)
                time.sleep(delay)
                attempt += 1

    def embed(self, texts: list[str]) -> list[list[float] | None]:
        if not texts:
            return []
        raw = self._call_with_retry(texts)
        if raw is None:
            return [None] * len(texts)
        vectors = _rows_to_vectors(raw, expected_n=len(texts), normalize=self.normalize)
        if vectors is None:
            return [None] * len(texts)
        return list(vectors)


# ── Factory + query entrypoint ────────────────────────────────────────────────

def build_provider(ctx: dict | None):
    """
    Build the configured provider from EMBED_PROVIDER + ctx.

    local  → LocalSTProvider using the model in ctx["embed_model"] (loaded once in
             worker on_startup). If no model is present (e.g. tests, or a worker that
             didn't load one), returns None and the caller skips the embed stage.
    hf_api → HFAPIProvider (no local model needed).
    """
    s = get_settings()
    if s.embed_provider == "hf_api":
        return HFAPIProvider(
            model_name=s.hf_embed_model,
            token=(s.hf_api_token or None),
            normalize=True,
        )
    # local (default)
    model = (ctx or {}).get("embed_model")
    if model is None:
        logger.warning(
            "EMBED_PROVIDER=local but ctx has no 'embed_model' — embedding will be skipped"
        )
        return None
    return LocalSTProvider(model=model, model_name=s.hf_embed_model, normalize=True)


def embed_query(text: str, provider) -> list[float] | None:
    """
    THE single query-embedding entrypoint.

    Routes through provider.embed([text]) so a query vector is produced with the
    SAME model and SAME normalization as the stored document vectors. Phase 8 MUST
    call this and MUST NOT build query embeddings any other way, or query/document
    vectors could diverge in model or normalization and break cosine ranking.
    """
    out = provider.embed([text])
    return out[0] if out else None


# ── Web-process query embedder (app.state.embed_fn) ───────────────────────────

_query_provider = None  # lazily-built, process-lifetime singleton


def _get_query_provider():
    """
    Build (once) the provider used to embed SEARCH QUERIES in the web process.

    The worker loads the ST model at startup, but the web process does not — so for
    EMBED_PROVIDER=local we load the model here on FIRST query (lazy: app startup and
    tests that never hit the search endpoints pay nothing). For hf_api no local model
    is needed. Cached module-level so the model loads at most once per web process.
    """
    global _query_provider
    if _query_provider is not None:
        return _query_provider
    s = get_settings()
    if s.embed_provider == "hf_api":
        _query_provider = HFAPIProvider(
            model_name=s.hf_embed_model, token=(s.hf_api_token or None), normalize=True
        )
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(s.hf_embed_model)
        _query_provider = LocalSTProvider(model=model, model_name=s.hf_embed_model, normalize=True)
    logger.info("Query embedder ready: provider=%s model=%s", _query_provider.provider_name, s.hf_embed_model)
    return _query_provider


def embed_single(text: str) -> list[float]:
    """
    Callable[[str], list[float]] for app.state.embed_fn — used by the /search and
    /generate routers. Routes through the SAME embed_query path (same model +
    normalization as stored document vectors).
    """
    vec = embed_query(text, _get_query_provider())
    if vec is None:
        raise RuntimeError("query embedding failed (provider returned no vector)")
    return vec
