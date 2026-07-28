"""
Story 7 — the answer cache.

A repeated (user, doc, doc_version, query) has a DETERMINISTIC answer, so the
whole chain — retrieval → generation → validation → groundedness — can be skipped
and the complete stored response returned in <10ms. The saving is dominated by
LATENCY (5-8s → sub-10ms) as much as by eliminated LLM cost.

Correctness comes from the KEY, not from purge logic:

    cache_key = sha256(f"{user_id}:{doc_id}:{doc_version}:{normalize(query)}")

`doc_version` (Document.version) is bumped on re-upload, which CHANGES the key,
which busts the cache automatically — no stale answers from an old document
version, ever. TTL is only a safety net for edge cases.

Backends (CACHE_BACKEND): "none" (disabled — pre-Story-7 behavior),
"memory" (in-process LRU, dev/tests), "redis" (shared, survives restarts).
The stored value is the COMPLETE response payload (GenerateAnswerResponse
.model_dump()): answer, citations, confidence, signals — everything.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import OrderedDict
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# A cached answer is just the full response payload as a plain JSON-able dict.
CachedAnswer = dict


def normalize_query(query: str) -> str:
    """Canonicalize a query for the cache key: strip + lowercase (good enough)."""
    return query.strip().lower()


def compute_cache_key(
    user_id: str, doc_id: str, doc_version: int, query: str, model: str | None = None
) -> str:
    """Deterministic, version-aware key. Any input change (incl. doc_version on a
    re-upload, or query case after normalization) yields a different key.

    `model` (the chat picker's tier: "gemini" | "offline") is appended ONLY when
    provided, so the two models never serve each other's cached answers. Omitting
    it (the default path / pre-picker callers) yields the exact pre-existing key —
    no cache migration, and existing entries stay valid."""
    raw = f"{user_id}:{doc_id}:{doc_version}:{normalize_query(query)}"
    if model:
        raw += f":{model}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@runtime_checkable
class AnswerCache(Protocol):
    def get(self, key: str) -> CachedAnswer | None:
        """Return the cached payload, or None on miss/expiry."""
        ...

    def set(self, key: str, answer: CachedAnswer, ttl: int) -> None:
        """Store the payload under key with a TTL in seconds (<=0 → no expiry)."""
        ...


class InMemoryAnswerCache:
    """
    Dict-backed cache with per-entry TTL and LRU eviction. For local dev / tests
    without Redis. Bounded by max_entries — the least-recently-used entry is
    evicted on overflow. Not shared across processes and lost on restart.
    """

    def __init__(self, max_entries: int = 1000):
        self._max = max(1, max_entries)
        # key -> (expires_at | None, payload); OrderedDict tracks LRU order.
        self._store: "OrderedDict[str, tuple[float | None, CachedAnswer]]" = OrderedDict()

    def get(self, key: str) -> CachedAnswer | None:
        item = self._store.get(key)
        if item is None:
            return None
        expires_at, value = item
        if expires_at is not None and time.time() > expires_at:
            del self._store[key]  # lazily drop expired entries on read
            return None
        self._store.move_to_end(key)  # mark most-recently-used
        return value

    def set(self, key: str, answer: CachedAnswer, ttl: int) -> None:
        expires_at = time.time() + ttl if ttl and ttl > 0 else None
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (expires_at, answer)
        while len(self._store) > self._max:
            self._store.popitem(last=False)  # evict the oldest / least-recently-used


class RedisAnswerCache:
    """
    Redis-backed cache: the complete response JSON under a namespaced key with a
    native Redis TTL (EX). Uses a synchronous redis client — a cache GET is
    sub-millisecond, so it doesn't meaningfully block the async endpoint.
    """

    def __init__(self, client, prefix: str = "docuchunk:answer:"):
        self._client = client
        self._prefix = prefix

    def get(self, key: str) -> CachedAnswer | None:
        try:
            raw = self._client.get(self._prefix + key)
        except Exception as exc:  # a cache outage must never break the request
            logger.warning("RedisAnswerCache.get failed (%s) — treating as miss", exc)
            return None
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return None

    def set(self, key: str, answer: CachedAnswer, ttl: int) -> None:
        try:
            data = json.dumps(answer)
            if ttl and ttl > 0:
                self._client.set(self._prefix + key, data, ex=ttl)
            else:
                self._client.set(self._prefix + key, data)
        except Exception as exc:  # a failed cache write is non-fatal — just skip it
            logger.warning("RedisAnswerCache.set failed (%s) — answer not cached", exc)


def build_cache(settings) -> AnswerCache | None:
    """Construct the answer cache from CACHE_BACKEND. None = caching disabled."""
    backend = (settings.cache_backend or "none").lower()

    if backend == "none":
        return None

    if backend == "memory":
        logger.info("Answer cache: in-memory LRU (max_entries=%d)", settings.cache_max_memory_entries)
        return InMemoryAnswerCache(max_entries=settings.cache_max_memory_entries)

    if backend == "redis":
        try:
            import redis
        except ImportError:
            logger.warning("CACHE_BACKEND=redis but the redis package is not installed — caching disabled")
            return None
        client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            db=settings.redis_db,
            decode_responses=True,
        )
        logger.info("Answer cache: Redis at %s:%s db=%s", settings.redis_host, settings.redis_port, settings.redis_db)
        return RedisAnswerCache(client)

    logger.warning("Unknown CACHE_BACKEND=%r — caching disabled", settings.cache_backend)
    return None
