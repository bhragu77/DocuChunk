"""
Phase 7/8 — Per-user BM25 lexical index.

WHY THIS EXISTS
===============
Dense (neural) retrieval blurs exact tokens: an all-MiniLM vector for a chunk
containing `88-AZ-0097` or `IP67` sits near other "part number / rating" chunks,
so a query for that exact identifier can rank the wrong chunk first. BM25 is the
opposite — it rewards exact lexical overlap. Phase 8's hybrid search fuses the two
(Reciprocal Rank Fusion) so identifier queries get BM25's precision while
paraphrase queries keep dense recall. This module is the BM25 leg.

CONTRACT (satisfies vector_store.BM25IndexProtocol)
---------------------------------------------------
  query(text, user_id, doc_id=None, top_n) -> [(chunk_id, score), ...]  desc by score
  upsert(chunk_id, text, doc_id, user_id)          — add/update one chunk
  upsert_many([(chunk_id, text, doc_id), ...], user_id)  — batch (the sink path)
  delete(doc_id, user_id)                          — drop a whole document
  delete_chunks(chunk_ids, user_id)                — drop specific chunks (tombstones)

KEYING + SYNC WITH THE VECTOR STORE
-----------------------------------
Chunks are keyed on the SAME deterministic chunk_id Chroma uses (sha256 of
doc_id + char span, Part 0). That is what keeps the two indexes in lockstep:
  * vector_store.batch_upsert_sink upserts each embedded batch into BOTH Chroma and
    here (upsert_many), so a chunk is never in one index but missing from the other.
  * vector_store.reconcile_document tombstones ghost chunk_ids from BOTH — otherwise
    BM25 would accumulate the exact ghosts the reconcile just removed from Chroma and
    keep surfacing dead chunk_ids that no longer resolve to any stored text.
  * the docs router calls delete(doc_id, user_id) on document deletion.

STORAGE
-------
One pickle per user at {persist_dir}/bm25_{user_id}.pkl holding
    {chunk_id: {"tokens": [str, ...], "doc_id": str}}
rank_bm25.BM25Okapi has no incremental add, so a write is load-modify-save and the
BM25Okapi is rebuilt lazily on the next query (cached per user, invalidated by the
file's mtime so a worker write is picked up by the web process). This is the
O(n)-rebuild "Option A" from the vector_store dependency note — fine for the MVP
corpus sizes; swap for an inverted index if a single user exceeds ~10k chunks.
"""
from __future__ import annotations

import logging
import os
import pickle
import re
import threading
from pathlib import Path

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Alphanumeric-run tokenizer. Lowercased so matching is case-insensitive. Identifiers
# survive usefully: `IP67`->`ip67`, `E4521`->`e4521`; `88-AZ-0097` splits into
# `88`,`az`,`0097` — but the QUERY splits identically, so exact-identifier queries
# still get a strong lexical match. This is intentional and standard.
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# chunk_id -> {"tokens": [...], "doc_id": ...}
Corpus = dict


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _safe(user_id: str) -> str:
    """Filesystem-safe per-user filename stem (user_ids are uuids in practice)."""
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(user_id))


class BM25Index:
    """Per-user persisted BM25 index. Thread-safe within a process (a lock guards
    every load-modify-save); cross-process consistency relies on mtime-based cache
    invalidation on the read path."""

    def __init__(self, persist_dir: str | os.PathLike):
        self._dir = Path(persist_dir)
        self._lock = threading.Lock()
        # user_id -> [mtime, corpus, bm25_or_None, ordered_ids_or_None]
        self._cache: dict[str, list] = {}

    # ── Paths / persistence ───────────────────────────────────────────────────

    def _path(self, user_id: str) -> Path:
        return self._dir / f"bm25_{_safe(user_id)}.pkl"

    def _load_corpus(self, user_id: str) -> Corpus:
        p = self._path(user_id)
        if not p.exists():
            return {}
        try:
            with open(p, "rb") as f:
                return pickle.load(f)
        except Exception as exc:  # corrupt/partial file — treat as empty, rebuilt on next write
            logger.warning("BM25: failed to load %s (%s) — treating as empty", p, exc)
            return {}

    def _save_corpus(self, user_id: str, corpus: Corpus) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        p = self._path(user_id)
        tmp = p.with_name(p.name + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(corpus, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, p)  # atomic swap so a reader never sees a half-written file
        self._cache.pop(user_id, None)

    # ── Read-path cache (mtime-keyed so worker writes reach the web process) ───

    def _cached(self, user_id: str) -> list:
        p = self._path(user_id)
        mtime = p.stat().st_mtime if p.exists() else None
        entry = self._cache.get(user_id)
        if entry is not None and entry[0] == mtime:
            return entry
        entry = [mtime, self._load_corpus(user_id), None, None]
        self._cache[user_id] = entry
        return entry

    # ── Writes ─────────────────────────────────────────────────────────────────

    def upsert(self, chunk_id: str, text: str, doc_id: str, user_id: str) -> None:
        self.upsert_many([(chunk_id, text, doc_id)], user_id)

    def upsert_many(self, items: list[tuple[str, str, str]], user_id: str) -> None:
        """items: [(chunk_id, text, doc_id), ...]. Idempotent by chunk_id."""
        if not items:
            return
        with self._lock:
            corpus = self._load_corpus(user_id)
            for chunk_id, text, doc_id in items:
                corpus[chunk_id] = {"tokens": _tokenize(text), "doc_id": doc_id}
            self._save_corpus(user_id, corpus)
        logger.debug("BM25 upsert user=%s +%d chunks", user_id, len(items))

    def delete(self, doc_id: str, user_id: str) -> int:
        """Remove every chunk belonging to doc_id. Returns number removed."""
        with self._lock:
            corpus = self._load_corpus(user_id)
            removed = [cid for cid, rec in corpus.items() if rec.get("doc_id") == doc_id]
            for cid in removed:
                del corpus[cid]
            if removed:
                self._save_corpus(user_id, corpus)
        if removed:
            logger.info("BM25 delete user=%s doc=%s removed=%d", user_id, doc_id, len(removed))
        return len(removed)

    def delete_chunks(self, chunk_ids, user_id: str) -> int:
        """Remove specific chunk_ids (reconcile tombstones). Returns number removed."""
        wanted = set(chunk_ids)
        if not wanted:
            return 0
        with self._lock:
            corpus = self._load_corpus(user_id)
            removed = [cid for cid in wanted if cid in corpus]
            for cid in removed:
                del corpus[cid]
            if removed:
                self._save_corpus(user_id, corpus)
        if removed:
            logger.info("BM25 tombstone user=%s removed=%d chunks", user_id, len(removed))
        return len(removed)

    # ── Query ──────────────────────────────────────────────────────────────────

    def query(self, text: str, user_id: str, doc_id: str | None = None,
              top_n: int = 50) -> list[tuple[str, float]]:
        """
        Score the user's chunks against `text`, return the top_n as
        [(chunk_id, score), ...] sorted by score descending. Only chunks that share
        at least one query token are returned, so a non-matching chunk never receives
        an artificial bm25_rank in RRF. (Overlap — not a positive-score gate — is the
        right filter: on a tiny corpus BM25's IDF goes negative for a term present in
        most/all documents, which would zero out otherwise-valid matches.)
        Restricts to a single document when doc_id is given.
        """
        tokens = _tokenize(text)
        if not tokens:
            return []
        qset = set(tokens)

        with self._lock:
            entry = self._cached(user_id)
            corpus: Corpus = entry[1]
            if not corpus:
                return []

            if doc_id is not None:
                ordered_ids = [c for c, r in corpus.items() if r.get("doc_id") == doc_id]
                if not ordered_ids:
                    return []
                bm25 = BM25Okapi([corpus[c]["tokens"] for c in ordered_ids])
            else:
                if entry[2] is None:  # build + cache the full-corpus BM25 once per mtime
                    entry[3] = list(corpus.keys())
                    entry[2] = BM25Okapi([corpus[c]["tokens"] for c in entry[3]])
                bm25, ordered_ids = entry[2], entry[3]

            scores = bm25.get_scores(tokens)
            # Keep only chunks with actual query-term overlap (independent of IDF sign).
            ranked = sorted(
                (
                    (cid, float(s))
                    for cid, s in zip(ordered_ids, scores)
                    if qset & set(corpus[cid]["tokens"])
                ),
                key=lambda x: x[1],
                reverse=True,
            )

        return ranked[:top_n]

    # ── Introspection (tests / diagnostics) ────────────────────────────────────

    def count(self, user_id: str) -> int:
        with self._lock:
            return len(self._cached(user_id)[1])

    def contains(self, chunk_id: str, user_id: str) -> bool:
        with self._lock:
            return chunk_id in self._cached(user_id)[1]
