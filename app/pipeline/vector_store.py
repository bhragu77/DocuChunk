"""
Pipeline Stage 7 — Vector Store (Part 2)

Two responsibilities, both keyed on the DETERMINISTIC chunk_id from Part 0:

  1. batch_upsert_sink — the real streaming sink that replaces the orchestrator's
     _noop_sink. Each batch of EmbeddedChunks the embedder streams out is UPSERTED
     (not add()) into the document owner's per-user Chroma collection. Because the
     id is content-addressable (sha256 of doc_id + char span), re-running the embed
     stage overwrites the same rows in place — idempotent by construction, so a
     re-sunk batch after a crash never duplicates a vector.

  2. reconcile_document — the incremental-index contract, computed BEFORE embedding
     so unchanged chunks are never re-embedded. It diffs the chunk_ids currently
     stored for a doc_id against the freshly-chunked set:
         to_delete  = stored ids absent from the new set          (ghosts → tombstoned)
         to_upsert  = new ids that are new OR whose content_hash changed
         unchanged  = ids in both with an identical content_hash   (SKIP embedding)
     Deletions are applied immediately; the caller embeds ONLY to_upsert. First
     upload falls out naturally (nothing stored → to_delete empty, to_upsert = all).

Every collection is obtained through app.core.chroma.get_or_create_collection — the
single cosine-space construction site — so document vectors and Phase 8 query vectors
share one distance metric. There is no second construction path here.

Chroma metadata must be flat scalars (str/int/float/bool); the sink flattens each
chunk's provenance into exactly the fields retrieval.py reads (doc_id, source,
page_number) plus the identity fields the reconcile contract needs (content_hash,
doc_version, chunk_index, strategy).

=============================================================================
PHASE 7 DEPENDENCY NOTE FOR THE BM25 INDEX
=============================================================================

retrieval.py (Phase 8) depends on a BM25IndexProtocol instance stored at
app.state.bm25.  Phase 7 MUST satisfy this contract.

Required behaviour
------------------
  * One index per user (keyed by user_id), mirroring ChromaDB's per-user
    collection naming convention ("user_{user_id}").
  * Filterable by doc_id so a user can search within a single document.
  * query() returns (chunk_id, score) pairs sorted descending by score.
  * upsert() is called every time new chunks are stored (after embedding).
  * delete() is called whenever a document is deleted from ChromaDB.

Storage options (Phase 7 decides — either is fine):
  Option A — pickle: serialize a rank_bm25.BM25Okapi instance to
    {CHROMA_PERSIST_DIR}/bm25_{user_id}.pkl
    Simple, fast. Rebuilding requires loading and re-pickling on every upsert.
    Fine for MVP; O(n) rebuild on every write.

  Option B — per-user SQLite inverted index:
    {CHROMA_PERSIST_DIR}/bm25_{user_id}.db  with tables:
      chunks(chunk_id TEXT PK, doc_id TEXT, tokens TEXT)
      term_freq(term TEXT, chunk_id TEXT, tf REAL)
      doc_freq(term TEXT, df INTEGER)
    More complex, but supports incremental updates without a full rebuild.
    Recommended if the corpus is large (>10k chunks per user).

Implementation checklist for Phase 7
-------------------------------------
  [ ] Implement BM25IndexProtocol (class below) for the chosen storage option.
  [ ] Call upsert() for each chunk inside vector_store.upsert_chunks().
  [ ] Call delete() inside vector_store.delete_by_doc() (called from docs router).
  [ ] On startup (main.py lifespan), instantiate the BM25 store and set:
        app.state.bm25 = YourBM25Index(persist_dir=settings.chroma_persist_dir)
  [ ] Wire embed_fn similarly:
        app.state.embed_fn = embedder.embed_single  # Callable[[str], list[float]]

=============================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from app.config import get_settings
from app.core.chroma import get_or_create_collection
from app.pipeline.types import Chunk, EmbeddedChunk

logger = logging.getLogger(__name__)


# ── Collection access (single construction site) ──────────────────────────────

def collection_name(user_id: str) -> str:
    """Per-user collection name — mirrors the docs router / retrieval convention."""
    return f"user_{user_id}"


def get_collection(client, user_id: str):
    """
    Return the owner's per-user collection via the ONE cosine-space construction
    site (get_or_create_collection). No second construction path lives here.
    """
    return get_or_create_collection(client, collection_name(user_id))


# ── Metadata flattening ───────────────────────────────────────────────────────

def _metadata_for(ec: EmbeddedChunk) -> dict:
    """
    Flatten an EmbeddedChunk's provenance into Chroma-legal scalar metadata.

    content_hash + doc_version are the keys the reconcile contract diffs on;
    doc_id + source + page_number are what retrieval.py reads back.
    """
    c = ec.chunk
    m = c.metadata or {}
    return {
        "doc_id": c.doc_id,
        "page_number": c.page_number,
        "chunk_index": c.chunk_index,
        "content_hash": ec.content_hash,
        "doc_version": ec.doc_version,
        "source": m.get("source", ""),
        "strategy": m.get("strategy", ""),
    }


# ── Batch-upsert sink (replaces _noop_sink) ───────────────────────────────────

def batch_upsert_sink(
    collection,
    bm25=None,
    user_id: str | None = None,
    max_batch: int | None = None,
) -> Callable[[list[EmbeddedChunk]], None]:
    """
    Build the streaming sink the embedder calls per batch (Part 1 contract).

    Each call UPSERTS the batch into `collection` keyed on the deterministic
    chunk_id, so re-sinking a batch overwrites the same ids in place (idempotent).
    A batch larger than CHROMA_UPSERT_MAX is split into multiple upsert calls to
    respect Chroma's per-upsert cap.

    BM25 co-write (Phase 8): when a bm25 index + user_id are supplied, the SAME
    batch is mirrored into the user's BM25 index (upsert_many, also keyed on
    chunk_id). This is how the lexical and vector indexes stay in lockstep — a
    chunk is never stored in one but missing from the other. The Chroma write
    happens first; a BM25 write failure is logged but does not fail the batch
    (the vector store is the source of truth; BM25 self-heals on the next re-embed).
    """
    cap = max_batch if max_batch is not None else get_settings().chroma_upsert_max

    def _sink(batch: list[EmbeddedChunk]) -> None:
        if not batch:
            return
        for start in range(0, len(batch), cap):
            sub = batch[start:start + cap]
            collection.upsert(
                ids=[ec.chunk_id for ec in sub],
                embeddings=[list(ec.embedding) for ec in sub],
                documents=[ec.chunk.text for ec in sub],
                metadatas=[_metadata_for(ec) for ec in sub],
            )
        if bm25 is not None and user_id is not None:
            try:
                bm25.upsert_many(
                    [(ec.chunk_id, ec.chunk.text, ec.chunk.doc_id) for ec in batch],
                    user_id,
                )
            except Exception:
                logger.exception("BM25 co-write failed for %d chunks (Chroma write already done)", len(batch))
        logger.debug("upsert sink: wrote %d chunks (cap=%d)", len(batch), cap)

    return _sink


# ── Incremental-index reconciliation ──────────────────────────────────────────

@dataclass
class ReconcileReport:
    """
    Result of diffing the stored index for a doc_id against a freshly-chunked set.
    to_upsert_ids are the ONLY chunks the caller should embed; to_delete_ids have
    already been tombstoned by reconcile_document before this is returned.
    """
    to_upsert_ids: set[str]
    to_delete_ids: set[str]
    unchanged_ids: set[str]

    @property
    def counts(self) -> dict[str, int]:
        return {
            "to_upsert": len(self.to_upsert_ids),
            "to_delete": len(self.to_delete_ids),
            "unchanged": len(self.unchanged_ids),
        }


def reconcile_document(
    collection,
    doc_id: str,
    new_chunks: list[Chunk],
    new_version: int,
    bm25=None,
    user_id: str | None = None,
) -> ReconcileReport:
    """
    Compute (and apply the delete half of) the incremental-index diff for `doc_id`.

    Runs BEFORE embedding: it reads only chunk_id + content_hash, so unchanged
    chunks never reach the embedder. Deletions (ghosts: stored ids no longer
    present in new_chunks) are applied immediately; the returned to_upsert set is
    what the caller feeds to the embedder.

    BM25 tombstone sync (Phase 8): ghosts are tombstoned from the user's BM25 index
    too (delete_chunks), not just Chroma. Without this, BM25 would keep the exact
    chunk_ids the reconcile just removed from Chroma and surface dead ids that no
    longer resolve to any stored text — the "ghost" the incremental index exists to
    eliminate. Requires bm25 + user_id; skipped when either is absent (tests / no store).

    First upload falls out naturally: nothing stored → old_ids empty → to_delete
    empty, to_upsert = every new chunk.
    """
    stored = collection.get(where={"doc_id": doc_id}, include=["metadatas"])
    old_ids: dict[str, str] = {
        cid: (meta or {}).get("content_hash", "")
        for cid, meta in zip(stored.get("ids", []), stored.get("metadatas") or [])
    }
    new_ids: dict[str, str] = {c.chunk_id: c.content_hash for c in new_chunks}

    to_delete = {cid for cid in old_ids if cid not in new_ids}
    to_upsert = {
        cid for cid, h in new_ids.items()
        if cid not in old_ids or old_ids[cid] != h
    }
    unchanged = {
        cid for cid, h in new_ids.items()
        if cid in old_ids and old_ids[cid] == h
    }

    if to_delete:
        collection.delete(ids=list(to_delete))
        if bm25 is not None and user_id is not None:
            try:
                bm25.delete_chunks(list(to_delete), user_id)
            except Exception:
                logger.exception("BM25 tombstone sync failed for doc=%s (%d ghosts)", doc_id, len(to_delete))

    report = ReconcileReport(to_upsert, to_delete, unchanged)
    logger.info(
        "reconcile doc=%s v%d: %s (stored=%d new=%d)",
        doc_id, new_version, report.counts, len(old_ids), len(new_ids),
    )
    return report


@runtime_checkable
class BM25IndexProtocol(Protocol):
    """
    Interface that Phase 7's BM25 implementation must satisfy.
    retrieval.py imports and type-checks against this Protocol.
    """

    def query(
        self,
        text: str,
        user_id: str,
        doc_id: str | None = None,
        top_n: int = 50,
    ) -> list[tuple[str, float]]:
        """
        Tokenise text, score the user's indexed chunks, return top_n as
        [(chunk_id, score), ...] sorted by score descending.
        If doc_id is given, restrict to chunks from that document only.
        """
        ...

    def upsert(self, chunk_id: str, text: str, doc_id: str, user_id: str) -> None:
        """Add or update a single chunk in the user's index."""
        ...

    def upsert_many(self, items: list[tuple[str, str, str]], user_id: str) -> None:
        """Batch add/update: items are (chunk_id, text, doc_id). The sink path."""
        ...

    def delete(self, doc_id: str, user_id: str) -> None:
        """Remove all chunks belonging to doc_id from the user's index."""
        ...

    def delete_chunks(self, chunk_ids, user_id: str) -> None:
        """Remove specific chunk_ids (reconcile tombstones)."""
        ...
