"""
Pipeline Stage 5 — Embedder.

Turns a document's chunks into vectors and streams them to a `sink`, tracking
progress on an EmbeddingJob so the stage is auditable and RESUMABLE.

Design guarantees
-----------------
* Streaming / constant memory: each finished batch of EmbeddedChunks is handed to
  `sink` and then dropped. embed_document never accumulates all vectors in memory;
  it returns the EmbeddingJob, not a list of vectors. (Part 2's sink is a Chroma
  batch-upsert; tests pass a list collector.)

* Deterministic batch boundaries: chunks are ordered by chunk_index before batching,
  so batch N always contains the same chunks across runs. This is what makes
  batch-level resume correct — checkpoint_batch=k means "batches 0..k are done".

* Identity pass-through: EmbeddedChunk copies chunk_id / content_hash / doc_version
  from its Chunk (see EmbeddedChunk.__post_init__). The embedder never mints identity.

* Never misalign: the provider returns one entry per text (None on per-entry
  failure); failed entries are recorded in job.failed_chunk_ids and NEVER sent to the
  sink. The sink only ever receives successfully-embedded chunks.

Size tiers (one code path; they differ ONLY in checkpoint writes + logging):
  SMALL  (<= EMBED_SMALL_MAX)  — no per-batch DB writes; persist once at the end.
  MEDIUM (in between)          — persist completed_chunks + failed tracking per batch.
  LARGE  (>= EMBED_LARGE_MIN)  — additionally advance+commit checkpoint_batch after
                                 each successful batch, plus a per-batch INFO log.

Inter-batch delay (EMBED_BATCH_DELAY_S) is applied ONLY for the hf_api provider
(rate-limit friendliness); the local in-process provider never sleeps.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime

from sqlalchemy.orm import object_session

from app.config import get_settings
from app.models.chunk import ChunkRecord
from app.models.job import EmbeddingJob, JobStatus
from app.pipeline.types import Chunk, EmbeddedChunk

logger = logging.getLogger(__name__)

SMALL = "SMALL"
MEDIUM = "MEDIUM"
LARGE = "LARGE"


# ── Tiering + batching ────────────────────────────────────────────────────────

def _tier(n_chunks: int, settings=None) -> str:
    s = settings or get_settings()
    if n_chunks <= s.embed_small_max:
        return SMALL
    if n_chunks >= s.embed_large_min:
        return LARGE
    return MEDIUM


def _batches(chunks: list[Chunk], batch_size: int) -> list[tuple[int, list[Chunk]]]:
    """Split ordered chunks into (batch_index, batch) pairs. Deterministic."""
    out: list[tuple[int, list[Chunk]]] = []
    for i in range(0, len(chunks), batch_size):
        out.append((i // batch_size, chunks[i:i + batch_size]))
    return out


def _persist(job: EmbeddingJob) -> None:
    """Commit the job if it is bound to a session; no-op for a transient job."""
    sess = object_session(job)
    if sess is not None:
        sess.commit()


def _finalize_status(job: EmbeddingJob) -> None:
    if job.completed_chunks == 0 and job.total_chunks > 0:
        job.status = JobStatus.failed
    elif job.failed_chunk_ids:
        job.status = JobStatus.completed_with_failures
    else:
        job.status = JobStatus.completed
    job.finished_at = datetime.utcnow()
    _persist(job)


def _init_job(job: EmbeddingJob, provider, total: int, fresh: bool) -> None:
    """Stamp provider/model/normalize audit fields; reset counters on a fresh run."""
    job.total_chunks = total
    job.model_name = provider.model_name
    job.provider_name = provider.provider_name
    job.normalize = bool(provider.normalize)
    if fresh:
        job.completed_chunks = 0
        job.failed_chunk_ids = []
        job.checkpoint_batch = -1
        job.finished_at = None
        job.started_at = datetime.utcnow()
    if job.failed_chunk_ids is None:
        job.failed_chunk_ids = []
    if job.started_at is None:
        job.started_at = datetime.utcnow()
    job.status = JobStatus.in_progress
    _persist(job)


# ── Core streaming loop (shared by fresh embed + resume) ──────────────────────

def _run(chunks: list[Chunk], doc_id: str, provider, sink, job: EmbeddingJob,
         start_batch: int) -> EmbeddingJob:
    """
    Embed `chunks` (already ordered) starting at batch `start_batch`, streaming each
    finished batch to `sink`. Shared by embed_document (start_batch=0) and
    resume_embedding (start_batch=checkpoint+1).
    """
    s = get_settings()
    batch_size = s.embed_batch_size
    tier = _tier(job.total_chunks, s)
    is_hf = provider.provider_name == "hf_api"
    delay = s.embed_batch_delay_s if is_hf else 0.0

    batches = _batches(chunks, batch_size)
    for batch_index, batch in batches:
        if batch_index < start_batch:
            continue  # already done in a prior run — resume skip

        texts = [c.text for c in batch]
        vectors = provider.embed(texts)
        if len(vectors) != len(batch):
            # Defensive: provider contract says aligned length, but never misalign.
            logger.error("provider returned %d vectors for %d texts — failing batch %d",
                         len(vectors), len(batch), batch_index)
            vectors = [None] * len(batch)

        embedded: list[EmbeddedChunk] = []
        batch_failed: list[str] = []
        for c, vec in zip(batch, vectors):
            if vec is None:
                batch_failed.append(c.chunk_id)
            else:
                embedded.append(EmbeddedChunk(chunk=c, embedding=vec))

        # Stream successful embeddings to the sink and drop them (constant memory).
        # If the sink raises, we do NOT advance the checkpoint for this batch — a
        # subsequent resume will reprocess it. (Checkpoint is written only after a
        # successful sink below.)
        sink(embedded)

        job.completed_chunks = (job.completed_chunks or 0) + len(embedded)
        if batch_failed:
            job.failed_chunk_ids = list(job.failed_chunk_ids or []) + batch_failed

        if tier == LARGE:
            job.checkpoint_batch = batch_index
            logger.info(
                "embed doc=%s batch %d/%d done: %d ok, %d failed (tier=LARGE)",
                doc_id, batch_index, len(batches) - 1, len(embedded), len(batch_failed),
            )
            _persist(job)
        elif tier == MEDIUM:
            _persist(job)  # persist progress + failures, but no checkpoint_batch
        # SMALL: no per-batch persistence; committed once in _finalize_status.

        if is_hf and delay and batch_index < len(batches) - 1:
            time.sleep(delay)

    _finalize_status(job)
    return job


# ── Public API ────────────────────────────────────────────────────────────────

def embed_document(chunks: list[Chunk], doc_id: str, provider, sink,
                   job: EmbeddingJob) -> EmbeddingJob:
    """
    Embed all `chunks` for `doc_id` from scratch, streaming batches to `sink`.
    Returns the EmbeddingJob (never a list of vectors). Chunks are ordered by
    chunk_index for deterministic batch boundaries.
    """
    ordered = sorted(chunks, key=lambda c: c.chunk_index)
    _init_job(job, provider, total=len(ordered), fresh=True)
    return _run(ordered, doc_id, provider, sink, job, start_batch=0)


def resume_embedding(doc_id: str, provider, sink, job: EmbeddingJob) -> EmbeddingJob:
    """
    Resume an interrupted embed job: re-fetch the document's chunks from the DB
    (ordered by chunk_index) and continue from checkpoint_batch + 1. Batches already
    done (index <= checkpoint_batch) are skipped, so their vectors are not recomputed
    and not re-sunk.
    """
    db = object_session(job)
    if db is None:
        raise RuntimeError("resume_embedding requires a DB-bound EmbeddingJob")

    records = (
        db.query(ChunkRecord)
        .filter(ChunkRecord.doc_id == doc_id)
        .order_by(ChunkRecord.chunk_index)
        .all()
    )
    ordered = [r.to_chunk() for r in records]

    _init_job(job, provider, total=len(ordered), fresh=False)
    start_batch = (job.checkpoint_batch if job.checkpoint_batch is not None else -1) + 1
    logger.info("resume embedding doc=%s from batch %d (checkpoint=%s)",
                doc_id, start_batch, job.checkpoint_batch)
    return _run(ordered, doc_id, provider, sink, job, start_batch=start_batch)


def retry_failed_embeddings(doc_id: str, provider, sink, job: EmbeddingJob) -> EmbeddingJob:
    """
    Re-embed ONLY the chunks in job.failed_chunk_ids. Recovered ids are removed from
    failed_chunk_ids, completed_chunks is bumped, and status is recomputed. Chunks are
    re-fetched from the DB (ordered by chunk_index) so texts are authoritative.
    """
    db = object_session(job)
    if db is None:
        raise RuntimeError("retry_failed_embeddings requires a DB-bound EmbeddingJob")

    failed = list(job.failed_chunk_ids or [])
    if not failed:
        logger.info("retry: doc=%s has no failed chunks — nothing to do", doc_id)
        return job

    records = (
        db.query(ChunkRecord)
        .filter(ChunkRecord.doc_id == doc_id, ChunkRecord.id.in_(failed))
        .order_by(ChunkRecord.chunk_index)
        .all()
    )
    chunks = [r.to_chunk() for r in records]
    texts = [c.text for c in chunks]
    vectors = provider.embed(texts)
    if len(vectors) != len(chunks):
        vectors = [None] * len(chunks)

    embedded: list[EmbeddedChunk] = []
    recovered: set[str] = set()
    for c, vec in zip(chunks, vectors):
        if vec is not None:
            embedded.append(EmbeddedChunk(chunk=c, embedding=vec))
            recovered.add(c.chunk_id)

    if embedded:
        sink(embedded)
        job.completed_chunks = (job.completed_chunks or 0) + len(embedded)

    job.failed_chunk_ids = [cid for cid in failed if cid not in recovered]
    _finalize_status(job)
    logger.info("retry: doc=%s recovered %d, still failed %d",
                doc_id, len(recovered), len(job.failed_chunk_ids))
    return job
