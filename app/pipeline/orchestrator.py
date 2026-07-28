"""
Pipeline Orchestrator — runs all stages in sequence for a single document.

GOVERNING RULE (worker migration)
=================================
The pipeline is invoked from an arq worker task (app/workers/tasks.py) that
receives ONLY serializable primitives — a `doc_id` string, never an ORM object.
ALL state is reconstructed by querying the DB inside the task/orchestrator from
that `doc_id`. Nothing about the document is passed in from the caller.

Because of that, `run_pipeline(doc_id)` is IDEMPOTENT and RESUME-AFTER-CRASH safe:
re-running it on the same `doc_id` never duplicates work or corrupts state. Two
mechanisms enforce this (see _run_stages):

  1. Resume-by-status: a document already in `ready` status is a no-op — the whole
     pipeline is done, so we return immediately.
  2. Resume-by-persisted-chunks: if chunk rows already exist for the document's
     CURRENT version, parse/clean/chunk are skipped and we jump to the later
     stages. Chunk writes are a single atomic commit, so this set is always
     complete-or-absent (a crash mid-chunk leaves zero rows, not a partial set).
  3. Delete-then-insert on (re)chunk: before inserting chunk rows we delete any
     existing rows for the doc. chunk_ids are DETERMINISTIC (Part 0), so re-running
     reproduces identical rows; deleting first prevents duplicate-key errors and
     clears stale chunks from a previous version.

Stage coverage:
  [✓] Stage 1 — Parse    (Phase 3)
  [✓] Stage 2 — Clean    (Phase 3)
  [✓] Stage 3 — Chunk    (Phase 4)
  [✓] Stage 4 — Analyse  (Phase 5)
  [ ] Stage 5 — Embed    (Phase 6 — stub; will checkpoint per-batch on the worker)
  [ ] Stage 6 — Store    (Phase 7 — stub)

Learning note — why a separate DB session?
  Each pipeline run creates its own session (via the injected session_factory).
  The web request's session is already closed by the time the worker runs, so the
  worker must never reuse it. The factory is injectable so tests can point the
  orchestrator at a temp DB without monkeypatching globals.
"""
import logging

from app.config import get_settings
from app.database import SessionLocal
from app.observability import span
from app.models.document import Document, DocumentStatus
from app.models.chunk import ChunkRecord
from app.models.job import EmbeddingJob, JobStatus
from app.pipeline.parser import parse
from app.pipeline.cleaner import clean_pages
from app.pipeline.chunker import chunk_document, default_chunker_config
from app.pipeline.analyser import analyse_chunks
from app.pipeline.embedder import embed_document, resume_embedding

logger = logging.getLogger(__name__)


def _noop_sink(embedded) -> None:
    """
    Fallback embed sink used ONLY when no Chroma client is available (e.g. the
    embedder unit tests call run_pipeline with provider but no chroma). It drops the
    batch (constant memory) so the embed stage still runs and advances status. When
    a Chroma client IS passed, the embed stage uses the real batch_upsert_sink and
    the incremental-index reconcile instead — see _embed_stage.
    """
    return None


def run_pipeline(doc_id: str, session_factory=None, provider=None, chroma=None, bm25=None) -> str:
    """
    Entry point for the pipeline. Reconstructs all state from `doc_id` + DB.

    Args:
        doc_id:          the document to process (the ONLY caller-supplied state).
        session_factory: callable returning a DB Session. Defaults to the app's
                         SessionLocal; injected by the worker (ctx) and by tests.
        provider:        the embedding provider (built from worker ctx + config). If
                         None, the embed stage is skipped (e.g. no model loaded / tests).
        chroma:          the shared ChromaDB client (from worker ctx). If provided, the
                         embed stage reconciles + upserts vectors into the owner's
                         per-user collection; if None, embedding runs with a no-op sink
                         and nothing is stored (embedder unit tests).

    Returns:
        The document's final status value (e.g. "ready" / "failed" / "not_found").

    Safe to call repeatedly on the same doc_id — see the module docstring for the
    idempotency / resume guarantees.
    """
    factory = session_factory or SessionLocal
    db = factory()
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if doc is None:
            logger.error("Pipeline: document %s not found in DB", doc_id)
            return "not_found"

        # Resume-by-status: fully-processed documents are a no-op on re-entry.
        if doc.status == DocumentStatus.ready:
            logger.info("Pipeline: doc %s already ready — nothing to do (resume no-op)", doc_id)
            return doc.status.value

        logger.info("Pipeline started for doc_id=%s (%s)", doc_id, doc.original_filename)
        _run_stages(doc, db, provider, chroma, bm25)
        return doc.status.value

    except Exception as exc:
        logger.exception("Pipeline crashed for doc_id=%s", doc_id)
        # Best-effort status update — a second exception here is swallowed
        try:
            doc = db.query(Document).filter(Document.id == doc_id).first()
            if doc:
                doc.status = DocumentStatus.failed
                doc.error_message = f"Pipeline error: {str(exc)[:800]}"
                db.commit()
        except Exception:
            pass
        return "failed"
    finally:
        db.close()


def _run_stages(doc: Document, db, provider=None, chroma=None, bm25=None) -> None:
    """Execute pipeline stages in order, updating status between each."""

    # Resume-by-persisted-chunks: if this document's current version already has
    # chunk rows, a previous run got at least through chunking. Skip parse/clean/
    # chunk and go straight to the embed/store stages. (Atomic chunk commit means
    # this set is always complete, never partial.)
    already_chunked = (
        db.query(ChunkRecord)
        .filter(ChunkRecord.doc_id == doc.id, ChunkRecord.doc_version == doc.version)
        .count()
    )
    if already_chunked > 0:
        logger.info(
            "Pipeline resume: doc=%s already has %d chunks for v%d — skipping parse/clean/chunk",
            doc.id, already_chunked, doc.version,
        )
        _run_embed_and_finish(doc, db, provider, chroma, bm25)
        return

    # ── Stage 1+2: Parse + Clean ─────────────────────────────────────────────
    _set_status(doc, DocumentStatus.parsing, db)

    try:
        with span("parse") as psp:
            raw_pages = parse(doc.file_path, doc.original_filename, doc.mime_type)
            psp.update(pages=len(raw_pages) if raw_pages else 0)
    except Exception as exc:
        _fail(doc, f"Parse error: {exc}", db)
        return

    if not raw_pages:
        _fail(doc, "No text could be extracted from the document. It may be a scanned image.", db)
        return

    with span("clean") as csp:
        cleaned_pages = clean_pages(raw_pages)
        csp.update(cleaned_pages=len(cleaned_pages) if cleaned_pages else 0)

    if not cleaned_pages:
        _fail(doc, "All pages were empty after cleaning.", db)
        return

    # Update page count now that we know it
    doc.page_count = len(raw_pages)
    db.commit()

    logger.info(
        "Parse+Clean done: doc=%s pages=%d cleaned_pages=%d",
        doc.id, len(raw_pages), len(cleaned_pages),
    )

    # ── Stage 3: Chunk ────────────────────────────────────────────────────────
    _set_status(doc, DocumentStatus.chunking, db)

    settings = get_settings()
    strategy = settings.default_chunk_strategy
    config_kwargs = default_chunker_config(strategy)

    try:
        with span("chunk", strategy=strategy) as chsp:
            chunks = chunk_document(
                pages=cleaned_pages,
                doc_id=doc.id,
                strategy=strategy,
                doc_version=doc.version,   # Phase 4 amendment: tag chunks with the doc's version
                **config_kwargs,
            )
            chsp.update(chunks=len(chunks) if chunks else 0)
    except Exception as exc:
        _fail(doc, f"Chunk error: {exc}", db)
        return

    if not chunks:
        _fail(doc, "Chunker produced zero chunks — document may be too short.", db)
        return

    doc.chunk_count = len(chunks)
    doc.chunker_config = {"strategy": strategy, **config_kwargs}

    # Idempotency: clear any chunks from a previous (possibly crashed or older-version)
    # run before inserting. Deterministic chunk_ids (Part 0) mean a re-run reproduces
    # the same rows, so delete-then-insert avoids duplicate-key errors and stale rows.
    deleted = db.query(ChunkRecord).filter(ChunkRecord.doc_id == doc.id).delete()
    if deleted:
        logger.info("Cleared %d pre-existing chunk rows for doc=%s before re-insert", deleted, doc.id)

    chunk_records = [
        ChunkRecord(
            id=c.chunk_id,
            doc_id=c.doc_id,
            chunk_index=c.chunk_index,
            page_number=c.page_number,
            text=c.text,
            char_start=c.char_start,
            char_end=c.char_end,
            token_count=c.token_count,
            content_hash=c.content_hash,
            doc_version=c.doc_version,
            chunk_metadata=c.metadata,
        )
        for c in chunks
    ]
    db.add_all(chunk_records)
    db.commit()
    logger.info(
        "Chunking done: doc=%s strategy=%s chunks=%d",
        doc.id, strategy, len(chunks),
    )

    # ── Stage 4: Analyse ──────────────────────────────────────────────────────
    stats = analyse_chunks(chunks, cleaned_pages)
    logger.info(
        "Analysis: doc=%s total_chunks=%d overlap_pct=%.2f%% warnings=%d",
        doc.id, stats["total_chunks"], stats["overlap_coverage_pct"], len(stats["warnings"]),
    )

    _run_embed_and_finish(doc, db, provider, chroma, bm25)


def _run_embed_and_finish(doc: Document, db, provider=None, chroma=None, bm25=None) -> None:
    """
    Stages 5-6 (embed + store) then mark ready. Split out so the resume path
    (chunks already persisted) can jump straight here without re-parsing.

    NESTED RESUME (two levels, decided here):
      * STAGE-level resume decides "embed or not": if a COMPLETED EmbeddingJob exists
        for the current doc_version, embedding is already done → skip it entirely.
      * BATCH-level resume decides "from which batch": if a job exists that was
        interrupted mid-embed (checkpoint_batch >= 0), call resume_embedding to
        continue from checkpoint_batch + 1 instead of restarting at batch 0.
      * Otherwise start a fresh embed_document from batch 0.

    Document.status handoff:
      completed                → ready
      completed_with_failures  → ready (usable; the EmbeddingJob keeps failed_chunk_ids
                                 inspectable / retryable — see retry_failed_embeddings)
      failed (all chunks None) → failed (halts; not marked ready)
    """
    # ── Stage 5: Embed ────────────────────────────────────────────────────────
    _set_status(doc, DocumentStatus.embedding, db)

    if provider is None:
        # No embedding backend available (e.g. worker didn't load a model / tests).
        logger.warning("No embedding provider — skipping embed stage for doc=%s", doc.id)
    else:
        with span("embed") as esp:
            job = _embed_stage(doc, db, provider, chroma, bm25)
            # chunk count + batch count (the spec's embed fields). Chroma upsert is
            # fused into the embed sink, so batches derive from the configured size.
            batch_size = get_settings().embed_batch_size or 1
            chunk_count = doc.chunk_count or 0
            esp.update(
                chunk_count=chunk_count,
                batch_count=(chunk_count + batch_size - 1) // batch_size,
            )
        if job is not None and job.status == JobStatus.failed:
            _fail(doc, "Embedding failed for all chunks.", db)
            return

    # ── Stage 6: Store ────────────────────────────────────────────────────────
    # Storage is fused into the embed stage: batches stream straight from the
    # embedder into batch_upsert_sink (Chroma upsert-by-chunk_id). When no Chroma
    # client is available the embed stage uses _noop_sink and nothing is stored.

    # Mark ready (also for completed_with_failures — partial success is still usable).
    with span("store"):
        doc.status = DocumentStatus.ready
        doc.error_message = None
        db.commit()
    logger.info("Pipeline complete for doc_id=%s", doc.id)


def _embed_stage(doc: Document, db, provider, chroma=None, bm25=None) -> EmbeddingJob | None:
    """
    Run (or resume, or skip) the embed job for this document version and return it.
    See _run_embed_and_finish for the two-level resume contract.

    INTEGRATION WITH THE VECTOR STORE (Part 2):
      When a Chroma client is provided, this stage upserts vectors into the owner's
      per-user collection and enforces the incremental-index contract:
        * Fresh embed → reconcile_document(collection, doc_id, chunks, version)
          tombstones ghost chunk_ids and returns the to_upsert subset. ONLY those
          chunks are embedded (unchanged chunks are already correctly stored, so we
          never pay to re-embed them). An identical re-upload yields an empty
          to_upsert set → embed_document([]) makes a completed job with zero provider
          calls.
        * Resume mid-embed → resume_embedding streams into the SAME upsert sink;
          re-sinking a batch is harmless because upsert is idempotent-by-chunk_id.
      When chroma is None the sink is _noop_sink and no reconcile runs (embedder unit
      tests) — the embed stage still advances status but stores nothing.
    """
    job = (
        db.query(EmbeddingJob)
        .filter(EmbeddingJob.doc_id == doc.id, EmbeddingJob.doc_version == doc.version)
        .order_by(EmbeddingJob.created_at.desc())
        .first()
    )

    # Build the real upsert sink only when a Chroma client is available.
    collection = None
    sink = _noop_sink
    if chroma is not None:
        from app.pipeline.vector_store import get_collection, batch_upsert_sink
        collection = get_collection(chroma, doc.user_id)
        # BM25 co-write is keyed on the same chunk_id; passing bm25+user_id makes the
        # sink mirror each embedded batch into the user's lexical index in lockstep.
        sink = batch_upsert_sink(collection, bm25=bm25, user_id=doc.user_id)

    # STAGE-level: already fully embedded for this version → skip.
    if job is not None and job.status == JobStatus.completed:
        logger.info("Embed stage: doc=%s already completed for v%d — skip", doc.id, doc.version)
        return job

    # BATCH-level: interrupted mid-embed → resume from checkpoint_batch + 1 into the
    # SAME sink. Reconcile already ran (and applied deletes) on the first attempt;
    # upsert idempotency makes re-sinking the in-flight batch harmless.
    if job is not None and job.checkpoint_batch is not None and job.checkpoint_batch >= 0:
        logger.info("Embed stage: doc=%s resuming from batch %d", doc.id, job.checkpoint_batch + 1)
        return resume_embedding(doc.id, provider, sink, job)

    # Fresh embed. Reuse an existing (pending/failed, no-checkpoint) job row if present.
    if job is None:
        job = EmbeddingJob(doc_id=doc.id, doc_version=doc.version, status=JobStatus.pending)
        db.add(job)
        db.commit()

    chunks = [
        r.to_chunk()
        for r in db.query(ChunkRecord)
        .filter(ChunkRecord.doc_id == doc.id, ChunkRecord.doc_version == doc.version)
        .order_by(ChunkRecord.chunk_index)
        .all()
    ]

    # Incremental-index reconcile (only with a real store): tombstone ghosts and
    # embed ONLY the new/changed chunks.
    if collection is not None:
        from app.pipeline.vector_store import reconcile_document
        report = reconcile_document(collection, doc.id, chunks, doc.version, bm25=bm25, user_id=doc.user_id)
        to_upsert = [c for c in chunks if c.chunk_id in report.to_upsert_ids]
        logger.info(
            "Embed stage: doc=%s reconcile %s — embedding %d/%d chunks",
            doc.id, report.counts, len(to_upsert), len(chunks),
        )
        return embed_document(to_upsert, doc.id, provider, sink, job)

    return embed_document(chunks, doc.id, provider, sink, job)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _set_status(doc: Document, new_status: DocumentStatus, db) -> None:
    doc.status = new_status
    db.commit()
    logger.debug("doc=%s status → %s", doc.id, new_status.value)


def _fail(doc: Document, message: str, db) -> None:
    doc.status = DocumentStatus.failed
    doc.error_message = message[:900]
    db.commit()
    logger.error("Pipeline failed for doc=%s: %s", doc.id, message)
