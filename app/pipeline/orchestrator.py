"""
Pipeline Orchestrator — runs all stages in sequence for a single document.

This function runs in a background thread (via FastAPI BackgroundTasks).
It creates its own DB session because the request session is already closed
by the time the background task executes.

Current stage coverage:
  [✓] Stage 1 — Parse   (Phase 3)
  [✓] Stage 2 — Clean   (Phase 3)
  [ ] Stage 3 — Chunk   (Phase 4 — stub)
  [ ] Stage 4 — Analyse (Phase 5 — stub)
  [ ] Stage 5 — Embed   (Phase 6 — stub)
  [ ] Stage 6 — Store   (Phase 7 — stub)

As each Phase is built, replace the stub below with the real import + call.

Learning note — why a separate DB session?
  FastAPI's get_db() dependency creates a session per request and closes it when
  the response is sent. A BackgroundTask runs AFTER the response. If you reuse
  the request session you will get "Session already closed" errors. Always create
  a fresh session inside background tasks.
"""
import logging
from datetime import datetime

from app.database import SessionLocal
from app.models.document import Document, DocumentStatus
from app.pipeline.parser import parse
from app.pipeline.cleaner import clean_pages

logger = logging.getLogger(__name__)


def run_pipeline(doc_id: str) -> None:
    """
    Entry point for the background pipeline task.
    Call this from any FastAPI endpoint via BackgroundTasks.add_task().
    """
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if doc is None:
            logger.error("Pipeline: document %s not found in DB", doc_id)
            return

        logger.info("Pipeline started for doc_id=%s (%s)", doc_id, doc.original_filename)
        _run_stages(doc, db)

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
    finally:
        db.close()


def _run_stages(doc: Document, db) -> None:
    """Execute pipeline stages in order, updating status between each."""

    # ── Stage 1+2: Parse + Clean ─────────────────────────────────────────────
    _set_status(doc, DocumentStatus.parsing, db)

    try:
        raw_pages = parse(doc.file_path, doc.original_filename, doc.mime_type)
    except Exception as exc:
        _fail(doc, f"Parse error: {exc}", db)
        return

    if not raw_pages:
        _fail(doc, "No text could be extracted from the document. It may be a scanned image.", db)
        return

    cleaned_pages = clean_pages(raw_pages)

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

    # STUB — replaced in Phase 4
    # chunks = chunker.chunk(cleaned_pages, config=default_chunker_config())
    # doc.chunk_count = len(chunks)
    # doc.chunker_config = config.__dict__
    # db.commit()

    # ── Stage 4: Analyse ──────────────────────────────────────────────────────
    # STUB — replaced in Phase 5
    # stats = analyser.analyse(chunks)

    # ── Stage 5: Embed ────────────────────────────────────────────────────────
    _set_status(doc, DocumentStatus.embedding, db)

    # STUB — replaced in Phase 6
    # embedded_chunks = embedder.embed(chunks)

    # ── Stage 6: Store ────────────────────────────────────────────────────────
    # STUB — replaced in Phase 7
    # vector_store.store(doc.user_id, doc.id, embedded_chunks)

    # Mark ready
    doc.status = DocumentStatus.ready
    db.commit()
    logger.info("Pipeline complete for doc_id=%s", doc.id)


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
