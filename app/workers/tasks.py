"""
arq worker tasks for the document pipeline.

GOVERNING RULE
==============
A task receives ONLY serializable primitives (here, `doc_id: str`) — never an ORM
object or an open DB session. It reconstructs everything it needs by querying the
DB from `doc_id`, using the shared resources placed on the arq `ctx` at worker
startup (see app/worker.py). Tasks are IDEMPOTENT: re-running one on the same
`doc_id` is safe and resumes rather than duplicating work. This is what makes
resume-after-crash possible — arq re-runs the task from the start after a crash,
and the orchestrator's resume-by-status / delete-then-insert logic absorbs it.

Testing note: tasks are unit-tested by calling the task FUNCTION directly with a
minimal ctx (a dict carrying a `session_factory`) and a real temp DB — no Redis
and no running worker. See tests/test_worker.py.
"""
import asyncio
import logging

from app.database import SessionLocal
from app.observability import flush as flush_traces, span
from app.pipeline.embedding_providers import build_provider
from app.pipeline.orchestrator import run_pipeline as _run_pipeline_sync

logger = logging.getLogger(__name__)


async def run_pipeline(ctx: dict, doc_id: str, trace_id: str | None = None) -> dict:
    """
    Worker task: run the full document pipeline for `doc_id`.

    Reconstructs all state from `doc_id` + the DB (nothing is passed from the
    enqueuer besides the id). The heavy, synchronous orchestrator runs in a thread
    so it never blocks the worker's event loop / other jobs.

    `trace_id` (Phase 9A) carries the API-side trace across the process boundary
    (the enqueuer adds it to the job kwargs); the "ingest.document" root starts
    under it so the whole ingestion shares one trace. It is optional and unused by
    any resume/idempotency logic, so an older enqueue without it still works.

    Idempotent: safe for arq to re-run after a crash (resume-by-status inside the
    orchestrator turns an already-finished doc into a no-op and an already-chunked
    doc into a jump straight to the embed stage).
    """
    # Prefer the worker-lifetime session factory from ctx; fall back to the app's
    # SessionLocal so the task also works if enqueued outside the worker (defensive).
    session_factory = ctx.get("session_factory", SessionLocal)
    # Build the embedding provider from ctx (the ST model is a worker-lifetime
    # singleton loaded in on_startup). None → embed stage is skipped.
    provider = build_provider(ctx)
    # The shared Chroma client (built once in on_startup). Passing it in turns on the
    # reconcile + upsert path in the embed stage; None → nothing is stored.
    chroma = ctx.get("chroma")
    # The shared BM25 index (built once in on_startup). Passed alongside chroma so the
    # embed stage keeps the lexical index in lockstep with the vector store.
    bm25 = ctx.get("bm25")

    logger.info("worker: run_pipeline start doc_id=%s (job_try=%s)", doc_id, ctx.get("job_try"))
    try:
        # The root span is entered here (async task); asyncio.to_thread copies this
        # context, so the orchestrator's stage spans (in the thread) nest under it.
        with span("ingest.document", trace_id=trace_id, doc_id=doc_id) as root:
            final_status = await asyncio.to_thread(
                _run_pipeline_sync, doc_id, session_factory, provider, chroma, bm25
            )
            root.update(status=final_status)
        logger.info("worker: run_pipeline done doc_id=%s status=%s", doc_id, final_status)
        return {"doc_id": doc_id, "status": final_status}
    finally:
        # Flush at the end of each worker job — the one place blocking is allowed
        # (no-op in Phase 9A). Fail-open.
        flush_traces()
