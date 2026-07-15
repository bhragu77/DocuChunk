"""
arq worker entrypoint.

Run with:  arq app.worker.WorkerSettings

The worker runs in its OWN process, separate from the FastAPI web app. Its job is
to execute the document pipeline (parse → clean → chunk → [embed later]) off the
web request path, so uploads return immediately and long/CPU-heavy work is
crash-resumable (arq re-runs a task after a crash; the tasks are idempotent).

Shared, heavy resources are built ONCE at worker startup and stored on the arq
`ctx` dict so tasks don't rebuild them per invocation:
  * session_factory — the SQLAlchemy session factory (tasks open their own session)
  * chroma          — the ChromaDB client, via the single construction site
  * (Part 1) embed_provider — the sentence-transformers model will load here as a
    worker-lifetime singleton, NOT per task.
"""
import logging

from app.config import get_settings
from app.core.chroma import build_chroma_client
from app.database import SessionLocal, init_db
from app.workers.tasks import run_pipeline

logger = logging.getLogger(__name__)
settings = get_settings()


async def on_startup(ctx: dict) -> None:
    """Build shared resources once per worker process and stash them on ctx."""
    init_db()  # idempotent — make sure tables exist before any task runs
    ctx["session_factory"] = SessionLocal
    ctx["chroma"] = build_chroma_client()

    # Per-user BM25 lexical index (Phase 8). Built once per worker so the embed
    # stage's sink can co-write it alongside Chroma and reconcile can tombstone
    # ghosts from it. Persisted next to Chroma so the web process reads the same files.
    from app.pipeline.bm25_index import BM25Index
    ctx["bm25"] = BM25Index(persist_dir=settings.chroma_persist_dir)

    # Load the sentence-transformers model ONCE per worker (a worker-lifetime
    # singleton), NOT per task. Guarded by EMBED_PROVIDER so hf_api mode doesn't
    # download/load a local model needlessly. build_provider(ctx) reads ctx["embed_model"].
    ctx["embed_model"] = None
    if settings.embed_provider == "local":
        from sentence_transformers import SentenceTransformer
        ctx["embed_model"] = SentenceTransformer(settings.hf_embed_model)
        logger.info("Loaded ST model once: %s", settings.hf_embed_model)
    else:
        logger.info("EMBED_PROVIDER=%s — no local ST model loaded", settings.embed_provider)

    # NOTE (Story 1): the worker deliberately does NOT build a generation provider.
    # Generation is interactive, latency-bound work that belongs in the REQUEST PATH
    # (the web process wires it onto app.state.llm_fn) so a user gets tokens back on
    # the same connection — and, next, so answers can STREAM. The worker exists for
    # batch, crash-resumable work (parse→chunk→embed, later map-reduce summarization);
    # putting synchronous LLM calls here would block the pipeline queue for no benefit.
    # So ctx has no "llm_fn" key by design — asserted in the tests.

    logger.info("arq worker startup complete [chroma_mode=%s]", settings.chroma_mode)


async def on_shutdown(ctx: dict) -> None:
    """Release worker-lifetime resources."""
    logger.info("arq worker shutting down")


class WorkerSettings:
    """
    arq reads this class by import path (app.worker.WorkerSettings). It lists the
    task functions the worker can run and wires in the Redis connection + lifecycle
    hooks. Redis connection details come from the single build_redis_settings()
    helper so the worker and the web app's pool always agree.
    """
    functions = [run_pipeline]
    on_startup = on_startup
    on_shutdown = on_shutdown
    redis_settings = settings.build_redis_settings()
