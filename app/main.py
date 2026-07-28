from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

import os
import logging

from app.config import get_settings
from app.core.chroma import build_chroma_client
from app.database import init_db
from app.routers import health
from app.routers.auth import router as auth_router
from app.routers.documents import router as docs_router
from app.routers.pages import router as pages_router
from app.routers.search import search_router, generate_router
from app.routers.agent import agent_router
from app.routers.pipeline import pipeline_router
from app.routers.chat import router as chat_router
from app.routers.eval import router as eval_router
from app.routers.analysis import router as analysis_router

settings = get_settings()

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: initialise DB tables, ChromaDB client, required directories.
    Shutdown: clean up resources.
    """
    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info("Starting %s v%s [%s]", settings.app_name, settings.app_version, settings.app_env)

    # Ensure required directories exist
    os.makedirs(settings.upload_dir, exist_ok=True)
    if settings.chroma_mode.lower() == "persistent":
        os.makedirs(settings.chroma_persist_dir, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    logger.info("Directories ready")

    # Create all DB tables (idempotent)
    init_db()
    logger.info("Database tables initialised")

    # Initialise ChromaDB (mode chosen by CHROMA_MODE) and store the client on app
    # state so routers/dependencies can access the one shared client.
    app.state.chroma = build_chroma_client()
    logger.info("ChromaDB ready [mode=%s]", settings.chroma_mode)

    # Phase 8 retrieval wiring — the /search and /generate routers read these off
    # app.state (they 503 until present).
    #   bm25     — per-user lexical index, persisted next to Chroma so it sees the
    #              same files the worker writes during ingestion.
    #   embed_fn — the query embedder (lazy: loads the ST model on first search only).
    from app.pipeline.bm25_index import BM25Index
    from app.pipeline.embedding_providers import embed_single
    app.state.bm25 = BM25Index(persist_dir=settings.chroma_persist_dir)
    app.state.embed_fn = embed_single
    logger.info("Phase 8 retrieval wired [bm25 + query embedder]")

    # Story 1 — generation provider. Built from GEN_PROVIDER and exposed as
    # app.state.llm_fn (Callable[[str], str]), the seam /generate/answer reads. Bound
    # to the configured max_tokens/temperature so callers just pass a prompt. If the
    # provider can't be built (e.g. GEN_PROVIDER=gemini with no key), we leave llm_fn
    # None so the endpoint returns Story 0's honest "generation_not_configured" shape
    # rather than crashing startup. Generation lives in the WEB process only (see
    # worker.py for why); the worker never builds one.
    from app.generation.factory import (
        build_gen_provider,
        build_verify_provider,
        make_llm_fn,
        make_llm_stream_fn,
        make_verify_fn,
    )
    app.state.llm_fn = None
    # Story 6 — the STREAMING seam. Parallel to llm_fn but returns an Iterator[str]
    # of text chunks. /generate/answer uses it for Phase 1 (token streaming) and
    # falls back to llm_fn if it's absent or fails on the first chunk. None when no
    # provider is configured (same guard as llm_fn).
    app.state.llm_stream_fn = None
    # Story 7 — the VERIFICATION seam (model tiering). validate_citations Tier-2 +
    # groundedness_check run through verify_fn, which may be a cheaper/lower-temp
    # model than the generator. When VERIFY_PROVIDER is unset, it reuses the
    # generation provider (backward compatible). None when generation is unconfigured.
    app.state.verify_fn = None
    # Phase 9A — the model names the "generate"/rerank spans record. Kept as plain
    # strings on app.state (the llm_fn seam returns only text, hiding the provider),
    # so the endpoint can label spans without reaching into the provider.
    app.state.gen_model_name = None
    app.state.verify_model_name = None
    try:
        _gen_provider = build_gen_provider(settings)
        app.state.llm_fn = make_llm_fn(_gen_provider, settings)
        app.state.llm_stream_fn = make_llm_stream_fn(_gen_provider, settings)
        _verify_provider = build_verify_provider(settings, gen_provider=_gen_provider)
        app.state.verify_fn = make_verify_fn(_verify_provider, settings)
        app.state.gen_model_name = _gen_provider.model_name
        app.state.verify_model_name = _verify_provider.model_name
        logger.info(
            "Generation wired [gen=%s/%s verify=%s/%s stream=%s]",
            _gen_provider.provider_name, _gen_provider.model_name,
            _verify_provider.provider_name, _verify_provider.model_name,
            settings.stream_enabled,
        )
    except Exception as exc:
        logger.warning(
            "Generation provider unavailable (%s) — /generate/answer will return "
            "generation_not_configured until GEN_PROVIDER is set with a valid key.", exc,
        )

    # Chat model picker — the per-request "gemini" | "offline" generation registry
    # (independent of the default seam above). Built best-effort; empty when neither
    # tier is configured, in which case /generate/answer just uses the default llm_fn.
    # Exposed to the UI via GET /generate/models so the picker can hide unavailable
    # tiers instead of erroring.
    app.state.gen_registry = {}
    try:
        from app.generation.factory import build_gen_registry
        app.state.gen_registry = build_gen_registry(settings)
        if app.state.gen_registry:
            logger.info(
                "Chat model picker wired [%s]",
                ", ".join(f"{k}={v['model_name']}" for k, v in app.state.gen_registry.items()),
            )
    except Exception as exc:
        logger.warning("Chat model registry unavailable (%s) — picker falls back to default.", exc)

    # Story 7 — the ANSWER CACHE. Skips the whole chain for a repeated
    # (user, doc, doc_version, query). None when CACHE_BACKEND=none (the default).
    from app.generation.cache import build_cache
    app.state.answer_cache = None
    try:
        app.state.answer_cache = build_cache(settings)
    except Exception as exc:
        logger.warning("Answer cache unavailable (%s) — caching disabled", exc)

    # arq Redis pool — used to ENQUEUE pipeline jobs onto the worker. The worker
    # itself is a separate process (arq app.worker.WorkerSettings). If Redis is
    # unavailable, the app still starts; the upload endpoint falls back to inline
    # execution (see get_arq_pool / upload_document).
    app.state.arq_pool = None
    try:
        from arq import create_pool
        app.state.arq_pool = await create_pool(settings.build_redis_settings())
        logger.info("arq Redis pool ready at %s:%s", settings.redis_host, settings.redis_port)
    except Exception as exc:
        logger.warning("arq Redis pool unavailable (%s) — uploads will run inline", exc)

    yield  # ← application runs here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Shutting down %s", settings.app_name)
    # Flush any buffered trace data (no-op tracer in Phase 9A; the real backend in
    # 9B blocks here to drain its queue). Fail-open — never blocks shutdown on error.
    from app.observability import flush as flush_traces
    flush_traces()
    if getattr(app.state, "arq_pool", None) is not None:
        await app.state.arq_pool.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Production-grade document retrieval & RAG platform",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        lifespan=lifespan,
    )

    # ── Session middleware (required for OAuth state/CSRF protection) ─────────
    # Must be added before CORS middleware
    app.add_middleware(SessionMiddleware, secret_key=settings.secret_key)

    # ── CORS ─────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Static files ─────────────────────────────────────────────────────────
    app.mount("/static", StaticFiles(directory="static"), name="static")

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(pages_router)   # HTML pages — must come before API routers
    app.include_router(health.router)
    app.include_router(auth_router)
    app.include_router(docs_router)
    app.include_router(search_router)    # Phase 8: POST /search/semantic
    app.include_router(generate_router)  # Phase 8: POST /generate/answer
    app.include_router(agent_router)     # Agentic RAG: POST /generate/agent (ReAct multi-hop)
    app.include_router(pipeline_router)  # Pipeline dashboard: record-then-replay trace artifacts
    app.include_router(chat_router)      # Chat sessions: /chat/sessions (per-user history + lock)
    app.include_router(eval_router)      # Admin-gated: GET /eval/run (dense-only baseline)
    app.include_router(analysis_router)  # Chunk analyser, embedding inspection, UMAP viz
    # Remaining routers wired in as phases complete:
    # app.include_router(chunks_router, prefix="/chunks")   # Phase 4/5
    # app.include_router(debug_router, prefix="/debug")     # Phase 9
    # (Phase 6 embed_fn + Phase 8 bm25 are wired on app.state inside lifespan.)

    return app


app = create_app()
