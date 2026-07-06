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
    app.include_router(eval_router)      # Admin-gated: GET /eval/run (dense-only baseline)
    app.include_router(analysis_router)  # Chunk analyser, embedding inspection, UMAP viz
    # Remaining routers wired in as phases complete:
    # app.include_router(chunks_router, prefix="/chunks")   # Phase 4/5
    # app.include_router(debug_router, prefix="/debug")     # Phase 9
    # (Phase 6 embed_fn + Phase 8 bm25 are wired on app.state inside lifespan.)

    return app


app = create_app()
