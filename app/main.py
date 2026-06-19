from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

import chromadb
import os
import logging

from app.config import get_settings
from app.database import init_db
from app.routers import health
from app.routers.auth import router as auth_router
from app.routers.documents import router as docs_router
from app.routers.pages import router as pages_router

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
    os.makedirs(settings.chroma_persist_dir, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    logger.info("Directories ready")

    # Create all DB tables (idempotent)
    init_db()
    logger.info("Database tables initialised")

    # Initialise ChromaDB and store client on app state so routers can access it
    chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    app.state.chroma = chroma_client
    logger.info("ChromaDB ready at %s", settings.chroma_persist_dir)

    yield  # ← application runs here

    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Shutting down %s", settings.app_name)


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
    # Future routers registered as we build each phase:
    # app.include_router(chunks_router, prefix="/chunks")
    # app.include_router(embed_router, prefix="/embed")
    # app.include_router(search_router, prefix="/search")
    # app.include_router(generate_router, prefix="/generate")
    # app.include_router(debug_router, prefix="/debug")

    return app


app = create_app()
