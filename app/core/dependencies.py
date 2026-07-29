import logging

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import chromadb
from app.database import get_db
from app.core.security import verify_access_token
from app.models.user import User

logger = logging.getLogger(__name__)

bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    FastAPI dependency — extracts and validates the JWT from the Authorization header.
    Raises 401 if missing or invalid.
    """
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    payload = verify_access_token(credentials.credentials)
    if payload is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    user = db.query(User).filter(User.id == payload.get("sub")).first()
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")

    return user


def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User | None:
    """Like get_current_user but returns None instead of 401 when the
    Authorization header is missing/invalid. Used by endpoints that also accept a
    `?token=` query param (e.g. iframe-loaded file previews)."""
    if credentials is None:
        return None
    payload = verify_access_token(credentials.credentials)
    if payload is None:
        return None
    user = db.query(User).filter(User.id == payload.get("sub")).first()
    if user is None or not user.is_active:
        return None
    return user


def get_chroma(request: Request) -> chromadb.ClientAPI:
    """
    FastAPI dependency — returns the shared ChromaDB client.

    Normally the client is built once in the app lifespan and stored on
    app.state.chroma. If it is missing (e.g. a request arrives before/without the
    lifespan having run — as TestClient does when not used as a context manager),
    build it lazily via the single construction site and cache it. This keeps the
    mode decision in exactly one place (build_chroma_client) either way.
    """
    client = getattr(request.app.state, "chroma", None)
    if client is None:
        from app.core.chroma import build_chroma_client
        client = build_chroma_client()
        request.app.state.chroma = client
    return client


def get_arq_pool(request: Request):
    """
    FastAPI dependency — returns the shared arq Redis pool used to enqueue pipeline
    jobs, or None if no pool is configured on this process.

    The pool is created in the app lifespan (needs a running Redis). When it is
    absent — e.g. TestClient not used as a context manager, or PIPELINE_SYNC dev
    mode — this returns None and the upload endpoint falls back to running the
    pipeline inline via BackgroundTasks. Building a pool lazily is not possible
    here because that requires an async Redis connection.
    """
    return getattr(request.app.state, "arq_pool", None)


def get_read_chroma(
    request: Request,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
    default_client=Depends(get_chroma),
):
    """Vector client for READS, resolved from what this user's documents actually use.

    With per-document routing a corpus can span backends, so a search across all
    documents has to ask each one and merge. `build_read_client` returns the single
    concrete client when only one backend is involved — the common case — and a
    fan-out only when the corpus is genuinely mixed, so nobody pays for the general
    case unless they are in it.

    Falls back to the process-wide client if anything goes wrong: a routing failure
    should degrade to the default store, not break search.
    """
    try:
        from app.core.vector_registry import backends_for_user, default_backend
        from app.pipeline.multi_backend_store import build_read_client

        backends = backends_for_user(db, current_user.id)
        # When every document lives on the default backend — the overwhelmingly
        # common case, and every test case — return the process client untouched.
        # Resolving through the registry here would build a SECOND client and
        # silently bypass any injected/overridden one, which is how tests and the
        # eval harnesses supply a temporary store.
        if backends == [default_backend()]:
            return default_client
        return build_read_client(backends)
    except Exception:
        logger.warning("read-client routing failed, using process default", exc_info=True)
        return default_client
