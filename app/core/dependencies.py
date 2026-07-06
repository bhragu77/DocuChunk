from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import chromadb
from app.database import get_db
from app.core.security import verify_access_token
from app.models.user import User

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
