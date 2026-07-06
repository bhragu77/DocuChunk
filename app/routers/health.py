from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.database import get_db
from app.config import get_settings
from app.core.dependencies import get_chroma
import os

router = APIRouter(tags=["health"])
settings = get_settings()


@router.get("/health")
def health_check(db: Session = Depends(get_db), chroma=Depends(get_chroma)):
    """
    Liveness + readiness check.
    Verifies DB connectivity and ChromaDB availability.
    Returns 200 when all systems are up.
    """
    checks: dict = {}

    # Database check
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {e}"

    # ChromaDB check — exercise the ACTUAL configured client (persistent or http)
    # from app.state, not a freshly-built one, so this reflects the client the app
    # really uses. heartbeat() round-trips to the backing store.
    try:
        chroma.heartbeat()
        checks["chromadb"] = "ok"
    except Exception as e:
        checks["chromadb"] = f"error: {e}"

    # Upload dir check
    checks["upload_dir"] = "ok" if os.path.isdir(settings.upload_dir) else "missing"

    all_ok = all(v == "ok" for v in checks.values())

    return {
        "status": "healthy" if all_ok else "degraded",
        "version": settings.app_version,
        "environment": settings.app_env,
        "checks": checks,
    }
