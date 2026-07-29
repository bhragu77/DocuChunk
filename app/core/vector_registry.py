"""
Per-document vector backend routing.

WHY THIS EXISTS
===============
`build_chroma_client()` returns ONE client for the whole process, chosen by
`VECTOR_BACKEND`. That is correct when every document lives in the same store. Once
a document can choose its own backend at upload time, three things change:

  1. Several clients must be live at once, not one.
  2. Ingestion must write to the backend that document chose.
  3. A search across ALL of a user's documents may span backends, so it has to query
     each and merge — see `MultiBackendCollection`.

Clients are cached per backend name because construction is expensive: the pgvector
client builds an engine and runs schema DDL, and the Pinecone client performs a
network round-trip to create-or-verify the index. Rebuilding either per request
would dominate retrieval latency.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It does not migrate a document between backends. The vectors physically live in one
store; moving them means re-embedding and re-indexing, which is an ingestion job,
not a routing concern. `documents.vector_backend` is therefore write-once.
"""
from __future__ import annotations

import logging
from threading import Lock

from app.config import get_settings

logger = logging.getLogger(__name__)

# Backends a document may be assigned to. "chroma" stays supported for the backend
# comparison but is not offered in the UI unless it is actually running.
KNOWN_BACKENDS = ("pgvector", "pinecone", "chroma")

_clients: dict[str, object] = {}
_lock = Lock()


def default_backend() -> str:
    return (getattr(get_settings(), "vector_backend", None) or "pgvector").lower()


def normalise(name: str | None) -> str:
    """Map a stored/requested value onto a known backend.

    Rows written before per-document routing existed have NULL here, and an unknown
    value should never hard-fail a read — both resolve to the configured default.
    """
    n = (name or "").strip().lower()
    return n if n in KNOWN_BACKENDS else default_backend()


def available_backends() -> list[dict]:
    """Backends this deployment can actually use, for the upload UI.

    Availability is a configuration fact, not a preference: Pinecone without an API
    key would fail at ingestion time, long after the user chose it, so it is filtered
    out here rather than offered and then rejected.
    """
    s = get_settings()
    out = [{
        "id": "pgvector",
        "label": "pgvector (Postgres)",
        "hint": "Self-hosted. Same database as your documents — transactional, fastest.",
        "available": True,
    }]
    out.append({
        "id": "pinecone",
        "label": "Pinecone (managed)",
        "hint": "Managed service. No indexes to operate; adds network latency.",
        "available": bool(getattr(s, "pinecone_api_key", "")),
        "unavailable_reason": None if getattr(s, "pinecone_api_key", "") else "PINECONE_API_KEY not configured",
    })
    return out


def get_client(backend: str | None):
    """Return (building once, then caching) the client for `backend`."""
    name = normalise(backend)
    cached = _clients.get(name)
    if cached is not None:
        return cached
    with _lock:
        cached = _clients.get(name)
        if cached is not None:
            return cached
        client = _build(name)
        _clients[name] = client
        logger.info("vector registry: built client for backend=%s", name)
        return client


def _build(name: str):
    s = get_settings()
    if name == "pinecone":
        from app.pipeline.pinecone_store import PineconeClient
        return PineconeClient(
            api_key=getattr(s, "pinecone_api_key", ""),
            index_name=getattr(s, "pinecone_index", "docuchunk"),
            cloud=getattr(s, "pinecone_cloud", "aws"),
            region=getattr(s, "pinecone_region", "us-east-1"),
        )
    if name == "pgvector":
        from app.pipeline.pgvector_store import PgVectorClient
        return PgVectorClient(getattr(s, "pgvector_dsn", "") or s.database_url)
    # chroma — reuse the existing single construction site
    from app.core.chroma import build_chroma_client
    return build_chroma_client()


def reset_cache() -> None:
    """Test hook — clients are cached for the process lifetime."""
    with _lock:
        _clients.clear()


def backends_for_user(db, user_id: str) -> list[str]:
    """Distinct backends holding this user's READY documents.

    Drives whether a search needs one client or a fan-out. Only ready documents
    count: a document still parsing has no vectors, so including its backend would
    add a network round-trip to a namespace guaranteed to be empty.
    """
    from app.models.document import Document, DocumentStatus

    rows = (
        db.query(Document.vector_backend)
        .filter(Document.user_id == user_id, Document.status == DocumentStatus.ready)
        .distinct()
        .all()
    )
    found = {normalise(r[0]) for r in rows}
    return sorted(found) or [default_backend()]
