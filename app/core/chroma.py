"""
Single, config-driven ChromaDB client construction site.

CHROMA_MODE (config.py) is the ONLY thing that decides persistent-vs-http. Nothing
else in the codebase constructs a Chroma client or hardcodes a mode, so dev/test
(persistent, local dir) and a deployed server (http) can never drift apart.

Why this file exists at all: a thin `chromadb-client` package installed alongside
the full `chromadb` used to shadow it and force "http-only client mode", which made
PersistentClient raise at startup and at test-collection time. Centralising client
creation makes the mode explicit and auditable so that failure can't recur silently.

Cosine space: every collection is created with metadata={"hnsw:space": "cosine"}.
This is mandatory, not cosmetic — the Part 1 embedder L2-normalizes its vectors, and
on unit vectors cosine similarity == dot product. A collection left on Chroma's
DEFAULT L2 space would silently rank by Euclidean distance and mis-order results.
"""
import logging

import chromadb

from app.config import get_settings

logger = logging.getLogger(__name__)

# The one true distance space for every collection. See module docstring.
COSINE_SPACE = {"hnsw:space": "cosine"}


def _collection_metadata() -> dict:
    """
    Collection metadata = cosine space + HNSW tuning knobs from config.

    These are set ONCE at collection creation and are immutable for the life of
    the collection, so changing hnsw_m / hnsw_construction_ef only affects NEW
    collections (delete & re-ingest to rebuild an existing one). hnsw_search_ef is
    the query-time recall/latency dial. See config.py for the trade-offs.
    """
    s = get_settings()
    return {
        **COSINE_SPACE,
        "hnsw:M": s.hnsw_m,
        "hnsw:construction_ef": s.hnsw_construction_ef,
        "hnsw:search_ef": s.hnsw_search_ef,
    }


def build_chroma_client() -> "chromadb.ClientAPI":
    """
    Construct the ChromaDB client from CHROMA_MODE alone. Called once at app
    startup (lifespan) and stored on app.state.chroma; tests build their own via
    the same function against a temp dir.
    """
    s = get_settings()
    mode = (s.chroma_mode or "").lower()

    if mode == "persistent":
        logger.info("ChromaDB client: persistent mode at %s", s.chroma_persist_dir)
        return chromadb.PersistentClient(path=s.chroma_persist_dir)

    if mode == "http":
        logger.info("ChromaDB client: http mode at %s:%s", s.chroma_host, s.chroma_port)
        return chromadb.HttpClient(host=s.chroma_host, port=s.chroma_port)

    raise ValueError(
        f"Invalid CHROMA_MODE={s.chroma_mode!r}; expected 'persistent' or 'http'."
    )


def get_or_create_collection(client: "chromadb.ClientAPI", name: str):
    """
    The single collection-creation site. ALWAYS cosine space.

    Phase 7's vector store must route every collection it touches through here so
    that document vectors and Phase 8 query vectors share the same distance metric
    and HNSW tuning.
    """
    return client.get_or_create_collection(name=name, metadata=_collection_metadata())
