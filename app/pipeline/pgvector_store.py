"""
pgvector vector store — a drop-in replacement for the Chroma collection API.

WHY AN ADAPTER RATHER THAN A REFACTOR
=====================================
Twelve call sites across the app use a Chroma *collection* object directly
(`query`, `get`, `upsert`, `delete`). Rewriting all of them to a new protocol
would touch retrieval, ingestion, analysis, document deletion, context expansion
and all three eval harnesses at once — a large diff whose correctness could only
be judged after the fact.

Instead this module implements exactly the four methods those call sites use, with
the same keyword arguments and the same response shapes (including Chroma's nested
`[[...]]` lists for `query`). That makes the backend a one-line configuration
switch, keeps both implementations live so they can be compared on the same eval
fixture, and means a bug here shows up as a retrieval regression the existing
harness already measures.

STORAGE MODEL
-------------
One table, `vector_chunks`, with a `collection` column standing in for Chroma's
per-user collections:

    collection TEXT   -- "user_<id>", the existing collection_name()
    chunk_id   TEXT   -- Chroma's `ids`
    document   TEXT   -- Chroma's `documents`
    metadata   JSONB  -- Chroma's `metadatas`  (GIN-indexed for doc_id filters)
    embedding  VECTOR -- pgvector

Primary key (collection, chunk_id) gives upsert-by-id semantics for free.

DISTANCE
--------
Cosine distance (`<=>`), matching Chroma's default for normalised embeddings, so
the fused RRF ranks stay comparable between backends. The HNSW index uses
`vector_cosine_ops` for the same reason.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Iterable, Sequence

from app.pipeline.embedding_providers import EMBED_DIM

logger = logging.getLogger(__name__)

TABLE = "vector_chunks"


def _vec_literal(vec: Sequence[float]) -> str:
    """pgvector's text input format: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{float(v):.8g}" for v in vec) + "]"


def _where_sql(where: dict | None, params: dict) -> str:
    """Translate Chroma's `where` filter into a JSONB predicate.

    Only equality on scalar metadata keys is supported, which is all the codebase
    uses (`{"doc_id": ...}`). Anything richer raises rather than silently matching
    everything — a filter that quietly stops filtering would leak another user's
    chunks into a result set.
    """
    if not where:
        return ""
    clauses = []
    for i, (key, val) in enumerate(where.items()):
        if isinstance(val, (dict, list)):
            raise ValueError(
                f"pgvector adapter supports only scalar equality filters, got {key}={val!r}"
            )
        pkey = f"w{i}"
        params[pkey] = str(val)
        clauses.append(f"metadata->>'{key}' = :{pkey}")
    return " AND " + " AND ".join(clauses)


class PgVectorCollection:
    """Chroma-collection-compatible view over one `collection` value."""

    def __init__(self, engine, name: str):
        self._engine = engine
        self.name = name

    # ── writes ────────────────────────────────────────────────────────────────

    def upsert(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[dict],
    ) -> None:
        from sqlalchemy import text

        if not ids:
            return
        rows = [
            {
                "collection": self.name,
                "chunk_id": cid,
                "document": doc or "",
                "metadata": json.dumps(meta or {}),
                "embedding": _vec_literal(emb),
            }
            for cid, emb, doc, meta in zip(ids, embeddings, documents, metadatas)
        ]
        stmt = text(
            f"""
            INSERT INTO {TABLE} (collection, chunk_id, document, metadata, embedding)
            VALUES (:collection, :chunk_id, :document, CAST(:metadata AS JSONB),
                    CAST(:embedding AS vector))
            ON CONFLICT (collection, chunk_id) DO UPDATE
              SET document  = EXCLUDED.document,
                  metadata  = EXCLUDED.metadata,
                  embedding = EXCLUDED.embedding
            """
        )
        with self._engine.begin() as conn:
            conn.execute(stmt, rows)

    def delete(self, ids: Sequence[str] | None = None, where: dict | None = None) -> None:
        from sqlalchemy import text

        params: dict[str, Any] = {"collection": self.name}
        sql = f"DELETE FROM {TABLE} WHERE collection = :collection"
        if ids:
            params["ids"] = list(ids)
            sql += " AND chunk_id = ANY(:ids)"
        sql += _where_sql(where, params)
        with self._engine.begin() as conn:
            conn.execute(text(sql), params)

    # ── reads ─────────────────────────────────────────────────────────────────

    def get(
        self,
        ids: Sequence[str] | None = None,
        where: dict | None = None,
        include: Iterable[str] | None = None,
        limit: int | None = None,
    ) -> dict:
        from sqlalchemy import text

        include = list(include or ["documents", "metadatas"])
        params: dict[str, Any] = {"collection": self.name}
        sql = (
            f"SELECT chunk_id, document, metadata, embedding FROM {TABLE} "
            "WHERE collection = :collection"
        )
        if ids:
            params["ids"] = list(ids)
            sql += " AND chunk_id = ANY(:ids)"
        sql += _where_sql(where, params)
        if limit:
            params["lim"] = int(limit)
            sql += " LIMIT :lim"
        with self._engine.connect() as conn:
            rows = conn.execute(text(sql), params).fetchall()
        return self._shape(rows, include)

    def query(
        self,
        query_embeddings: Sequence[Sequence[float]],
        n_results: int = 10,
        where: dict | None = None,
        include: Iterable[str] | None = None,
    ) -> dict:
        from sqlalchemy import text

        include = list(include or ["documents", "metadatas", "distances"])
        out: dict[str, list] = {k: [] for k in ("ids", "documents", "metadatas", "distances")}
        for vec in query_embeddings:
            params: dict[str, Any] = {
                "collection": self.name,
                "q": _vec_literal(vec),
                "lim": int(n_results),
            }
            sql = (
                f"SELECT chunk_id, document, metadata, embedding, "
                f"embedding <=> CAST(:q AS vector) AS distance FROM {TABLE} "
                "WHERE collection = :collection"
            )
            sql += _where_sql(where, params)
            sql += " ORDER BY embedding <=> CAST(:q AS vector) LIMIT :lim"
            with self._engine.connect() as conn:
                rows = conn.execute(text(sql), params).fetchall()
            shaped = self._shape(rows, include, with_distance=True)
            for key in out:
                out[key].append(shaped.get(key, []))
        # Chroma nests one list per query embedding, and omits keys not requested.
        keys = ["ids"] + [k for k in ("documents", "metadatas", "distances") if k in include]
        return {k: out[k] for k in keys}

    def count(self) -> int:
        from sqlalchemy import text

        with self._engine.connect() as conn:
            return int(
                conn.execute(
                    text(f"SELECT count(*) FROM {TABLE} WHERE collection = :c"),
                    {"c": self.name},
                ).scalar_one()
            )

    # ── shaping ───────────────────────────────────────────────────────────────

    @staticmethod
    def _shape(rows, include: list[str], *, with_distance: bool = False) -> dict:
        out: dict[str, list] = {"ids": [r[0] for r in rows]}
        if "documents" in include:
            out["documents"] = [r[1] for r in rows]
        if "metadatas" in include:
            out["metadatas"] = [dict(r[2] or {}) for r in rows]
        if "embeddings" in include:
            out["embeddings"] = [_parse_vec(r[3]) for r in rows]
        if with_distance and "distances" in include:
            out["distances"] = [float(r[4]) for r in rows]
        return out


def _parse_vec(raw) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [float(x) for x in raw]
    return [float(x) for x in str(raw).strip("[]").split(",") if x.strip()]


class PgVectorClient:
    """Chroma-client-compatible factory over a Postgres engine."""

    def __init__(self, dsn: str, dim: int = EMBED_DIM):
        from sqlalchemy import create_engine

        self._engine = create_engine(dsn, pool_pre_ping=True, pool_size=10, max_overflow=20)
        self._dim = dim
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        from sqlalchemy import text

        with self._engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(
                text(
                    f"""
                    CREATE TABLE IF NOT EXISTS {TABLE} (
                        collection TEXT   NOT NULL,
                        chunk_id   TEXT   NOT NULL,
                        document   TEXT   NOT NULL DEFAULT '',
                        metadata   JSONB  NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding  vector({self._dim}),
                        PRIMARY KEY (collection, chunk_id)
                    )
                    """
                )
            )
            # GIN on metadata: every filtered read is `metadata->>'doc_id' = ...`.
            conn.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS {TABLE}_metadata_gin "
                    f"ON {TABLE} USING GIN (metadata)"
                )
            )
            # HNSW for ANN search. Cosine ops to match Chroma's default distance so
            # ranks stay comparable between backends.
            conn.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS {TABLE}_embedding_hnsw "
                    f"ON {TABLE} USING hnsw (embedding vector_cosine_ops)"
                )
            )
        logger.info("pgvector schema ready (dim=%d)", self._dim)

    # Chroma client API surface actually used by the app.
    def get_collection(self, name: str) -> PgVectorCollection:
        return PgVectorCollection(self._engine, name)

    def get_or_create_collection(self, name: str, **_kw) -> PgVectorCollection:
        return PgVectorCollection(self._engine, name)

    def delete_collection(self, name: str) -> None:
        from sqlalchemy import text

        with self._engine.begin() as conn:
            conn.execute(
                text(f"DELETE FROM {TABLE} WHERE collection = :c"), {"c": name}
            )

    def heartbeat(self) -> int:
        from sqlalchemy import text

        with self._engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return 1
