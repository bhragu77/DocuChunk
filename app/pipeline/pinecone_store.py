"""
Pinecone vector store — a managed-service backend behind the Chroma collection API.

WHY THIS EXISTS
===============
pgvector and Chroma are both self-hosted. Pinecone is a *managed* vector database:
no index building, no HNSW tuning, no disk to run out of — and network latency plus
a bill instead. Supporting all three behind one interface is what makes the trade-off
measurable rather than a matter of opinion: `scripts/backend_benchmark.py` runs the
identical eval fixture across every backend and reports MRR, recall@k and retrieval
latency side by side.

It implements exactly the four methods the twelve call sites in this codebase use
(`upsert` / `query` / `get` / `delete`), with the same keyword arguments and the same
response shapes as Chroma — including Chroma's nested `[[...]]` lists for `query`.
Same contract as `pgvector_store.py`.

THREE TRANSLATIONS THAT MUST BE EXACT
-------------------------------------
Each one silently corrupts the benchmark if wrong — the system keeps working and the
numbers stay plausible, which is the worst kind of bug to have in a measurement tool.

1. NAMESPACE — our per-user `collection_name(user_id)` maps to a Pinecone namespace.
   Namespaces are the isolation boundary; getting this wrong leaks one user's chunks
   into another's results.

2. FILTER SYNTAX — ours is `{"doc_id": "x"}`, Pinecone's is `{"doc_id": {"$eq": "x"}}`.

3. SCORE → DISTANCE — Pinecone returns cosine SIMILARITY (higher is better);
   `hybrid_search` consumes DISTANCE (lower is better) and feeds rank order into RRF.
   Passing similarity through unconverted inverts the dense leg's ranking, and RRF
   would then fuse a correctly-ordered BM25 list with a backwards dense list. We
   convert `distance = 1 - score` so ranks stay comparable across backends.

STORAGE MODEL
-------------
One Pinecone index, one namespace per user. Chunk text is stored in metadata under
`_document` because Pinecone has no separate document field; it is stripped back out
on read so callers see the same shape Chroma returns.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

from app.pipeline.embedding_providers import EMBED_DIM

logger = logging.getLogger(__name__)

# Pinecone has no dedicated "document" field, so the chunk text rides in metadata.
# Underscore-prefixed to keep it out of the user-facing metadata dict on read.
DOC_KEY = "_document"

# Pinecone metadata values must be str / number / bool / list[str]. Anything else
# (None, dict, nested list) is dropped rather than sent, because the API rejects the
# whole upsert on a single bad value — losing a batch of chunks to one null field.
_SCALARS = (str, int, float, bool)


def _clean_metadata(meta: dict) -> dict:
    out: dict[str, Any] = {}
    for k, v in (meta or {}).items():
        if v is None:
            continue
        if isinstance(v, _SCALARS):
            out[k] = v
        elif isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v):
            out[k] = list(v)
        else:
            out[k] = str(v)
    return out


def translate_filter(where: dict | None) -> dict | None:
    """Chroma's scalar-equality filter -> Pinecone's `$eq` form.

    Only scalar equality is supported, which is all this codebase uses
    (`{"doc_id": ...}`). Anything richer raises rather than being silently dropped:
    a filter that quietly stops filtering would leak another user's chunks into a
    result set. Same policy as the pgvector adapter.
    """
    if not where:
        return None
    out: dict[str, Any] = {}
    for key, val in where.items():
        if isinstance(val, dict):
            # Already an operator form ({"$eq": ...}) — pass through untouched.
            out[key] = val
        elif isinstance(val, (list, tuple)):
            raise ValueError(
                f"pinecone adapter supports only scalar equality filters, got {key}={val!r}"
            )
        else:
            out[key] = {"$eq": val}
    return out


def score_to_distance(score: float) -> float:
    """Cosine similarity (higher better) -> distance (lower better).

    The index is created with metric="cosine", for which Pinecone returns a
    similarity in [-1, 1]. Retrieval ranks ascending by distance, so this inversion
    is what keeps dense ranks — and therefore RRF fusion — identical in ORDER to the
    self-hosted backends.
    """
    return 1.0 - float(score)


class PineconeCollection:
    """Chroma-collection-compatible view over one Pinecone namespace."""

    def __init__(self, index, namespace: str):
        self._index = index
        self.name = namespace

    # ── writes ────────────────────────────────────────────────────────────────

    def upsert(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[dict],
    ) -> None:
        if not ids:
            return
        vectors = []
        for cid, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
            md = _clean_metadata(meta)
            md[DOC_KEY] = doc or ""
            vectors.append({"id": cid, "values": [float(x) for x in emb], "metadata": md})
        # Pinecone caps a single upsert request; batch to stay well inside it.
        for i in range(0, len(vectors), 100):
            self._index.upsert(vectors=vectors[i : i + 100], namespace=self.name)

    def delete(self, ids: Sequence[str] | None = None, where: dict | None = None) -> None:
        # Pinecone 404s when the namespace does not exist yet, but Chroma treats
        # deleting from an empty collection as a no-op. The app hits this whenever a
        # user removes their last document (or removes one before anything was ever
        # indexed), so surfacing the 404 would turn an ordinary delete into a 500.
        # Deleting nothing from nothing is success.
        try:
            if ids:
                self._index.delete(ids=list(ids), namespace=self.name)
                return
            flt = translate_filter(where)
            if flt:
                self._index.delete(filter=flt, namespace=self.name)
                return
            self._index.delete(delete_all=True, namespace=self.name)
        except Exception as exc:
            if _is_missing_namespace(exc):
                logger.debug("pinecone: namespace %s absent, delete is a no-op", self.name)
                return
            raise

    # ── reads ─────────────────────────────────────────────────────────────────

    def get(
        self,
        ids: Sequence[str] | None = None,
        where: dict | None = None,
        include: Iterable[str] | None = None,
        limit: int | None = None,
    ) -> dict:
        include = list(include or ["documents", "metadatas"])

        if ids:
            res = self._index.fetch(ids=list(ids), namespace=self.name)
            vectors = _as_dict(res, "vectors") or {}
            rows = [(cid, v) for cid, v in vectors.items()]
        else:
            # Pinecone has no "scan by filter" read. A metadata-filtered query with a
            # zero vector returns matches without meaningful ranking, which is exactly
            # what a `get(where=...)` wants: membership, not order.
            flt = translate_filter(where)
            res = self._index.query(
                vector=[0.0] * EMBED_DIM,
                top_k=int(limit or 10000),
                namespace=self.name,
                filter=flt,
                include_metadata=True,
                include_values="embeddings" in include,
            )
            rows = [(m["id"], m) for m in _matches(res)]

        return self._shape(rows, include, with_distance=False)

    def query(
        self,
        query_embeddings: Sequence[Sequence[float]],
        n_results: int = 10,
        where: dict | None = None,
        include: Iterable[str] | None = None,
    ) -> dict:
        include = list(include or ["documents", "metadatas", "distances"])
        flt = translate_filter(where)
        out: dict[str, list] = {k: [] for k in ("ids", "documents", "metadatas", "distances")}

        for vec in query_embeddings:
            res = self._index.query(
                vector=[float(x) for x in vec],
                top_k=int(n_results),
                namespace=self.name,
                filter=flt,
                include_metadata=True,
                include_values="embeddings" in include,
            )
            rows = [(m["id"], m) for m in _matches(res)]
            shaped = self._shape(rows, include, with_distance=True)
            for key in out:
                out[key].append(shaped.get(key, []))

        # Chroma nests one list per query embedding and omits keys not requested.
        keys = ["ids"] + [k for k in ("documents", "metadatas", "distances") if k in include]
        return {k: out[k] for k in keys}

    def count(self) -> int:
        stats = self._index.describe_index_stats()
        namespaces = _as_dict(stats, "namespaces") or {}
        ns = namespaces.get(self.name) or {}
        return int(_as_dict(ns, "vector_count") or 0)

    # ── shaping ───────────────────────────────────────────────────────────────

    @staticmethod
    def _shape(rows, include: list[str], *, with_distance: bool) -> dict:
        out: dict[str, list] = {"ids": [cid for cid, _ in rows]}
        metas = [dict(_as_dict(v, "metadata") or {}) for _, v in rows]
        if "documents" in include:
            out["documents"] = [m.get(DOC_KEY, "") for m in metas]
        if "metadatas" in include:
            # Strip the smuggled document back out so callers see Chroma's shape.
            out["metadatas"] = [{k: v for k, v in m.items() if k != DOC_KEY} for m in metas]
        if "embeddings" in include:
            out["embeddings"] = [list(_as_dict(v, "values") or []) for _, v in rows]
        if with_distance and "distances" in include:
            out["distances"] = [score_to_distance(_as_dict(v, "score") or 0.0) for _, v in rows]
        return out


def _as_dict(obj, key):
    """Pinecone's SDK returns objects that are sometimes dict-like, sometimes
    attribute-based, depending on version. Read either without branching at call sites."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _is_missing_namespace(exc: Exception) -> bool:
    """True when Pinecone reports a namespace that was never created.

    Matched on message rather than type so the adapter does not depend on which
    exception class a given SDK version raises for a 404.
    """
    text = str(exc).lower()
    return "namespace not found" in text or ("404" in text and "namespace" in text)


def _matches(res) -> list:
    m = _as_dict(res, "matches")
    return list(m or [])


class PineconeClient:
    """Chroma-client-compatible factory over one Pinecone index.

    Namespaces provide per-user isolation, so a single index serves every user —
    which matters on the free tier, where you get exactly one index.
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        *,
        dim: int = EMBED_DIM,
        cloud: str = "aws",
        region: str = "us-east-1",
        client=None,          # injectable for tests
    ):
        if not api_key:
            raise ValueError("PINECONE_API_KEY is required for VECTOR_BACKEND=pinecone")
        self._index_name = index_name
        self._dim = dim
        if client is not None:
            self._pc = client
        else:
            from pinecone import Pinecone  # lazy: not a base requirement
            self._pc = Pinecone(api_key=api_key)
        self._ensure_index(cloud, region)
        self._index = self._pc.Index(index_name)
        logger.info("PineconeClient ready [index=%s dim=%d]", index_name, dim)

    def _ensure_index(self, cloud: str, region: str) -> None:
        try:
            existing = {i["name"] for i in _index_list(self._pc.list_indexes())}
        except Exception:  # pragma: no cover - SDK shape varies
            existing = set()
        if self._index_name in existing:
            return
        from pinecone import ServerlessSpec

        # metric="cosine" is deliberate: it matches the pgvector HNSW
        # (vector_cosine_ops) and Chroma's default, so ranks stay comparable.
        self._pc.create_index(
            name=self._index_name,
            dimension=self._dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        logger.info("Created Pinecone index %s", self._index_name)

    # Chroma client API surface used by the app.
    def get_collection(self, name: str) -> PineconeCollection:
        return PineconeCollection(self._index, name)

    def get_or_create_collection(self, name: str, **_kw) -> PineconeCollection:
        return PineconeCollection(self._index, name)

    def delete_collection(self, name: str) -> None:
        PineconeCollection(self._index, name).delete()

    def heartbeat(self) -> int:
        self._index.describe_index_stats()
        return 1


def _index_list(raw) -> list:
    """list_indexes() returns a list in older SDKs and an object with .indexes in newer."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return [i if isinstance(i, dict) else {"name": getattr(i, "name", str(i))} for i in raw]
    inner = getattr(raw, "indexes", None) or []
    return [i if isinstance(i, dict) else {"name": getattr(i, "name", str(i))} for i in inner]
