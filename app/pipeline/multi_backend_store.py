"""
Fan-out vector client — one logical collection over several physical backends.

WHY THIS EXISTS
===============
Once a document picks its own vector store at upload, a user's corpus can be split:
some documents in pgvector, others in Pinecone. A search scoped to one document can
be routed to that document's backend, but a search across ALL documents cannot —
it has to ask every backend and merge.

Rather than teach `hybrid_search` about multiple stores, this presents the SAME
collection interface the single-backend adapters expose (`query` / `get` / `upsert`
/ `delete` / `count`) and fans out internally. Retrieval, reranking, RRF fusion and
every eval harness stay exactly as they are; they cannot tell the difference.

THE MERGE
---------
`query` asks each backend for `n_results`, concatenates, sorts by DISTANCE ascending
and truncates. That is only valid because every backend is configured with the same
metric (cosine) and the same embedding model, so distances are directly comparable —
the invariant `eval/BACKEND_COMPARISON.md` exists to verify. If a backend were ever
added with a different metric, this merge would silently produce a wrong order, so
the constraint is enforced by convention and documented at every adapter.

WRITES ARE NOT FANNED OUT
-------------------------
`upsert` raises. A document belongs to exactly one backend, and duplicating its
vectors everywhere would double storage, double cost and make deletion ambiguous.
Ingestion always resolves a single concrete client through the registry; this class
is a READ path.
"""
from __future__ import annotations

import logging
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)


class MultiBackendCollection:
    """Read-only view over the same-named collection in several backends."""

    def __init__(self, collections: list, names: list[str]):
        self._collections = collections
        self._names = names
        self.name = collections[0].name if collections else ""

    # ── reads ─────────────────────────────────────────────────────────────────

    def query(
        self,
        query_embeddings: Sequence[Sequence[float]],
        n_results: int = 10,
        where: dict | None = None,
        include: Iterable[str] | None = None,
    ) -> dict:
        include = list(include or ["documents", "metadatas", "distances"])
        # Distances are needed to merge even if the caller did not ask for them;
        # they are dropped again below if unrequested.
        fetch = list(dict.fromkeys([*include, "distances"]))

        merged: dict[str, list] = {k: [] for k in ("ids", "documents", "metadatas", "distances")}

        for qi in range(len(query_embeddings)):
            rows: list[tuple] = []
            for coll, name in zip(self._collections, self._names):
                try:
                    raw = coll.query(
                        query_embeddings=[query_embeddings[qi]],
                        n_results=n_results, where=where, include=fetch,
                    )
                except Exception:
                    # One backend being unreachable must not blank the whole search;
                    # returning the other backend's hits is strictly better than none.
                    logger.warning("multi-backend query failed on %s, skipping", name,
                                   exc_info=True)
                    continue
                ids = (raw.get("ids") or [[]])[0]
                docs = (raw.get("documents") or [[]])[0] if "documents" in fetch else [""] * len(ids)
                metas = (raw.get("metadatas") or [[]])[0] if "metadatas" in fetch else [{}] * len(ids)
                dists = (raw.get("distances") or [[]])[0]
                for i, cid in enumerate(ids):
                    rows.append((
                        float(dists[i]) if i < len(dists) else float("inf"),
                        cid,
                        docs[i] if i < len(docs) else "",
                        metas[i] if i < len(metas) else {},
                    ))

            rows.sort(key=lambda r: r[0])
            rows = rows[:n_results]
            merged["ids"].append([r[1] for r in rows])
            merged["documents"].append([r[2] for r in rows])
            merged["metadatas"].append([r[3] for r in rows])
            merged["distances"].append([r[0] for r in rows])

        keys = ["ids"] + [k for k in ("documents", "metadatas", "distances") if k in include]
        return {k: merged[k] for k in keys}

    def get(
        self,
        ids: Sequence[str] | None = None,
        where: dict | None = None,
        include: Iterable[str] | None = None,
        limit: int | None = None,
    ) -> dict:
        out: dict[str, list] = {"ids": []}
        for coll, name in zip(self._collections, self._names):
            try:
                part = coll.get(ids=ids, where=where, include=include, limit=limit)
            except Exception:
                logger.warning("multi-backend get failed on %s, skipping", name, exc_info=True)
                continue
            for key, vals in part.items():
                out.setdefault(key, []).extend(vals)
        return out

    def count(self) -> int:
        total = 0
        for coll in self._collections:
            try:
                total += int(coll.count())
            except Exception:
                logger.debug("count failed on one backend", exc_info=True)
        return total

    # ── writes ────────────────────────────────────────────────────────────────

    def upsert(self, *a, **kw):
        raise RuntimeError(
            "MultiBackendCollection is read-only. A document belongs to exactly one "
            "backend; resolve a concrete client via vector_registry.get_client()."
        )

    def delete(self, *a, **kw):
        raise RuntimeError(
            "MultiBackendCollection is read-only. Delete through the document's own "
            "backend, or vectors would be orphaned in the others."
        )


class MultiBackendClient:
    """Client facade returning fan-out collections."""

    def __init__(self, clients: list, names: list[str]):
        self._clients = clients
        self._names = names

    def _collect(self, name: str, create: bool):
        colls, live = [], []
        for client, backend in zip(self._clients, self._names):
            try:
                coll = (client.get_or_create_collection(name) if create
                        else client.get_collection(name))
            except Exception:
                # A backend with no collection for this user simply contributes
                # nothing; that is normal, not an error.
                logger.debug("no collection %s on %s", name, backend)
                continue
            colls.append(coll)
            live.append(backend)
        return MultiBackendCollection(colls, live)

    def get_collection(self, name: str):
        return self._collect(name, create=False)

    def get_or_create_collection(self, name: str, **_kw):
        return self._collect(name, create=True)

    def heartbeat(self) -> int:
        for client in self._clients:
            client.heartbeat()
        return 1


def build_read_client(backends: list[str]):
    """A client that reads across `backends`.

    Returns the single concrete client when only one backend is involved, so the
    common case pays no fan-out overhead at all.
    """
    from app.core.vector_registry import get_client, normalise

    names = list(dict.fromkeys(normalise(b) for b in backends)) or ["pgvector"]
    if len(names) == 1:
        return get_client(names[0])
    return MultiBackendClient([get_client(n) for n in names], names)
