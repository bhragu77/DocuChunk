"""
Pinecone adapter tests — hermetic, no network, no pinecone package required.

The adapter's job is TRANSLATION between our Chroma-shaped call sites and Pinecone's
API. Every test here targets a translation that, if wrong, leaves the system working
and the numbers plausible while silently invalidating the backend benchmark:

  * namespace isolation  — one user's chunks leaking into another's results
  * `$eq` filter form    — a filter that stops filtering
  * score -> distance    — dense ranks inverted, so RRF fuses a correct BM25 list
                           with a backwards dense list
  * response shapes      — Chroma's nested [[...]] lists that retrieval.py indexes
                           as raw["ids"][0]
"""
from __future__ import annotations

import pytest

from app.pipeline.pinecone_store import (
    DOC_KEY,
    PineconeClient,
    PineconeCollection,
    score_to_distance,
    translate_filter,
)


# ── Fake Pinecone index (records calls, returns SDK-shaped payloads) ──────────

class FakeIndex:
    def __init__(self):
        self.store: dict[str, dict[str, dict]] = {}   # namespace -> id -> vector
        self.calls: list[tuple] = []

    def upsert(self, vectors, namespace):
        self.calls.append(("upsert", namespace, len(vectors)))
        ns = self.store.setdefault(namespace, {})
        for v in vectors:
            ns[v["id"]] = v

    def delete(self, ids=None, filter=None, namespace=None, delete_all=False):
        self.calls.append(("delete", namespace, ids, filter, delete_all))
        ns = self.store.setdefault(namespace, {})
        if delete_all:
            ns.clear()
        elif ids:
            for i in ids:
                ns.pop(i, None)

    def fetch(self, ids, namespace):
        ns = self.store.get(namespace, {})
        return {"vectors": {i: ns[i] for i in ids if i in ns}}

    def query(self, vector, top_k, namespace, filter=None, include_metadata=True,
              include_values=False):
        self.calls.append(("query", namespace, top_k, filter))
        ns = self.store.get(namespace, {})
        matches = []
        for cid, v in ns.items():
            md = v.get("metadata", {})
            if filter:
                ok = all(md.get(k) == cond.get("$eq") for k, cond in filter.items())
                if not ok:
                    continue
            # Deterministic descending similarity so ordering is assertable.
            matches.append({"id": cid, "score": 0.9 - 0.1 * len(matches), "metadata": md,
                            "values": v.get("values", [])})
        return {"matches": matches[:top_k]}

    def describe_index_stats(self):
        return {"namespaces": {ns: {"vector_count": len(v)} for ns, v in self.store.items()}}


def _coll(index=None, namespace="user_1") -> PineconeCollection:
    return PineconeCollection(index or FakeIndex(), namespace)


# ── Filter translation ───────────────────────────────────────────────────────

def test_scalar_filter_becomes_eq_form():
    assert translate_filter({"doc_id": "d1"}) == {"doc_id": {"$eq": "d1"}}


def test_empty_filter_is_none():
    assert translate_filter(None) is None
    assert translate_filter({}) is None


def test_operator_form_passes_through():
    assert translate_filter({"page": {"$gte": 3}}) == {"page": {"$gte": 3}}


def test_list_filter_raises_rather_than_silently_matching_everything():
    """A filter that quietly stops filtering would leak another user's chunks."""
    with pytest.raises(ValueError):
        translate_filter({"doc_id": ["a", "b"]})


# ── Score -> distance (the RRF-correctness invariant) ─────────────────────────

def test_score_to_distance_inverts_similarity():
    assert score_to_distance(1.0) == 0.0     # identical -> zero distance
    assert score_to_distance(0.0) == 1.0
    assert score_to_distance(0.5) == 0.5


def test_distances_ascend_while_similarity_descends():
    """Retrieval ranks ASCENDING by distance. Pinecone returns similarity DESCENDING.
    If the conversion were skipped, the dense leg's order would invert and RRF would
    fuse a correct BM25 list with a backwards dense list."""
    idx = FakeIndex()
    c = _coll(idx)
    c.upsert(["a", "b", "c"], [[0.1] * 384] * 3, ["A", "B", "C"], [{}, {}, {}])
    out = c.query([[0.1] * 384], n_results=3)
    dists = out["distances"][0]
    assert dists == sorted(dists), "distances must ascend for best-first ordering"


# ── Chroma response shapes ───────────────────────────────────────────────────

def test_query_returns_chroma_nested_lists():
    """retrieval.py indexes raw["ids"][0] — one nested list per query embedding."""
    idx = FakeIndex()
    c = _coll(idx)
    c.upsert(["a"], [[0.1] * 384], ["text A"], [{"doc_id": "d1"}])
    out = c.query([[0.1] * 384], n_results=5)
    assert out["ids"] == [["a"]]
    assert out["documents"] == [["text A"]]
    assert out["metadatas"] == [[{"doc_id": "d1"}]]
    assert len(out["distances"][0]) == 1


def test_query_omits_keys_not_included():
    idx = FakeIndex()
    c = _coll(idx)
    c.upsert(["a"], [[0.1] * 384], ["t"], [{}])
    out = c.query([[0.1] * 384], n_results=1, include=["documents"])
    assert "documents" in out and "ids" in out
    assert "distances" not in out and "metadatas" not in out


def test_document_is_stripped_out_of_metadata_on_read():
    """The chunk text rides in metadata (Pinecone has no document field) but callers
    must see Chroma's shape, not our smuggling."""
    idx = FakeIndex()
    c = _coll(idx)
    c.upsert(["a"], [[0.1] * 384], ["hello"], [{"doc_id": "d1"}])
    out = c.query([[0.1] * 384], n_results=1)
    assert out["documents"][0][0] == "hello"
    assert DOC_KEY not in out["metadatas"][0][0]
    assert out["metadatas"][0][0] == {"doc_id": "d1"}


def test_get_by_ids_returns_flat_lists():
    idx = FakeIndex()
    c = _coll(idx)
    c.upsert(["a", "b"], [[0.1] * 384] * 2, ["A", "B"], [{}, {}])
    out = c.get(ids=["a"])
    assert out["ids"] == ["a"]          # get() is flat, unlike query()
    assert out["documents"] == ["A"]


def test_get_by_where_filters():
    idx = FakeIndex()
    c = _coll(idx)
    c.upsert(["a", "b"], [[0.1] * 384] * 2, ["A", "B"],
             [{"doc_id": "d1"}, {"doc_id": "d2"}])
    out = c.get(where={"doc_id": "d1"})
    assert out["ids"] == ["a"]


# ── Namespace isolation ──────────────────────────────────────────────────────

def test_namespaces_isolate_users():
    idx = FakeIndex()
    a, b = _coll(idx, "user_a"), _coll(idx, "user_b")
    a.upsert(["x"], [[0.1] * 384], ["A doc"], [{}])
    b.upsert(["y"], [[0.2] * 384], ["B doc"], [{}])
    assert a.query([[0.1] * 384], n_results=10)["ids"] == [["x"]]
    assert b.query([[0.1] * 384], n_results=10)["ids"] == [["y"]]


def test_delete_by_ids_scoped_to_namespace():
    idx = FakeIndex()
    a, b = _coll(idx, "user_a"), _coll(idx, "user_b")
    a.upsert(["x"], [[0.1] * 384], ["A"], [{}])
    b.upsert(["x"], [[0.1] * 384], ["B"], [{}])
    a.delete(ids=["x"])
    assert a.count() == 0
    assert b.count() == 1, "deleting in one namespace must not touch another"


# ── Metadata hygiene ─────────────────────────────────────────────────────────

def test_none_and_complex_metadata_values_are_sanitised():
    """Pinecone rejects the WHOLE upsert on one bad metadata value, which would lose
    an entire batch of chunks to a single null field."""
    idx = FakeIndex()
    c = _coll(idx)
    c.upsert(["a"], [[0.1] * 384], ["t"],
             [{"good": "x", "num": 3, "nothing": None, "nested": {"k": "v"}}])
    md = idx.store["user_1"]["a"]["metadata"]
    assert md["good"] == "x" and md["num"] == 3
    assert "nothing" not in md              # dropped, not sent as null
    assert isinstance(md["nested"], str)    # coerced, not rejected


def test_upsert_batches_large_writes():
    idx = FakeIndex()
    c = _coll(idx)
    n = 250
    c.upsert([f"c{i}" for i in range(n)], [[0.1] * 384] * n,
             ["t"] * n, [{}] * n)
    upserts = [call for call in idx.calls if call[0] == "upsert"]
    assert len(upserts) == 3                      # 100 + 100 + 50
    assert all(call[2] <= 100 for call in upserts)
    assert c.count() == n


def test_empty_upsert_is_a_noop():
    idx = FakeIndex()
    _coll(idx).upsert([], [], [], [])
    assert idx.calls == []


# ── Client surface ───────────────────────────────────────────────────────────

class FakePC:
    def __init__(self, index):
        self._index = index
        self.created = []

    def list_indexes(self):
        return [{"name": "docuchunk"}]

    def create_index(self, **kw):
        self.created.append(kw)

    def Index(self, name):
        return self._index


def test_client_exposes_chroma_surface_and_reuses_existing_index():
    idx = FakeIndex()
    pc = FakePC(idx)
    client = PineconeClient(api_key="k", index_name="docuchunk", client=pc)
    assert client.heartbeat() == 1
    assert isinstance(client.get_collection("user_1"), PineconeCollection)
    assert isinstance(client.get_or_create_collection("user_1"), PineconeCollection)
    assert pc.created == [], "must not recreate an index that already exists"


def test_client_requires_an_api_key():
    with pytest.raises(ValueError):
        PineconeClient(api_key="", index_name="x", client=FakePC(FakeIndex()))
