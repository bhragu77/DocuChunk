"""
Phase 8 — BM25 index + BM25↔Chroma sync tests.

Real rank_bm25 index in a temp dir + real persistent Chroma (no server, no Redis).
Coverage:
  * tokenizer keeps identifiers usefully matchable
  * upsert / query ranking / doc_id filter / positive-score gate
  * per-user isolation and cross-instance persistence (worker writes → web reads)
  * whole-doc delete and specific-chunk delete
  * THE GHOST TEST (BM25 edition): reconcile_document tombstones a removed chunk from
    BOTH Chroma and BM25 — a ghost never lingers in the lexical index
  * hybrid_search: dense alone MISSES an exact-identifier query, but BM25 + RRF
    surfaces the right chunk inside top-k
"""
import chromadb
import pytest

from app.pipeline.bm25_index import BM25Index, _tokenize
from app.pipeline.types import Chunk, EmbeddedChunk
from app.pipeline.chunker import make_chunk_id, make_content_hash
from app.pipeline.vector_store import (
    batch_upsert_sink,
    get_collection,
    reconcile_document,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk(i, doc_id="docV", text=None):
    text = text if text is not None else f"chunk number {i} content"
    start, end = i * 1000, i * 1000 + 500
    return Chunk(
        chunk_id=make_chunk_id(doc_id, start, end),
        doc_id=doc_id, text=text, chunk_index=i, page_number=i + 1,
        char_start=start, char_end=end, token_count=len(text) // 4,
        content_hash=make_content_hash(text), doc_version=1,
        metadata={"strategy": "sentence", "source": "f.pdf"},
    )


@pytest.fixture
def bm25(tmp_path):
    return BM25Index(persist_dir=str(tmp_path / "bm25"))


@pytest.fixture
def chroma_client(tmp_path):
    return chromadb.PersistentClient(path=str(tmp_path / "chroma"))


# ── Tokenizer ─────────────────────────────────────────────────────────────────

def test_tokenizer_lowercases_and_keeps_identifiers():
    assert _tokenize("IP67 rating") == ["ip67", "rating"]
    assert _tokenize("Error code E4521 occurred") == ["error", "code", "e4521", "occurred"]
    # A hyphenated part number splits into its alphanumeric runs — but a query for the
    # same string splits identically, so the match still lands.
    assert _tokenize("88-AZ-0097") == ["88", "az", "0097"]
    assert _tokenize("88-AZ-0097") == _tokenize("part 88-AZ-0097")[1:]


# ── Basic upsert / query ──────────────────────────────────────────────────────

def test_upsert_and_query_ranks_lexical_match_first(bm25):
    bm25.upsert("c1", "The battery part number is 88-AZ-0097 for the GX-4200.", "docA", "u1")
    bm25.upsert("c2", "The device operates between minus ten and fifty degrees.", "docA", "u1")
    bm25.upsert("c3", "Firmware upgrades are delivered over the air automatically.", "docA", "u1")

    res = bm25.query("battery part number 88-AZ-0097", "u1", top_n=5)
    assert res, "expected at least one BM25 hit"
    assert res[0][0] == "c1"
    # Only chunks with actual term overlap are returned (positive-score gate).
    assert all(score > 0 for _, score in res)


def test_query_doc_id_filter(bm25):
    bm25.upsert("a1", "shared keyword alpha content", "docA", "u1")
    bm25.upsert("b1", "shared keyword alpha content", "docB", "u1")

    all_docs = {cid for cid, _ in bm25.query("alpha keyword", "u1", top_n=10)}
    assert all_docs == {"a1", "b1"}

    only_a = {cid for cid, _ in bm25.query("alpha keyword", "u1", doc_id="docA", top_n=10)}
    assert only_a == {"a1"}


def test_query_empty_or_no_overlap_returns_empty(bm25):
    bm25.upsert("c1", "completely unrelated content here", "docA", "u1")
    assert bm25.query("", "u1") == []
    assert bm25.query("zzzznonexistenttoken", "u1") == []


def test_user_isolation(bm25):
    bm25.upsert("u1c1", "quarterly revenue numbers", "docA", "u1")
    bm25.upsert("u2c1", "quarterly revenue numbers", "docA", "u2")
    res = {cid for cid, _ in bm25.query("quarterly revenue", "u1", top_n=10)}
    assert res == {"u1c1"}   # u2's chunk never appears in u1's results


# ── Persistence across instances (worker writes → web process reads) ──────────

def test_persistence_across_instances(tmp_path):
    d = str(tmp_path / "bm25")
    writer = BM25Index(persist_dir=d)
    writer.upsert("c1", "persisted lexical content token", "docA", "u1")

    reader = BM25Index(persist_dir=d)   # separate instance, same dir
    assert reader.contains("c1", "u1")
    assert reader.count("u1") == 1
    assert reader.query("lexical token", "u1")[0][0] == "c1"


def test_mtime_cache_invalidation_sees_new_writes(tmp_path):
    """A reader that already queried once must still see a later write (mtime cache)."""
    d = str(tmp_path / "bm25")
    idx = BM25Index(persist_dir=d)
    idx.upsert("c1", "first content alpha", "docA", "u1")
    assert idx.query("alpha", "u1")           # warms the cache
    idx.upsert("c2", "second content beta", "docA", "u1")
    assert idx.contains("c2", "u1")
    assert idx.query("beta", "u1")[0][0] == "c2"


# ── Deletes ───────────────────────────────────────────────────────────────────

def test_delete_removes_whole_doc(bm25):
    bm25.upsert("a1", "doc a content one", "docA", "u1")
    bm25.upsert("a2", "doc a content two", "docA", "u1")
    bm25.upsert("b1", "doc b content", "docB", "u1")

    removed = bm25.delete("docA", "u1")
    assert removed == 2
    assert not bm25.contains("a1", "u1")
    assert not bm25.contains("a2", "u1")
    assert bm25.contains("b1", "u1")


def test_delete_chunks_removes_specific_ids(bm25):
    bm25.upsert("c1", "content one", "docA", "u1")
    bm25.upsert("c2", "content two", "docA", "u1")
    bm25.upsert("c3", "content three", "docA", "u1")

    removed = bm25.delete_chunks(["c2"], "u1")
    assert removed == 1
    assert bm25.contains("c1", "u1")
    assert not bm25.contains("c2", "u1")
    assert bm25.contains("c3", "u1")


# ── Sink co-write ─────────────────────────────────────────────────────────────

def test_batch_sink_cowrites_bm25(bm25, chroma_client):
    coll = get_collection(chroma_client, "u1")
    chunks = [_chunk(i) for i in range(3)]
    ecs = [EmbeddedChunk(chunk=c, embedding=[0.1] * 384) for c in chunks]

    sink = batch_upsert_sink(coll, bm25=bm25, user_id="u1")
    sink(ecs)

    # Every chunk written to Chroma is also in the BM25 index, keyed on the same id.
    assert bm25.count("u1") == 3
    for c in chunks:
        assert bm25.contains(c.chunk_id, "u1")
    assert set(coll.get(where={"doc_id": "docV"})["ids"]) == {c.chunk_id for c in chunks}


# ── THE GHOST TEST (BM25 edition) ─────────────────────────────────────────────

def test_reconcile_tombstones_ghost_from_bm25_and_chroma(bm25, chroma_client):
    """A chunk dropped on re-chunk must vanish from BOTH the vector store and BM25 —
    otherwise BM25 keeps surfacing a dead chunk_id that resolves to no stored text."""
    coll = get_collection(chroma_client, "u1")
    sink = batch_upsert_sink(coll, bm25=bm25, user_id="u1")

    v1 = [_chunk(i) for i in range(3)]
    sink([EmbeddedChunk(chunk=c, embedding=[0.1] * 384) for c in v1])
    ghost_id = v1[2].chunk_id
    assert bm25.contains(ghost_id, "u1")

    # Re-chunk with #2 removed → reconcile must tombstone the ghost from both stores.
    v2 = [_chunk(0), _chunk(1)]
    report = reconcile_document(coll, "docV", v2, new_version=2, bm25=bm25, user_id="u1")

    assert report.to_delete_ids == {ghost_id}
    assert coll.get(ids=[ghost_id])["ids"] == []      # gone from Chroma
    assert not bm25.contains(ghost_id, "u1")          # gone from BM25 (the ghost edition)
    assert bm25.contains(v2[0].chunk_id, "u1")
    assert bm25.contains(v2[1].chunk_id, "u1")


# ── Hybrid: BM25 rescues an exact-identifier query dense misses ───────────────

def test_hybrid_bm25_surfaces_identifier_dense_misses(bm25, chroma_client):
    """
    Dense retrieval MISSES the exact-identifier chunk (its vector is pushed out of the
    dense top-k by distractors), but BM25 matches the identifier token exactly and RRF
    pulls it back into the fused top-k. This is the core Phase 8 identifier win.
    """
    from app.pipeline.retrieval import hybrid_search

    coll = get_collection(chroma_client, "u1")
    DIM = 16

    def unit(idx):
        v = [0.0] * DIM
        v[idx] = 1.0
        return v

    # The identifier chunk sits on axis 0; the query sits on axis 1 (orthogonal), so
    # cosine similarity to the identifier is ~0. Six distractors sit near the query.
    ident_id, ident_text = "cid_ident", "Battery part number 88-AZ-0097 replacement pack."
    coll.upsert(
        ids=[ident_id], embeddings=[unit(0)], documents=[ident_text],
        metadatas=[{"doc_id": "docA", "source": "manual.pdf", "page_number": 4}],
    )
    for i in range(6):
        coll.upsert(
            ids=[f"distract_{i}"], embeddings=[unit(1)],
            documents=[f"General maintenance note {i}."],
            metadatas=[{"doc_id": "docA", "source": "manual.pdf", "page_number": i}],
        )

    # BM25 knows all chunks; only the identifier chunk carries the part-number tokens.
    bm25.upsert(ident_id, ident_text, "docA", "u1")
    for i in range(6):
        bm25.upsert(f"distract_{i}", f"General maintenance note {i}.", "docA", "u1")

    query = "battery part number 88-AZ-0097"
    # Sanity: dense alone (top-5) does NOT contain the identifier chunk.
    dense_only = coll.query(query_embeddings=[unit(1)], n_results=5)["ids"][0]
    assert ident_id not in dense_only

    results = hybrid_search(
        query=query, user_id="u1", chroma_client=chroma_client,
        bm25_index=bm25, embed_fn=lambda q: unit(1), top_n=5,
    )
    top_ids = [r.chunk_id for r in results]
    assert ident_id in top_ids, f"BM25+RRF failed to surface the identifier chunk: {top_ids}"
    ident = next(r for r in results if r.chunk_id == ident_id)
    assert ident.bm25_rank == 1          # BM25 ranked it first
    assert ident.dense_rank is None      # dense never retrieved it
