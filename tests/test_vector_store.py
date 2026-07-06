"""
Part 2 — Vector Store tests.

Real persistent Chroma in a temp dir (no server, no Redis, no mocks). A deterministic
FakeSTModel/SpyProvider gives stable 384-d vectors so the store is exercised end to
end. Every test builds its own throwaway Chroma client so per-user collections never
leak across tests.

Coverage mirrors the incremental-index contract:
  * first upload            → everything upserted, metadata correct
  * identical re-upload     → all unchanged, nothing embedded (the skip optimization)
  * one chunk changed       → only that chunk re-embedded (content_hash differs)
  * chunk removed           → THE GHOST TEST: old chunk_id tombstoned + unretrievable
  * chunk added             → new chunk_id upserted, others untouched
  * batch cap               → oversized sink batch split into multiple upsert calls
  * whole-doc delete        → the exact op the DELETE /docs/{id} endpoint performs
  * nested resume           → mid-embed crash resumes into the upsert sink, no dups
  * end-to-end run_pipeline → upload → pipeline → queryable store with embeddings
"""
import hashlib

import chromadb
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.database import Base
from app.models.chunk import ChunkRecord
from app.models.document import DocumentStatus
from app.models.job import EmbeddingJob, JobStatus
from app.pipeline.chunker import make_chunk_id, make_content_hash
from app.pipeline.embedder import embed_document, resume_embedding
from app.pipeline.orchestrator import run_pipeline
from app.pipeline.types import Chunk, EmbeddedChunk
from app.pipeline.vector_store import (
    ReconcileReport,
    batch_upsert_sink,
    collection_name,
    get_collection,
    reconcile_document,
)


# ── Fakes / helpers ───────────────────────────────────────────────────────────

class FakeSTModel:
    """Deterministic fake ST model: identical text → identical 384-d vector."""
    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts):
        rows = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            rows.append(rng.normal(size=self.dim))
        return np.array(rows, dtype=np.float32)


class SpyProvider:
    """Records texts per embed() call; returns L2-normalized deterministic vectors."""
    provider_name = "local"

    def __init__(self, model_name="fake-model", normalize=True):
        self.model_name = model_name
        self.normalize = normalize
        self.calls: list[list[str]] = []
        self._model = FakeSTModel()

    def embed(self, texts):
        self.calls.append(list(texts))
        raw = self._model.encode(texts)
        out = []
        for row in raw:
            v = np.asarray(row, dtype=np.float32)
            if self.normalize:
                n = np.linalg.norm(v)
                v = v / n if n else v
            out.append(v.tolist())
        return out


def make_chunk(i, doc_id="docV", text=None, version=1):
    """
    Chunk with a FIXED char span per index, so chunk_id (a hash of doc_id + span)
    is stable regardless of the text. That lets a test change a chunk's text while
    keeping its id — the exact "content_hash differs, same chunk_id" case reconcile
    must catch. New index i → new span → new id (the add/remove cases).
    """
    text = text if text is not None else f"chunk number {i} content"
    start, end = i * 1000, i * 1000 + 500
    return Chunk(
        chunk_id=make_chunk_id(doc_id, start, end),
        doc_id=doc_id, text=text, chunk_index=i, page_number=i + 1,
        char_start=start, char_end=end, token_count=len(text) // 4,
        content_hash=make_content_hash(text), doc_version=version,
        metadata={"strategy": "sentence", "source": "f.pdf", "params": {}},
    )


@pytest.fixture
def chroma_client(tmp_path):
    """A fresh persistent Chroma client against a throwaway temp dir."""
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    yield client


@pytest.fixture
def db_factory(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path/'vs.db'}", connect_args={"check_same_thread": False}
    )
    Session = sessionmaker(bind=engine)
    from app.models import user, document, job, chunk  # noqa: F401
    Base.metadata.create_all(bind=engine)
    try:
        yield Session
    finally:
        engine.dispose()


def _embed(chunks, provider, collection, max_batch=None, job=None):
    """Embed `chunks` straight into the collection's batch-upsert sink."""
    doc_id = chunks[0].doc_id if chunks else "docV"
    job = job or EmbeddingJob(doc_id=doc_id, doc_version=1, status=JobStatus.pending)
    sink = batch_upsert_sink(collection, max_batch=max_batch)
    return embed_document(chunks, doc_id, provider, sink, job)


def _insert_chunks(session, chunks):
    for c in chunks:
        session.add(ChunkRecord(
            id=c.chunk_id, doc_id=c.doc_id, chunk_index=c.chunk_index,
            page_number=c.page_number, text=c.text, char_start=c.char_start,
            char_end=c.char_end, token_count=c.token_count,
            content_hash=c.content_hash, doc_version=c.doc_version,
            chunk_metadata=c.metadata,
        ))
    session.commit()


def _insert_doc(session, doc_id, status, file_path="/nonexistent.pdf", version=1, user_id="u"):
    from app.models.document import Document
    session.add(Document(
        id=doc_id, user_id=user_id, filename="f.pdf", original_filename="f.pdf",
        file_path=file_path, file_size=123, mime_type="application/pdf",
        status=status, version=version,
    ))
    session.commit()


def _make_pdf(path, pages):
    import fitz
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((50, 72), text)
    doc.save(str(path))
    doc.close()


# ═══════════════════════════════════════════════════════════════════════════════
# First upload
# ═══════════════════════════════════════════════════════════════════════════════

def test_first_upload_all_upserted_with_metadata(chroma_client):
    coll = get_collection(chroma_client, "u")
    chunks = [make_chunk(i) for i in range(3)]

    report = reconcile_document(coll, "docV", chunks, new_version=1)
    assert report.to_upsert_ids == {c.chunk_id for c in chunks}
    assert report.to_delete_ids == set()
    assert report.unchanged_ids == set()

    _embed([c for c in chunks if c.chunk_id in report.to_upsert_ids], SpyProvider(), coll)

    got = coll.get(where={"doc_id": "docV"}, include=["metadatas", "embeddings"])
    assert set(got["ids"]) == {c.chunk_id for c in chunks}
    by_id = dict(zip(got["ids"], got["metadatas"]))
    for c in chunks:
        m = by_id[c.chunk_id]
        assert m["content_hash"] == c.content_hash
        assert m["doc_version"] == 1
        assert m["page_number"] == c.page_number
        assert m["doc_id"] == "docV"
        assert m["source"] == "f.pdf"
    assert all(len(e) == 384 for e in got["embeddings"])


# ═══════════════════════════════════════════════════════════════════════════════
# Re-upload: identical / changed / removed / added
# ═══════════════════════════════════════════════════════════════════════════════

def test_identical_reupload_all_unchanged_no_embed(chroma_client):
    coll = get_collection(chroma_client, "u")
    _embed([make_chunk(i) for i in range(3)], SpyProvider(), coll)

    new = [make_chunk(i, version=2) for i in range(3)]   # same text, bumped version
    report = reconcile_document(coll, "docV", new, new_version=2)
    assert report.unchanged_ids == {c.chunk_id for c in new}
    assert report.to_upsert_ids == set()
    assert report.to_delete_ids == set()

    spy = SpyProvider()
    to_upsert = [c for c in new if c.chunk_id in report.to_upsert_ids]
    _embed(to_upsert, spy, coll)
    assert spy.calls == []   # skip-unchanged optimization: embedder never invoked


def test_one_chunk_changed_only_it_reembedded(chroma_client):
    coll = get_collection(chroma_client, "u")
    _embed([make_chunk(i) for i in range(3)], SpyProvider(), coll)

    changed = make_chunk(1, text="COMPLETELY different content now", version=2)
    new = [make_chunk(0, version=2), changed, make_chunk(2, version=2)]
    report = reconcile_document(coll, "docV", new, new_version=2)
    assert report.to_upsert_ids == {changed.chunk_id}
    assert report.unchanged_ids == {new[0].chunk_id, new[2].chunk_id}
    assert report.to_delete_ids == set()

    spy = SpyProvider()
    _embed([c for c in new if c.chunk_id in report.to_upsert_ids], spy, coll)
    assert spy.calls == [[changed.text]]   # only the changed chunk embedded

    got = coll.get(ids=[changed.chunk_id], include=["metadatas", "documents"])
    assert got["metadatas"][0]["content_hash"] == changed.content_hash
    assert got["metadatas"][0]["doc_version"] == 2
    assert got["documents"][0] == changed.text


def test_removed_chunk_is_ghosted(chroma_client):
    """THE GHOST TEST — a removed chunk is tombstoned and becomes unretrievable."""
    coll = get_collection(chroma_client, "u")
    _embed([make_chunk(i) for i in range(3)], SpyProvider(), coll)
    removed_id = make_chunk(2).chunk_id

    new = [make_chunk(0, version=2), make_chunk(1, version=2)]   # #2 dropped
    report = reconcile_document(coll, "docV", new, new_version=2)
    assert report.to_delete_ids == {removed_id}

    assert coll.get(ids=[removed_id])["ids"] == []          # gone by direct id
    remaining = coll.get(where={"doc_id": "docV"})["ids"]
    assert removed_id not in remaining
    assert set(remaining) == {c.chunk_id for c in new}


def test_added_chunk_upserted_others_unchanged(chroma_client):
    coll = get_collection(chroma_client, "u")
    _embed([make_chunk(i) for i in range(2)], SpyProvider(), coll)

    added = make_chunk(2, version=2)
    new = [make_chunk(0, version=2), make_chunk(1, version=2), added]
    report = reconcile_document(coll, "docV", new, new_version=2)
    assert report.to_upsert_ids == {added.chunk_id}
    assert report.unchanged_ids == {new[0].chunk_id, new[1].chunk_id}
    assert report.to_delete_ids == set()

    _embed([added], SpyProvider(), coll)
    assert set(coll.get(where={"doc_id": "docV"})["ids"]) == {c.chunk_id for c in new}


# ═══════════════════════════════════════════════════════════════════════════════
# Batch cap
# ═══════════════════════════════════════════════════════════════════════════════

class _CountingCollection:
    """Proxy that counts upsert() calls, delegating get/delete to the real collection."""
    def __init__(self, coll):
        self._c = coll
        self.upsert_calls = 0

    def upsert(self, **kw):
        self.upsert_calls += 1
        return self._c.upsert(**kw)

    def get(self, *a, **k):
        return self._c.get(*a, **k)

    def delete(self, *a, **k):
        return self._c.delete(*a, **k)


def test_batch_cap_splits_into_multiple_upserts(chroma_client):
    real = get_collection(chroma_client, "u")
    counting = _CountingCollection(real)
    chunks = [make_chunk(i) for i in range(10)]
    ecs = [EmbeddedChunk(chunk=c, embedding=[0.1] * 384) for c in chunks]

    sink = batch_upsert_sink(counting, max_batch=3)
    sink(ecs)   # 10 records, cap 3 → ceil(10/3) = 4 upsert calls

    assert counting.upsert_calls == 4
    assert len(real.get(where={"doc_id": "docV"})["ids"]) == 10   # all landed


# ═══════════════════════════════════════════════════════════════════════════════
# Whole-document delete (the DELETE /docs/{doc_id} operation)
# ═══════════════════════════════════════════════════════════════════════════════

def test_delete_by_doc_removes_only_that_docs_chunks(chroma_client):
    from app.routers.documents import _chroma_collection_name

    coll = get_collection(chroma_client, "u")
    a = [make_chunk(i, doc_id="docA") for i in range(3)]
    b = [make_chunk(i, doc_id="docB") for i in range(2)]
    _embed(a, SpyProvider(), coll)
    _embed(b, SpyProvider(), coll)

    # Exactly what the DELETE /docs/{doc_id} endpoint runs on the per-user collection.
    coll.delete(where={"doc_id": "docA"})

    assert coll.get(where={"doc_id": "docA"})["ids"] == []
    assert set(coll.get(where={"doc_id": "docB"})["ids"]) == {c.chunk_id for c in b}
    # The router's collection name matches the vector store's convention.
    assert _chroma_collection_name("u") == coll.name == collection_name("u")


# ═══════════════════════════════════════════════════════════════════════════════
# Nested resume into the upsert sink (idempotency)
# ═══════════════════════════════════════════════════════════════════════════════

def test_nested_resume_into_upsert_sink_no_duplicates(db_factory, chroma_client, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "embed_batch_size", 1)
    monkeypatch.setattr(s, "embed_small_max", 0)
    monkeypatch.setattr(s, "embed_large_min", 1)   # LARGE → per-batch checkpoints

    coll = get_collection(chroma_client, "u")
    sess = db_factory()
    chunks = [make_chunk(i, doc_id="docR", text=f"r{i}") for i in range(5)]
    _insert_chunks(sess, chunks)
    job = EmbeddingJob(doc_id="docR", doc_version=1, status=JobStatus.pending)
    sess.add(job)
    sess.commit()

    provider = SpyProvider()
    real_sink = batch_upsert_sink(coll)
    n = {"c": 0}

    def crashing_sink(batch):
        n["c"] += 1
        real_sink(batch)                 # batch IS upserted...
        if n["c"] == 3:                  # ...then crash before checkpoint commit
            raise RuntimeError("crash after upsert, before checkpoint")

    with pytest.raises(RuntimeError):
        embed_document(chunks, "docR", provider, crashing_sink, job)

    sess.refresh(job)
    assert job.checkpoint_batch == 1     # batch 2 upserted but checkpoint not advanced

    # Resume into the SAME upsert sink: batch 2 is re-embedded and re-upserted, which
    # is harmless because upsert is idempotent-by-chunk_id.
    resume_embedding("docR", provider, real_sink, job)

    sess.refresh(job)
    assert job.status == JobStatus.completed
    assert job.checkpoint_batch == 4

    got = coll.get(where={"doc_id": "docR"})
    assert len(got["ids"]) == 5                              # no duplicates from re-sink
    assert sorted(got["ids"]) == sorted(c.chunk_id for c in chunks)
    sess.close()


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-end: upload → run_pipeline → searchable store
# ═══════════════════════════════════════════════════════════════════════════════

def test_end_to_end_pipeline_produces_searchable_store(db_factory, chroma_client, tmp_path):
    pdf = tmp_path / "e2e.pdf"
    _make_pdf(pdf, [
        "Alpha beta gamma. Delta epsilon zeta. Eta theta iota.",
        "Second page content here. More sentences follow. And even more text.",
    ])
    sess = db_factory()
    _insert_doc(sess, "docE", status=DocumentStatus.uploaded, file_path=str(pdf))
    sess.close()

    provider = SpyProvider()
    status = run_pipeline("docE", session_factory=db_factory, provider=provider, chroma=chroma_client)
    assert status == "ready"

    coll = get_collection(chroma_client, "u")   # doc user_id defaults to "u"
    got = coll.get(where={"doc_id": "docE"}, include=["embeddings", "documents", "metadatas"])
    assert len(got["ids"]) > 0
    assert all(len(e) == 384 for e in got["embeddings"])
    assert all(m["doc_id"] == "docE" for m in got["metadatas"])

    # The store is actually queryable: a stored chunk's own vector retrieves itself.
    q = provider.embed([got["documents"][0]])[0]
    res = coll.query(query_embeddings=[q], n_results=1)
    assert res["ids"][0][0] in got["ids"]


def test_end_to_end_reupload_reconciles_ghost(db_factory, chroma_client, tmp_path):
    """
    Full loop: first upload stores everything; a shortened re-upload (fewer chunks)
    tombstones the vanished chunk_ids in the live store via run_pipeline.
    """
    long_pdf = tmp_path / "v1.pdf"
    _make_pdf(long_pdf, [
        "Alpha beta gamma. Delta epsilon zeta. Eta theta iota. Kappa lambda mu.",
        "Nu xi omicron pi. Rho sigma tau upsilon. Phi chi psi omega done.",
    ])
    sess = db_factory()
    _insert_doc(sess, "docG", status=DocumentStatus.uploaded, file_path=str(long_pdf))
    sess.close()

    run_pipeline("docG", session_factory=db_factory, provider=SpyProvider(), chroma=chroma_client)
    coll = get_collection(chroma_client, "u")
    v1_ids = set(coll.get(where={"doc_id": "docG"})["ids"])
    assert len(v1_ids) > 1

    # Re-upload a shorter document: bump version, re-point the file, re-run the pipeline.
    short_pdf = tmp_path / "v2.pdf"
    _make_pdf(short_pdf, ["Alpha beta gamma. Delta epsilon zeta. Eta theta iota. Kappa lambda mu."])
    sess = db_factory()
    from app.models.document import Document
    doc = sess.query(Document).filter_by(id="docG").first()
    doc.version = 2
    doc.status = DocumentStatus.uploaded          # force a fresh run
    doc.file_path = str(short_pdf)
    sess.query(ChunkRecord).filter_by(doc_id="docG").delete()   # re-upload replaces chunk rows
    sess.commit()
    sess.close()

    run_pipeline("docG", session_factory=db_factory, provider=SpyProvider(), chroma=chroma_client)
    v2_ids = set(coll.get(where={"doc_id": "docG"})["ids"])

    # Ghosts from v1 that are absent in v2 must be gone from the live store.
    assert v2_ids and v2_ids != v1_ids
    ghosts = v1_ids - v2_ids
    assert ghosts, "expected the shortened re-upload to drop at least one chunk"
    for gid in ghosts:
        assert coll.get(ids=[gid])["ids"] == []
