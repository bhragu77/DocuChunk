"""
Part 1 — Embedder tests.

No network, no model download: a FakeSTModel returns deterministic 384-d vectors,
and the HF provider is driven by a stub client. DB-backed cases use a temp SQLite DB.
"""
import hashlib
import os

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.database import Base
from app.models.chunk import ChunkRecord
from app.models.job import EmbeddingJob, JobStatus
from app.pipeline.chunker import make_chunk_id, make_content_hash
from app.pipeline.types import Chunk, EmbeddedChunk
from app.pipeline.embedding_providers import (
    LocalSTProvider,
    HFAPIProvider,
    embed_query,
    build_provider,
)
from app.pipeline.embedder import (
    embed_document,
    resume_embedding,
    retry_failed_embeddings,
    _tier,
    SMALL, MEDIUM, LARGE,
)
from app.pipeline.orchestrator import run_pipeline


# ── Fakes / helpers ───────────────────────────────────────────────────────────

class FakeSTModel:
    """Deterministic fake sentence-transformers model. Returns NON-unit 384-d
    vectors (so normalization is observable). Identical text → identical vector."""
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
    """Records the texts passed to each embed() call. Optional per-text failure."""
    provider_name = "local"

    def __init__(self, model_name="fake-model", normalize=True, fail_texts=None):
        self.model_name = model_name
        self.normalize = normalize
        self.calls: list[list[str]] = []
        self.fail_texts = set(fail_texts or [])
        self._model = FakeSTModel()

    def embed(self, texts):
        self.calls.append(list(texts))
        raw = self._model.encode(texts)
        out = []
        for t, row in zip(texts, raw):
            if t in self.fail_texts:
                out.append(None)
                continue
            v = np.asarray(row, dtype=np.float32)
            if self.normalize:
                n = np.linalg.norm(v)
                v = v / n if n else v
            out.append(v.tolist())
        return out


def make_chunk(i, doc_id="doc1", text=None, doc_version=1):
    text = text if text is not None else f"chunk number {i} content"
    start, end = i * 1000, i * 1000 + len(text)
    return Chunk(
        chunk_id=make_chunk_id(doc_id, start, end),
        doc_id=doc_id, text=text, chunk_index=i, page_number=1,
        char_start=start, char_end=end, token_count=len(text) // 4,
        content_hash=make_content_hash(text), doc_version=doc_version,
    )


@pytest.fixture
def db_factory(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path/'emb.db'}", connect_args={"check_same_thread": False}
    )
    Session = sessionmaker(bind=engine)
    from app.models import user, document, job, chunk  # noqa: F401
    Base.metadata.create_all(bind=engine)
    try:
        yield Session
    finally:
        engine.dispose()


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


def _insert_doc(session, doc_id, status, file_path="/nonexistent.pdf", version=1):
    from app.models.document import Document
    size = os.path.getsize(file_path) if os.path.exists(file_path) else 123
    session.add(Document(
        id=doc_id, user_id="u", filename="f.pdf", original_filename="f.pdf",
        file_path=file_path, file_size=size, mime_type="application/pdf",
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
# Providers
# ═══════════════════════════════════════════════════════════════════════════════

def test_local_normalize_true_gives_unit_vectors():
    p = LocalSTProvider(FakeSTModel(), "m", normalize=True)
    vecs = p.embed(["alpha", "beta gamma"])
    for v in vecs:
        assert v is not None
        assert abs(np.linalg.norm(v) - 1.0) < 1e-5


def test_local_normalize_false_not_unit():
    p = LocalSTProvider(FakeSTModel(), "m", normalize=False)
    v = p.embed(["alpha"])[0]
    assert abs(np.linalg.norm(v) - 1.0) > 0.1  # raw normals have norm ~ sqrt(384)


def test_local_batch_failure_returns_all_none_never_raises():
    class BoomModel:
        def encode(self, texts):
            raise RuntimeError("model exploded")
    p = LocalSTProvider(BoomModel(), "m", normalize=True)
    assert p.embed(["a", "b", "c"]) == [None, None, None]


def test_embed_query_routes_through_same_provider_and_normalize():
    p = LocalSTProvider(FakeSTModel(), "m", normalize=True)
    via_query = embed_query("identical text", p)
    via_doc = p.embed(["identical text"])[0]
    assert via_query == via_doc  # same model + same normalization → identical vector


def test_embeddedchunk_preserves_identity_unchanged():
    c = make_chunk(0, doc_id="docX")
    ec = EmbeddedChunk(chunk=c, embedding=[0.0] * 384)
    assert ec.chunk_id == c.chunk_id
    assert ec.content_hash == c.content_hash
    assert ec.doc_version == c.doc_version


# ═══════════════════════════════════════════════════════════════════════════════
# Tiering + streaming
# ═══════════════════════════════════════════════════════════════════════════════

def test_tier_routing_50_500_5000():
    assert _tier(50) == SMALL
    assert _tier(100) == SMALL      # boundary: <= EMBED_SMALL_MAX
    assert _tier(500) == MEDIUM
    assert _tier(999) == MEDIUM
    assert _tier(1000) == LARGE     # boundary: >= EMBED_LARGE_MIN
    assert _tier(5000) == LARGE


def test_streaming_sink_batch_sizes_and_no_accumulation():
    chunks = [make_chunk(i, doc_id="docS") for i in range(100)]
    provider = LocalSTProvider(FakeSTModel(), "fake-model", normalize=True)
    sizes = []
    job = EmbeddingJob(doc_id="docS", doc_version=1, status=JobStatus.pending)

    result = embed_document(chunks, "docS", provider, lambda batch: sizes.append(len(batch)), job)

    assert sizes == [32, 32, 32, 4]          # 100 chunks, batch 32
    assert isinstance(result, EmbeddingJob)  # returns a job, not an accumulated list
    assert result.total_chunks == 100
    assert result.normalize is True
    assert result.model_name == "fake-model"
    assert result.status == JobStatus.completed


def test_checkpoint_writes_only_in_large(db_factory, monkeypatch):
    """SMALL/MEDIUM never write checkpoint_batch; LARGE advances it per batch."""
    s = get_settings()
    monkeypatch.setattr(s, "embed_batch_size", 2)
    provider = LocalSTProvider(FakeSTModel(), "m", normalize=True)

    # Same 6-chunk fixture, reclassified via thresholds so we exercise all tiers.
    def run_tier(small_max, large_min):
        monkeypatch.setattr(s, "embed_small_max", small_max)
        monkeypatch.setattr(s, "embed_large_min", large_min)
        chunks = [make_chunk(i, doc_id="docK") for i in range(6)]
        job = EmbeddingJob(doc_id="docK", doc_version=1, status=JobStatus.pending)
        return embed_document(chunks, "docK", provider, lambda b: None, job)

    small = run_tier(small_max=100, large_min=1000)   # 6 → SMALL
    medium = run_tier(small_max=0, large_min=1000)    # 6 → MEDIUM
    large = run_tier(small_max=0, large_min=1)        # 6 → LARGE (3 batches of 2)

    assert small.checkpoint_batch == -1
    assert medium.checkpoint_batch == -1
    assert large.checkpoint_batch == 2   # last batch index (0,1,2)


# ═══════════════════════════════════════════════════════════════════════════════
# Batch checkpoint / resume
# ═══════════════════════════════════════════════════════════════════════════════

def test_batch_checkpoint_and_resume(db_factory, monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "embed_batch_size", 1)
    monkeypatch.setattr(s, "embed_small_max", 0)
    monkeypatch.setattr(s, "embed_large_min", 1)  # LARGE → per-batch checkpoints

    sess = db_factory()
    chunks = [make_chunk(i, doc_id="docR", text=f"r{i}") for i in range(5)]
    _insert_chunks(sess, chunks)
    job = EmbeddingJob(doc_id="docR", doc_version=1, status=JobStatus.pending)
    sess.add(job)
    sess.commit()

    provider = SpyProvider()
    calls = {"n": 0}

    def raising_sink(batch):
        calls["n"] += 1
        if calls["n"] == 3:            # 3rd sink call == batch index 2
            raise RuntimeError("sink boom")

    with pytest.raises(RuntimeError):
        embed_document(chunks, "docR", provider, raising_sink, job)

    sess.refresh(job)
    assert job.checkpoint_batch == 1   # batches 0,1 committed; batch 2's sink failed
    assert provider.calls == [[chunks[0].text], [chunks[1].text], [chunks[2].text]]

    # Resume → processes ONLY batches 2,3,4.
    provider.calls.clear()
    recovered = []
    resume_embedding("docR", provider, lambda b: recovered.extend(b), job)

    sess.refresh(job)
    assert provider.calls == [[chunks[2].text], [chunks[3].text], [chunks[4].text]]
    assert job.checkpoint_batch == 4
    assert job.status == JobStatus.completed
    assert len(recovered) == 3
    sess.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Partial failure + retry
# ═══════════════════════════════════════════════════════════════════════════════

def test_partial_failure_completed_with_failures(monkeypatch):
    s = get_settings()
    monkeypatch.setattr(s, "embed_batch_size", 2)
    chunks = [make_chunk(i, doc_id="docP", text=f"p{i}") for i in range(4)]
    provider = SpyProvider(fail_texts={chunks[2].text, chunks[3].text})  # 2nd batch all-None
    collected = []
    job = EmbeddingJob(doc_id="docP", doc_version=1, status=JobStatus.pending)

    embed_document(chunks, "docP", provider, lambda b: collected.extend(b), job)

    assert job.status == JobStatus.completed_with_failures
    assert set(job.failed_chunk_ids) == {chunks[2].chunk_id, chunks[3].chunk_id}
    assert job.completed_chunks == 2
    assert all(ec.embedding is not None for ec in collected)  # sink never gets None
    assert {ec.chunk_id for ec in collected} == {chunks[0].chunk_id, chunks[1].chunk_id}


def test_all_none_marks_job_failed():
    chunks = [make_chunk(i, doc_id="docZ", text=f"z{i}") for i in range(3)]
    provider = SpyProvider(fail_texts={c.text for c in chunks})
    collected = []
    job = EmbeddingJob(doc_id="docZ", doc_version=1, status=JobStatus.pending)
    embed_document(chunks, "docZ", provider, lambda b: collected.extend(b), job)
    assert job.status == JobStatus.failed
    assert collected == []


def test_retry_failed_embeddings(db_factory):
    sess = db_factory()
    chunks = [make_chunk(i, doc_id="docT", text=f"t{i}") for i in range(4)]
    _insert_chunks(sess, chunks)
    job = EmbeddingJob(
        doc_id="docT", doc_version=1, status=JobStatus.completed_with_failures,
        total_chunks=4, completed_chunks=2,
        failed_chunk_ids=[chunks[2].chunk_id, chunks[3].chunk_id],
    )
    sess.add(job)
    sess.commit()

    provider = SpyProvider()  # now succeeds for everything
    collected = []
    retry_failed_embeddings("docT", provider, lambda b: collected.extend(b), job)

    sess.refresh(job)
    assert provider.calls == [[chunks[2].text, chunks[3].text]]  # exactly the failed texts
    assert job.failed_chunk_ids == []
    assert job.status == JobStatus.completed
    assert job.completed_chunks == 4
    assert {ec.chunk_id for ec in collected} == {chunks[2].chunk_id, chunks[3].chunk_id}
    sess.close()


# ═══════════════════════════════════════════════════════════════════════════════
# HF API provider
# ═══════════════════════════════════════════════════════════════════════════════

class _FakeResp:
    def __init__(self, status, headers=None):
        self.status_code = status
        self.headers = headers or {}


class _FakeHTTPError(Exception):
    def __init__(self, status, headers=None):
        super().__init__(f"HTTP {status}")
        self.response = _FakeResp(status, headers)


class _SeqHFClient:
    def __init__(self, behaviors):
        self.behaviors = list(behaviors)
        self.calls = []

    def feature_extraction(self, texts, **kw):
        self.calls.append(texts)
        b = self.behaviors.pop(0)
        if isinstance(b, Exception):
            raise b
        return b


def test_hf_503_then_200_succeeds_after_one_retry(monkeypatch):
    import app.pipeline.embedding_providers as ep
    slept = []
    monkeypatch.setattr(ep.time, "sleep", lambda d: slept.append(d))

    client = _SeqHFClient([_FakeHTTPError(503), np.ones((1, 384), dtype=np.float32)])
    provider = HFAPIProvider("m", client=client, base_delay=0.01, max_retries=4)

    out = provider.embed(["hi"])
    assert len(out) == 1 and out[0] is not None
    assert abs(np.linalg.norm(out[0]) - 1.0) < 1e-5
    assert len(client.calls) == 2   # failed once, then succeeded
    assert len(slept) == 1


def test_hf_honors_retry_after(monkeypatch):
    import app.pipeline.embedding_providers as ep
    slept = []
    monkeypatch.setattr(ep.time, "sleep", lambda d: slept.append(d))
    client = _SeqHFClient([
        _FakeHTTPError(429, headers={"Retry-After": "2.5"}),
        np.ones((1, 384), dtype=np.float32),
    ])
    provider = HFAPIProvider("m", client=client, base_delay=0.01, max_retries=4)
    provider.embed(["x"])
    assert slept == [2.5]


def test_hf_exhausted_returns_all_none(monkeypatch):
    import app.pipeline.embedding_providers as ep
    monkeypatch.setattr(ep.time, "sleep", lambda d: None)
    client = _SeqHFClient([_FakeHTTPError(503)] * 10)
    provider = HFAPIProvider("m", client=client, base_delay=0.001, max_retries=2)
    assert provider.embed(["a", "b"]) == [None, None]
    assert len(client.calls) == 3   # attempt 0,1,2 then give up


def test_hf_length_mismatch_marks_all_none():
    client = _SeqHFClient([np.ones((1, 384), dtype=np.float32)])  # 1 vector for 2 texts
    provider = HFAPIProvider("m", client=client, max_retries=0)
    assert provider.embed(["a", "b"]) == [None, None]


class _OkHFClient:
    def feature_extraction(self, texts, **kw):
        n = len(texts) if isinstance(texts, list) else 1
        return np.ones((n, 384), dtype=np.float32)


def test_batch_delay_applied_for_hf_not_local(monkeypatch):
    import app.pipeline.embedder as emb
    s = get_settings()
    monkeypatch.setattr(s, "embed_batch_size", 1)
    monkeypatch.setattr(s, "embed_batch_delay_s", 0.01)
    slept = []
    monkeypatch.setattr(emb.time, "sleep", lambda d: slept.append(d))

    chunks = [make_chunk(i, doc_id="docD", text=f"d{i}") for i in range(2)]  # 2 batches

    hf = HFAPIProvider("m", client=_OkHFClient())
    job = EmbeddingJob(doc_id="docD", doc_version=1, status=JobStatus.pending)
    embed_document(chunks, "docD", hf, lambda b: None, job)
    assert len(slept) == 1   # one inter-batch delay for hf_api

    slept.clear()
    local = LocalSTProvider(FakeSTModel(), "m", normalize=True)
    job2 = EmbeddingJob(doc_id="docD", doc_version=1, status=JobStatus.pending)
    embed_document(chunks, "docD", local, lambda b: None, job2)
    assert slept == []       # local never sleeps


# ═══════════════════════════════════════════════════════════════════════════════
# Nested resume in run_pipeline (the important one)
# ═══════════════════════════════════════════════════════════════════════════════

def test_run_pipeline_skips_embed_when_job_completed(db_factory):
    from app.models.document import DocumentStatus
    sess = db_factory()
    chunks = [make_chunk(i, doc_id="docN") for i in range(3)]
    _insert_doc(sess, "docN", status=DocumentStatus.embedding)
    _insert_chunks(sess, chunks)
    sess.add(EmbeddingJob(
        doc_id="docN", doc_version=1, status=JobStatus.completed,
        total_chunks=3, completed_chunks=3, checkpoint_batch=-1,
    ))
    sess.commit()
    sess.close()

    provider = SpyProvider()
    status = run_pipeline("docN", session_factory=db_factory, provider=provider)

    assert status == "ready"
    assert provider.calls == []   # STAGE-level resume: embed skipped entirely


def test_run_pipeline_resumes_embed_from_checkpoint_without_reparsing(db_factory, monkeypatch):
    from app.models.document import DocumentStatus
    import app.pipeline.orchestrator as orch

    s = get_settings()
    monkeypatch.setattr(s, "embed_batch_size", 1)
    monkeypatch.setattr(s, "embed_small_max", 0)
    monkeypatch.setattr(s, "embed_large_min", 1)
    # If parse runs, resume-by-persisted-chunks is broken. Fail loudly if it does.
    monkeypatch.setattr(orch, "parse", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("parse must not run when chunks already exist")))

    sess = db_factory()
    chunks = [make_chunk(i, doc_id="docC", text=f"c{i}") for i in range(5)]
    _insert_doc(sess, "docC", status=DocumentStatus.embedding)
    _insert_chunks(sess, chunks)
    sess.add(EmbeddingJob(
        doc_id="docC", doc_version=1, status=JobStatus.in_progress,
        total_chunks=5, completed_chunks=3, checkpoint_batch=2,   # done through batch 2
    ))
    sess.commit()
    sess.close()

    provider = SpyProvider()
    status = run_pipeline("docC", session_factory=db_factory, provider=provider)

    assert status == "ready"
    # BATCH-level resume: only batches 3,4 embedded (not from 0), and no re-parse.
    assert provider.calls == [[chunks[3].text], [chunks[4].text]]

    check = db_factory()
    job = check.query(EmbeddingJob).filter_by(doc_id="docC").first()
    assert job.status == JobStatus.completed
    assert job.checkpoint_batch == 4
    check.close()


def test_run_pipeline_full_then_idempotent_double_run(db_factory, tmp_path):
    from app.models.document import DocumentStatus
    pdf = tmp_path / "d.pdf"
    _make_pdf(pdf, [
        "Alpha beta gamma. Delta epsilon zeta. Eta theta iota.",
        "Second page here. More sentences follow. And even more content.",
    ])
    sess = db_factory()
    _insert_doc(sess, "docF", status=DocumentStatus.uploaded, file_path=str(pdf))
    sess.close()

    provider = SpyProvider()
    st1 = run_pipeline("docF", session_factory=db_factory, provider=provider)
    assert st1 == "ready"
    calls_first = len(provider.calls)
    assert calls_first > 0

    check = db_factory()
    n_chunks = check.query(ChunkRecord).filter_by(doc_id="docF").count()
    job = check.query(EmbeddingJob).filter_by(doc_id="docF").first()
    assert job.status == JobStatus.completed
    assert n_chunks > 0
    check.close()

    # Second run → top-level ready no-op: no new work, no duplicate chunks.
    st2 = run_pipeline("docF", session_factory=db_factory, provider=provider)
    assert st2 == "ready"
    assert len(provider.calls) == calls_first   # provider not called again

    check = db_factory()
    assert check.query(ChunkRecord).filter_by(doc_id="docF").count() == n_chunks
    check.close()
