"""
Worker-migration tests.

Tasks are tested by calling the task FUNCTION directly with a minimal fake ctx
(a dict carrying a session_factory) against a real temp SQLite DB — NO Redis and
NO running worker in the test process. This keeps the suite fast and deterministic
while still exercising the real orchestrator stages.

Covers:
  * run_pipeline(ctx, doc_id) advances an uploaded doc through parse/clean/chunk
    to 'ready' (ported from the old BackgroundTasks integration test).
  * IDEMPOTENCY: re-running the task on the same doc_id does not duplicate chunks
    or corrupt status — it no-ops when already done and resumes after a simulated
    crash. This is the whole point of the migration.
  * /docs/upload enqueues run_pipeline (mocked pool) with the correct doc_id and
    returns immediately.
"""
import os
import uuid
from unittest.mock import AsyncMock

import pytest
import chromadb
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models.document import Document, DocumentStatus
from app.models.chunk import ChunkRecord
from app.workers.tasks import run_pipeline


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_pdf(path, pages):
    import fitz
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((50, 72), text)
    doc.save(str(path))
    doc.close()


def _insert_uploaded_doc(session, file_path) -> str:
    doc = Document(
        id=str(uuid.uuid4()),
        user_id="test-user",
        filename=os.path.basename(str(file_path)),
        original_filename="test.pdf",
        file_path=str(file_path),
        file_size=os.path.getsize(str(file_path)),
        mime_type="application/pdf",
        status=DocumentStatus.uploaded,
    )
    session.add(doc)
    session.commit()
    return doc.id


@pytest.fixture
def temp_db(tmp_path):
    """A throwaway SQLite DB + session factory for direct task invocation."""
    engine = create_engine(
        f"sqlite:///{tmp_path/'worker_test.db'}",
        connect_args={"check_same_thread": False},
    )
    TestSession = sessionmaker(bind=engine)
    from app.models import user, document, job, chunk  # noqa: F401 register tables
    Base.metadata.create_all(bind=engine)
    try:
        yield TestSession, tmp_path
    finally:
        engine.dispose()


# ── 1. Ported integration test: task advances doc to ready ────────────────────

@pytest.mark.asyncio
async def test_run_pipeline_task_advances_to_ready(temp_db):
    TestSession, tmp_path = temp_db
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, [
        "First page with content. It has two sentences.",
        "Second page continues with more content. Another sentence follows. And a third.",
    ])
    s = TestSession()
    doc_id = _insert_uploaded_doc(s, pdf)
    s.close()

    ctx = {"session_factory": TestSession}
    result = await run_pipeline(ctx, doc_id)

    assert result == {"doc_id": doc_id, "status": "ready"}

    s = TestSession()
    doc = s.get(Document, doc_id)
    assert doc.status == DocumentStatus.ready
    assert doc.chunk_count > 0
    assert doc.chunker_config is not None and "strategy" in doc.chunker_config
    persisted = s.query(ChunkRecord).filter_by(doc_id=doc_id).count()
    assert persisted == doc.chunk_count
    s.close()


# ── 2. Idempotency: the whole point of the migration ──────────────────────────

@pytest.mark.asyncio
async def test_run_pipeline_is_idempotent(temp_db):
    TestSession, tmp_path = temp_db
    pdf = tmp_path / "doc.pdf"
    _make_pdf(pdf, [
        "Alpha content here. Beta content follows. Gamma ends the first page.",
        "Second page adds more detail. It has several more sentences to chunk.",
    ])
    s = TestSession()
    doc_id = _insert_uploaded_doc(s, pdf)
    s.close()

    ctx = {"session_factory": TestSession}

    # First run → ready with N chunks.
    r1 = await run_pipeline(ctx, doc_id)
    assert r1["status"] == "ready"
    s = TestSession()
    ids_first = sorted(c.id for c in s.query(ChunkRecord).filter_by(doc_id=doc_id).all())
    s.close()
    assert len(ids_first) > 0

    # Second run on the SAME doc_id → no-op (status already ready). No new chunks.
    r2 = await run_pipeline(ctx, doc_id)
    assert r2["status"] == "ready"
    s = TestSession()
    ids_second = sorted(c.id for c in s.query(ChunkRecord).filter_by(doc_id=doc_id).all())
    s.close()
    assert ids_second == ids_first, "re-run must not duplicate or change chunks"

    # Simulate a CRASH after chunking: reset status while chunks remain persisted.
    # Re-entry must resume (skip parse/clean/chunk via resume-by-persisted-chunks),
    # NOT duplicate chunks, and return to ready.
    s = TestSession()
    doc = s.get(Document, doc_id)
    doc.status = DocumentStatus.chunking
    s.commit()
    s.close()

    r3 = await run_pipeline(ctx, doc_id)
    assert r3["status"] == "ready"
    s = TestSession()
    ids_third = sorted(c.id for c in s.query(ChunkRecord).filter_by(doc_id=doc_id).all())
    doc = s.get(Document, doc_id)
    s.close()
    assert ids_third == ids_first, "resume-after-crash must not duplicate chunks"
    assert doc.chunk_count == len(ids_first)


# ── 3. Upload enqueues a job (mocked pool) and returns immediately ────────────

@pytest.mark.asyncio
async def test_upload_enqueues_run_pipeline_job(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from app.main import app
    from app.database import get_db
    from app.core.dependencies import get_chroma, get_arq_pool
    import app.routers.documents as documents_module

    engine = create_engine(
        f"sqlite:///{tmp_path/'upload_test.db'}",
        connect_args={"check_same_thread": False},
    )
    UpSession = sessionmaker(bind=engine)
    from app.models import user, document, job, chunk  # noqa: F401
    Base.metadata.create_all(bind=engine)

    def _db():
        db = UpSession()
        try:
            yield db
        finally:
            db.close()

    fake_pool = AsyncMock()

    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[get_chroma] = lambda: chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    app.dependency_overrides[get_arq_pool] = lambda: fake_pool
    # Force the enqueue path (not the inline PIPELINE_SYNC fallback).
    monkeypatch.setattr(documents_module.settings, "pipeline_sync", False)

    client = TestClient(app)
    try:
        client.post("/auth/register", json={"email": "enq@example.com", "password": "password123"})
        token = client.post("/auth/login", json={
            "email": "enq@example.com", "password": "password123",
        }).json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        pdf = tmp_path / "up.pdf"
        _make_pdf(pdf, ["Enqueue test content for the worker."])
        with open(pdf, "rb") as f:
            resp = client.post(
                "/docs/upload",
                files={"file": ("up.pdf", f, "application/pdf")},
                headers=headers,
            )

        assert resp.status_code == 201
        body = resp.json()
        # Returns immediately: pipeline was NOT run inline (fake pool does nothing),
        # so the doc is still in 'uploaded' status.
        assert body["status"] == "uploaded"

        # The job was enqueued with the right task name and doc_id.
        fake_pool.enqueue_job.assert_awaited_once()
        await_args = fake_pool.enqueue_job.await_args
        assert await_args.args[0] == "run_pipeline"
        assert await_args.args[1] == body["id"]
    finally:
        app.dependency_overrides.clear()
        engine.dispose()
