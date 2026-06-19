import io
import pytest
import chromadb
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database import Base, get_db
from app.core.dependencies import get_chroma

# ── Test DB setup ─────────────────────────────────────────────────────────────

TEST_DATABASE_URL = "sqlite:///./test_docs.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# In-memory ChromaDB for tests — avoids needing the lifespan to run
_test_chroma = chromadb.EphemeralClient()


def override_get_db():
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()


def override_get_chroma():
    return _test_chroma


@pytest.fixture(autouse=True, scope="module")
def setup_db():
    from app.models import user, document, job  # noqa
    Base.metadata.create_all(bind=engine)
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_chroma] = override_get_chroma
    yield
    Base.metadata.drop_all(bind=engine)
    app.dependency_overrides.clear()


client = TestClient(app)


# ── File factories ────────────────────────────────────────────────────────────

def make_pdf(text: str = "Hello PDF") -> bytes:
    """Create a minimal valid PDF in memory using PyMuPDF."""
    import fitz  # PyMuPDF
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def make_docx(text: str = "Hello DOCX") -> bytes:
    """Create a minimal valid DOCX in memory using python-docx."""
    from docx import Document
    doc = Document()
    doc.add_paragraph(text)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


# ── Auth helper ───────────────────────────────────────────────────────────────

def get_auth_header(email: str = "docuser@example.com", password: str = "password123") -> dict:
    """Register (or login) and return Authorization header."""
    client.post("/auth/register", json={"email": email, "password": password})
    res = client.post("/auth/login", json={"email": email, "password": password})
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# ── Upload ────────────────────────────────────────────────────────────────────

def test_upload_pdf_success():
    headers = get_auth_header("pdf@example.com")
    pdf_bytes = make_pdf("Test PDF upload")
    res = client.post(
        "/docs/upload",
        files={"file": ("test.pdf", pdf_bytes, "application/pdf")},
        headers=headers,
    )
    assert res.status_code == 201
    body = res.json()
    assert body["original_filename"] == "test.pdf"
    assert body["mime_type"] == "application/pdf"
    assert body["status"] == "uploaded"
    assert body["file_size"] > 0
    assert "id" in body


def test_upload_docx_success():
    headers = get_auth_header("docx@example.com")
    docx_bytes = make_docx("Test DOCX upload")
    res = client.post(
        "/docs/upload",
        files={"file": ("test.docx", docx_bytes, "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        headers=headers,
    )
    assert res.status_code == 201
    body = res.json()
    assert body["original_filename"] == "test.docx"
    assert body["status"] == "uploaded"


def test_upload_invalid_extension():
    headers = get_auth_header("txt@example.com")
    res = client.post(
        "/docs/upload",
        files={"file": ("notes.txt", b"some text", "text/plain")},
        headers=headers,
    )
    assert res.status_code == 422
    assert "Unsupported" in res.json()["detail"]


def test_upload_empty_file():
    headers = get_auth_header("empty@example.com")
    res = client.post(
        "/docs/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
        headers=headers,
    )
    assert res.status_code == 422
    assert "empty" in res.json()["detail"].lower()


def test_upload_requires_auth():
    pdf_bytes = make_pdf()
    res = client.post(
        "/docs/upload",
        files={"file": ("test.pdf", pdf_bytes, "application/pdf")},
    )
    assert res.status_code == 401


# ── List ──────────────────────────────────────────────────────────────────────

def test_list_documents_empty():
    headers = get_auth_header("listuser@example.com")
    res = client.get("/docs/list", headers=headers)
    assert res.status_code == 200
    body = res.json()
    assert body["items"] == []
    assert body["total"] == 0
    assert body["page"] == 1


def test_list_documents_shows_own_docs_only():
    # User A uploads a doc
    headers_a = get_auth_header("usera@example.com")
    pdf_bytes = make_pdf()
    client.post("/docs/upload", files={"file": ("a.pdf", pdf_bytes, "application/pdf")}, headers=headers_a)

    # User B uploads a doc
    headers_b = get_auth_header("userb@example.com")
    client.post("/docs/upload", files={"file": ("b.pdf", pdf_bytes, "application/pdf")}, headers=headers_b)

    # User A should only see their own doc
    res = client.get("/docs/list", headers=headers_a)
    assert res.status_code == 200
    body = res.json()
    emails_in_result = [item["original_filename"] for item in body["items"]]
    assert "a.pdf" in emails_in_result
    assert "b.pdf" not in emails_in_result


def test_list_pagination():
    headers = get_auth_header("paginate@example.com")
    pdf_bytes = make_pdf()

    # Upload 5 documents
    for i in range(5):
        client.post(
            "/docs/upload",
            files={"file": (f"doc{i}.pdf", pdf_bytes, "application/pdf")},
            headers=headers,
        )

    # First page of 2
    res = client.get("/docs/list?page=1&page_size=2", headers=headers)
    assert res.status_code == 200
    body = res.json()
    assert len(body["items"]) == 2
    assert body["total"] == 5
    assert body["total_pages"] == 3

    # Second page
    res2 = client.get("/docs/list?page=2&page_size=2", headers=headers)
    assert res2.status_code == 200
    body2 = res2.json()
    assert len(body2["items"]) == 2
    # Ensure pages are different
    ids_p1 = {item["id"] for item in body["items"]}
    ids_p2 = {item["id"] for item in body2["items"]}
    assert ids_p1.isdisjoint(ids_p2)


def test_list_requires_auth():
    res = client.get("/docs/list")
    assert res.status_code == 401


# ── Get single document ───────────────────────────────────────────────────────

def test_get_document_success():
    headers = get_auth_header("getdoc@example.com")
    pdf_bytes = make_pdf()
    upload_res = client.post(
        "/docs/upload",
        files={"file": ("get_test.pdf", pdf_bytes, "application/pdf")},
        headers=headers,
    )
    doc_id = upload_res.json()["id"]

    res = client.get(f"/docs/{doc_id}", headers=headers)
    assert res.status_code == 200
    assert res.json()["id"] == doc_id
    assert res.json()["original_filename"] == "get_test.pdf"


def test_get_document_not_found():
    headers = get_auth_header("getdoc@example.com")
    res = client.get("/docs/nonexistent-id", headers=headers)
    assert res.status_code == 404


def test_get_document_other_users_doc():
    """User B cannot fetch User A's document."""
    headers_a = get_auth_header("ownera@example.com")
    headers_b = get_auth_header("ownerb@example.com")
    pdf_bytes = make_pdf()

    upload_res = client.post(
        "/docs/upload",
        files={"file": ("private.pdf", pdf_bytes, "application/pdf")},
        headers=headers_a,
    )
    doc_id = upload_res.json()["id"]

    res = client.get(f"/docs/{doc_id}", headers=headers_b)
    assert res.status_code == 404  # appears as not-found, not 403 (no info leak)


# ── Delete ────────────────────────────────────────────────────────────────────

def test_delete_document_success():
    headers = get_auth_header("delete@example.com")
    pdf_bytes = make_pdf()
    upload_res = client.post(
        "/docs/upload",
        files={"file": ("to_delete.pdf", pdf_bytes, "application/pdf")},
        headers=headers,
    )
    doc_id = upload_res.json()["id"]

    del_res = client.delete(f"/docs/{doc_id}", headers=headers)
    assert del_res.status_code == 200
    assert del_res.json()["document_id"] == doc_id

    # Subsequent GET should 404
    get_res = client.get(f"/docs/{doc_id}", headers=headers)
    assert get_res.status_code == 404


def test_delete_removes_from_list():
    headers = get_auth_header("deletelist@example.com")
    pdf_bytes = make_pdf()
    upload_res = client.post(
        "/docs/upload",
        files={"file": ("will_be_gone.pdf", pdf_bytes, "application/pdf")},
        headers=headers,
    )
    doc_id = upload_res.json()["id"]

    client.delete(f"/docs/{doc_id}", headers=headers)

    list_res = client.get("/docs/list", headers=headers)
    ids = [item["id"] for item in list_res.json()["items"]]
    assert doc_id not in ids


def test_delete_other_users_doc():
    """User B cannot delete User A's document."""
    headers_a = get_auth_header("del_owner@example.com")
    headers_b = get_auth_header("del_thief@example.com")
    pdf_bytes = make_pdf()

    upload_res = client.post(
        "/docs/upload",
        files={"file": ("protected.pdf", pdf_bytes, "application/pdf")},
        headers=headers_a,
    )
    doc_id = upload_res.json()["id"]

    res = client.delete(f"/docs/{doc_id}", headers=headers_b)
    assert res.status_code == 404
