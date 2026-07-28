"""
Chat session persistence — ordering and blank-session behaviour.

These cover the contract the chat UI relies on to reopen the RIGHT conversation
when the user navigates back to /chat: the sidebar list is ordered by real chat
activity (not by creation time), and a document never accumulates blank sessions
that could out-rank the conversation the user was actually in.
"""
import time

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.dependencies import get_current_user
from app.database import Base, get_db
from app.main import app

TEST_DATABASE_URL = "sqlite:///./test_chat_sessions.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True, scope="module")
def setup_db():
    from app.models import chat, document, user  # noqa: F401
    Base.metadata.create_all(bind=engine)
    app.dependency_overrides[get_db] = override_get_db
    yield
    Base.metadata.drop_all(bind=engine)
    app.dependency_overrides.clear()


client = TestClient(app)


def auth(email: str) -> dict:
    client.post("/auth/register", json={"email": email, "password": "password123"})
    res = client.post("/auth/login", json={"email": email, "password": "password123"})
    return {"Authorization": f"Bearer {res.json()['access_token']}"}


def make_doc(headers: dict, name: str) -> str:
    """Insert a ready Document straight into the DB (no upload/pipeline needed)."""
    from app.models.document import Document, DocumentStatus

    me = client.get("/auth/me", headers=headers).json()
    db = TestingSession()
    try:
        doc = Document(
            user_id=me["id"],
            filename=f"{name}.pdf",
            original_filename=f"{name}.pdf",
            file_path=f"/tmp/{name}.pdf",
            file_size=1024,
            mime_type="application/pdf",
            status=DocumentStatus.ready,
            page_count=1,
        )
        db.add(doc)
        db.commit()
        db.refresh(doc)
        return doc.id
    finally:
        db.close()


def new_session(headers: dict, doc_id: str) -> str:
    res = client.post("/api/chat/sessions", json={"doc_id": doc_id}, headers=headers)
    assert res.status_code == 201, res.text
    return res.json()["session_id"]


def say(headers: dict, sid: str, text: str):
    res = client.post(
        f"/api/chat/sessions/{sid}/messages",
        json={"role": "user", "content": text},
        headers=headers,
    )
    assert res.status_code == 201, res.text


def test_ongoing_chat_outranks_a_newer_session():
    """The list is ordered by last activity, so the conversation the user is
    actually chatting in comes first — even when another session was created
    more recently."""
    headers = auth("order@example.com")
    doc_a, doc_b = make_doc(headers, "a"), make_doc(headers, "b")

    old = new_session(headers, doc_a)
    say(headers, old, "first question")       # title set → first updated_at bump
    time.sleep(1.1)                           # sqlite CURRENT_TIMESTAMP is 1s-granular

    newer = new_session(headers, doc_b)
    say(headers, newer, "unrelated")
    time.sleep(1.1)

    say(headers, old, "second question")      # back to the original conversation

    sessions = client.get("/api/chat/sessions", headers=headers).json()
    assert sessions[0]["session_id"] == old
    assert sessions[0]["message_count"] == 2


def test_blank_session_is_reused_not_duplicated():
    """Repeated 'New Chat' on the same document must not stack up blanks — the
    untouched session is handed back instead."""
    headers = auth("blank@example.com")
    doc = make_doc(headers, "blank")

    first = new_session(headers, doc)
    assert new_session(headers, doc) == first

    sessions = client.get("/api/chat/sessions", headers=headers).json()
    assert [s["session_id"] for s in sessions] == [first]

    # Once it has a turn it is a real conversation, so New Chat starts a fresh one.
    say(headers, first, "hello")
    second = new_session(headers, doc)
    assert second != first
    assert len(client.get("/api/chat/sessions", headers=headers).json()) == 2


def test_blank_session_reports_zero_messages():
    """The UI skips message_count == 0 when restoring; the API must report it."""
    headers = auth("count@example.com")
    doc = make_doc(headers, "count")
    sid = new_session(headers, doc)

    summary = next(
        s for s in client.get("/api/chat/sessions", headers=headers).json()
        if s["session_id"] == sid
    )
    assert summary["message_count"] == 0
    assert summary["preview"] == ""
