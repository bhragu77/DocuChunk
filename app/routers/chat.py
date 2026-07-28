"""
Chat session persistence API — powers the "Chat with Document" sidebar and
history. Generation itself stays in /generate/answer; this router only stores and
replays the conversation.

Lifecycle (every conversation stays resumable):
  * POST /chat/sessions        — start a fresh session for a document. Other
                                 sessions are left untouched — a user can keep
                                 several conversations and return to any of them.
  * POST /chat/sessions/{id}/messages — append a turn (user or assistant). Any
                                 owned session is writable; appending re-activates
                                 a session an older build may have left locked.
  * POST /chat/sessions/{id}/lock     — optional manual archive (unused by the UI
                                 now; a re-opened session becomes active again).
  * GET  /chat/sessions        — sidebar list (newest first, with preview).
  * GET  /chat/sessions/{id}   — a session with its full message history (the
                                 context restored when the user clicks it).
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user
from app.core.timefmt import iso_utc
from app.database import get_db
from app.models.chat import ChatMessage, ChatSession, ChatSessionStatus
from app.models.document import Document
from app.models.user import User

# NOTE: prefix is /api/chat, NOT /chat — the pages router owns GET /chat/{doc_id}
# (the chat HTML page) and is registered first, so it would shadow /chat/sessions.
router = APIRouter(prefix="/api/chat", tags=["chat"])
logger = logging.getLogger(__name__)


# ── Schemas ─────────────────────────────────────────────────────────────────

class CreateSessionRequest(BaseModel):
    doc_id: str


class AppendMessageRequest(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = ""
    citations: dict | None = None
    confidence: float | None = None
    confidence_signals: dict | None = None
    abstained: bool = False
    verified: bool | None = None


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    citations: dict | None = None
    confidence: float | None = None
    confidence_signals: dict | None = None
    abstained: bool = False
    verified: bool | None = None
    created_at: str

    @classmethod
    def of(cls, m: ChatMessage) -> "MessageOut":
        return cls(
            id=m.id, role=m.role, content=m.content, citations=m.citations,
            confidence=m.confidence, confidence_signals=m.confidence_signals,
            abstained=m.abstained, verified=m.verified,
            created_at=iso_utc(m.created_at) or "",
        )


class SessionSummary(BaseModel):
    session_id: str
    doc_id: str
    doc_name: str
    title: str | None
    status: str
    created_at: str
    updated_at: str
    message_count: int
    preview: str


class SessionDetail(SessionSummary):
    messages: list[MessageOut] = []


# ── Helpers ─────────────────────────────────────────────────────────────────

def _get_session_or_404(session_id: str, user: User, db: Session) -> ChatSession:
    s = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == user.id)
        .first()
    )
    if s is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Chat session not found")
    return s


def _summary(s: ChatSession) -> SessionSummary:
    first_user = next((m for m in s.messages if m.role == "user"), None)
    return SessionSummary(
        session_id=s.id,
        doc_id=s.doc_id,
        doc_name=s.doc_name,
        title=s.title,
        status=s.status.value if hasattr(s.status, "value") else str(s.status),
        created_at=iso_utc(s.created_at) or "",
        updated_at=iso_utc(s.updated_at) or "",
        message_count=len(s.messages),
        preview=(first_user.content[:120] if first_user else (s.title or "")),
    )


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/sessions", response_model=SessionDetail, status_code=status.HTTP_201_CREATED)
def create_session(
    payload: CreateSessionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    doc = (
        db.query(Document)
        .filter(Document.id == payload.doc_id, Document.user_id == current_user.id)
        .first()
    )
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found")

    # Sessions are no longer auto-locked: a user can keep several conversations
    # going and return to any of them later (see append_message — every owned
    # session stays writable). Creating a new session just adds another one.
    #
    # …except an empty one: if the user already has a session for this document
    # that was never chatted in, hand that back instead of stacking up another
    # identical blank. Otherwise every "New Chat" click that goes nowhere leaves a
    # dummy session behind, cluttering the sidebar and competing to be restored.
    blank = next(
        (
            existing
            for existing in db.query(ChatSession)
            .filter(ChatSession.user_id == current_user.id, ChatSession.doc_id == doc.id)
            .order_by(ChatSession.created_at.desc())
            .all()
            if not existing.messages
        ),
        None,
    )
    if blank is not None:
        return SessionDetail(**_summary(blank).model_dump(), messages=[])

    s = ChatSession(
        user_id=current_user.id,
        doc_id=doc.id,
        doc_name=doc.original_filename,
        status=ChatSessionStatus.active,
    )
    db.add(s)
    db.commit()
    db.refresh(s)
    detail = SessionDetail(**_summary(s).model_dump(), messages=[])
    return detail


@router.get("/sessions", response_model=list[SessionSummary])
def list_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    sessions = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.updated_at.desc())
        .all()
    )
    return [_summary(s) for s in sessions]


@router.get("/sessions/{session_id}", response_model=SessionDetail)
def get_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    s = _get_session_or_404(session_id, current_user, db)
    return SessionDetail(
        **_summary(s).model_dump(),
        messages=[MessageOut.of(m) for m in s.messages],
    )


@router.post("/sessions/{session_id}/messages", response_model=MessageOut, status_code=status.HTTP_201_CREATED)
def append_message(
    session_id: str,
    payload: AppendMessageRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    s = _get_session_or_404(session_id, current_user, db)
    # Any owned session is continuable — appending a turn re-activates a session
    # that an older build may have left `locked`, so returning to it just works.
    if s.status != ChatSessionStatus.active:
        s.status = ChatSessionStatus.active

    m = ChatMessage(
        session_id=s.id,
        role=payload.role,
        content=payload.content,
        citations=payload.citations,
        confidence=payload.confidence,
        confidence_signals=payload.confidence_signals,
        abstained=payload.abstained,
        verified=payload.verified,
    )
    db.add(m)
    # First user turn becomes the sidebar title.
    if payload.role == "user" and not s.title:
        s.title = payload.content[:120]
    # Stamp updated_at explicitly: `onupdate` only fires when the session row is
    # actually dirty, and from the second turn on nothing on it changes (the title
    # is already set). Without this the "newest first" ordering degrades into
    # "most recently created", so a brand-new empty session outranks the
    # conversation the user has actually been chatting in.
    s.updated_at = func.now()
    db.add(s)
    db.commit()
    db.refresh(m)
    return MessageOut.of(m)


@router.post("/sessions/{session_id}/lock", response_model=SessionSummary)
def lock_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    s = _get_session_or_404(session_id, current_user, db)
    s.status = ChatSessionStatus.locked
    db.add(s)
    db.commit()
    db.refresh(s)
    return _summary(s)


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    s = _get_session_or_404(session_id, current_user, db)
    db.delete(s)
    db.commit()
