"""
Chat session persistence — the per-user conversation history behind the
"Chat with Document" UI.

A ChatSession is one conversation, always bound to a single document. Sessions
are per-user and stay RESUMABLE: clicking a session in the sidebar reloads its
full message history and the user can keep chatting in it. The `status` column is
retained for compatibility (and an optional manual archive), but the UI no longer
auto-locks sessions — appending a turn to any owned session re-activates it.

ChatMessage stores both the user turns and the assistant turns. Assistant turns
carry the full trust payload from /generate/answer (citations, confidence, the
confidence_signals breakdown, abstained/verified) so the history renders with the
exact same badges and confidence bar as a live answer — no re-generation needed.
"""
import enum
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum as SAEnum, Float, ForeignKey, JSON, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class ChatSessionStatus(str, enum.Enum):
    active = "active"    # the user's current, writable conversation
    locked = "locked"    # read-only history (user left / started a new chat)


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    doc_id: Mapped[str] = mapped_column(
        String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Denormalized so the sidebar can render "report.pdf" without a join even if
    # the document is later deleted.
    doc_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    # First user message, truncated — the sidebar preview line.
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)

    status: Mapped[ChatSessionStatus] = mapped_column(
        SAEnum(ChatSessionStatus), default=ChatSessionStatus.active, nullable=False, index=True
    )

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    messages: Mapped[list["ChatMessage"]] = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(
        String, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")

    # Assistant-only trust payload (null for user turns). Mirrors the shapes the
    # frontend already renders from a live /generate/answer response.
    citations: Mapped[dict | None] = mapped_column(JSON, nullable=True)          # {cited_sources, dropped_sources}
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    confidence_signals: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    abstained: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    verified: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    session: Mapped["ChatSession"] = relationship("ChatSession", back_populates="messages")
