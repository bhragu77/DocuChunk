import uuid
from datetime import datetime
from sqlalchemy import String, Boolean, DateTime, ForeignKey, func, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column
import enum
from app.database import Base


class AuthEventType(str, enum.Enum):
    register_email   = "register_email"
    login_email      = "login_email"
    register_google  = "register_google"
    login_google     = "login_google"
    register_github  = "register_github"
    login_github     = "login_github"
    password_reset_request  = "password_reset_request"
    password_reset_complete = "password_reset_complete"
    token_refresh    = "token_refresh"


class AuthEvent(Base):
    """
    Audit log for every authentication action.
    Insert on every login/register/oauth/reset — never update.

    Learning note: audit logs should be append-only. Never update or delete rows.
    This table is your forensic record of who accessed the system and how.
    """
    __tablename__ = "auth_events"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str | None] = mapped_column(
        String, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )  # nullable: failed logins have no user_id

    event_type: Mapped[AuthEventType] = mapped_column(SAEnum(AuthEventType), nullable=False, index=True)
    provider: Mapped[str | None] = mapped_column(String(50), nullable=True)   # "email" | "google" | "github"
    success: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)   # IPv4 or IPv6
    user_agent: Mapped[str | None] = mapped_column(String(500), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), index=True)

    def __repr__(self) -> str:
        return f"<AuthEvent {self.event_type} user={self.user_id} ok={self.success}>"
