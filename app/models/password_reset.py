import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


class PasswordResetToken(Base):
    """
    Stores password reset tokens — but never the raw token itself.

    Security model:
      1. We generate: raw_token = secrets.token_urlsafe(48)
      2. We store:    token_hash = sha256(raw_token)  ← only the HASH goes in DB
      3. We email:    link contains raw_token
      4. On reset:    hash the submitted token, look up by hash
      5. After use:   set used_at (token cannot be reused)

    Why hash? If the DB is compromised, the attacker has hashes, not tokens.
    SHA-256 hashes cannot be reversed to recover the raw token.
    This is the same principle as password hashing — never store secrets in plain text.
    """
    __tablename__ = "password_reset_tokens"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    token_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)  # SHA-256 hex = 64 chars
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    @property
    def is_valid(self) -> bool:
        from datetime import timezone
        return self.used_at is None and self.expires_at > datetime.now(timezone.utc).replace(tzinfo=None)

    def __repr__(self) -> str:
        return f"<PasswordResetToken user={self.user_id} valid={self.is_valid}>"
