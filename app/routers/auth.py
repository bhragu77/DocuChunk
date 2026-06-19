import uuid
import hashlib
import secrets
import logging
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User
from app.models.auth_event import AuthEvent, AuthEventType
from app.models.password_reset import PasswordResetToken
from app.schemas.auth import (
    RegisterRequest, LoginRequest, RefreshRequest,
    TokenResponse, UserResponse, MessageResponse,
    ForgotPasswordRequest, ForgotPasswordResponse, ResetPasswordRequest,
)
from app.core.security import (
    hash_password, verify_password,
    create_access_token, create_refresh_token, verify_refresh_token,
)
from app.core.dependencies import get_current_user
from app.core.oauth import oauth
from app.config import get_settings

router = APIRouter(prefix="/auth", tags=["auth"])
settings = get_settings()
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _client_ip(request: Request) -> str | None:
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else None


def _log_event(
    db: Session,
    event_type: AuthEventType,
    request: Request,
    user_id: str | None = None,
    provider: str | None = None,
    success: bool = True,
) -> None:
    """Append-only insert to auth_events — never update this table."""
    event = AuthEvent(
        id=str(uuid.uuid4()),
        user_id=user_id,
        event_type=event_type,
        provider=provider,
        success=success,
        ip_address=_client_ip(request),
        user_agent=request.headers.get("User-Agent", "")[:500],
    )
    db.add(event)
    db.commit()
    logger.info("AuthEvent: type=%s user=%s provider=%s ok=%s ip=%s",
                event_type.value, user_id, provider, success, event.ip_address)


def _welcome_redirect(access_token: str, refresh_token: str, provider: str, name: str) -> str:
    """Build the URL for the welcome animation page."""
    params = urlencode({
        "access_token":  access_token,
        "refresh_token": refresh_token,
        "provider":      provider,
        "name":          name or "",
    })
    return f"/welcome?{params}"


def _hash_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


# ── JWT Auth ──────────────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest, request: Request, db: Session = Depends(get_db)):
    """
    Create a new email/password account.
    Logs a register_email event with IP + user-agent.
    """
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        _log_event(db, AuthEventType.register_email, request, success=False, provider="email")
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="An account with this email already exists")

    user = User(
        id=str(uuid.uuid4()),
        email=payload.email,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    _log_event(db, AuthEventType.register_email, request, user_id=user.id, provider="email")

    return TokenResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
    )


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, request: Request, db: Session = Depends(get_db)):
    """
    Email + password login.
    Logs a login_email event regardless of success/failure (for breach detection).
    """
    user = db.query(User).filter(User.email == payload.email).first()
    ok = (user is not None and user.hashed_password is not None
          and verify_password(payload.password, user.hashed_password))

    _log_event(
        db, AuthEventType.login_email, request,
        user_id=user.id if user else None,
        provider="email", success=ok,
    )

    if not ok:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid email or password")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled")

    return TokenResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
    )


@router.post("/refresh", response_model=TokenResponse)
def refresh_token(payload: RefreshRequest, request: Request, db: Session = Depends(get_db)):
    """Rotate refresh token — exchange for a new access + refresh pair."""
    token_data = verify_refresh_token(payload.refresh_token)
    if token_data is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid or expired refresh token")

    user = db.query(User).filter(User.id == token_data.get("sub")).first()
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    _log_event(db, AuthEventType.token_refresh, request, user_id=user.id, provider="email")

    return TokenResponse(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
    )


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user


# ── Forgot Password ───────────────────────────────────────────────────────────

@router.post("/forgot-password", response_model=ForgotPasswordResponse)
def forgot_password(payload: ForgotPasswordRequest, request: Request, db: Session = Depends(get_db)):
    """
    Request a password reset link.

    Security design:
    - Always return 200 even if the email doesn't exist (no user enumeration).
    - The raw token is NEVER stored — only its SHA-256 hash.
    - Token expires in 1 hour and can only be used once.
    - In development: reset URL is returned in the response body.
    - In production: send via email (wire in your SMTP here).
    """
    user = db.query(User).filter(User.email == payload.email).first()

    reset_url = None
    if user and user.is_active:
        # Invalidate any existing unused tokens for this user
        db.query(PasswordResetToken).filter(
            PasswordResetToken.user_id == user.id,
            PasswordResetToken.used_at.is_(None),
        ).delete()

        raw_token = secrets.token_urlsafe(48)
        expires_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(
            hours=settings.password_reset_expire_hours
        )

        prt = PasswordResetToken(
            id=str(uuid.uuid4()),
            user_id=user.id,
            token_hash=_hash_token(raw_token),
            expires_at=expires_at,
        )
        db.add(prt)
        db.commit()

        _log_event(db, AuthEventType.password_reset_request, request,
                   user_id=user.id, provider="email")

        reset_url = f"{settings.frontend_url}/reset-password?token={raw_token}"
        logger.info("Password reset URL (dev only): %s", reset_url)

        # ── Production: replace with real email send ──────────────────────
        # from app.core.email import send_reset_email
        # await send_reset_email(to=user.email, reset_url=reset_url)
        # ─────────────────────────────────────────────────────────────────

    return ForgotPasswordResponse(
        message="If an account with that email exists, a reset link has been sent.",
        reset_url=reset_url if settings.debug else None,  # only expose in dev
    )


@router.post("/reset-password", response_model=MessageResponse)
def reset_password(payload: ResetPasswordRequest, request: Request, db: Session = Depends(get_db)):
    """
    Consume a reset token and update the password.
    Token is invalidated immediately after use.
    """
    token_hash = _hash_token(payload.token)

    prt = db.query(PasswordResetToken).filter(
        PasswordResetToken.token_hash == token_hash,
    ).first()

    if prt is None or not prt.is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reset link is invalid or has expired. Please request a new one.",
        )

    user = db.query(User).filter(User.id == prt.user_id).first()
    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User not found")

    # Update password and mark token as used
    user.hashed_password = hash_password(payload.new_password)
    prt.used_at = datetime.now(timezone.utc).replace(tzinfo=None)
    db.commit()

    _log_event(db, AuthEventType.password_reset_complete, request,
               user_id=user.id, provider="email")

    return MessageResponse(message="Password updated successfully. You can now sign in.")


# ── Google OAuth ──────────────────────────────────────────────────────────────

@router.get("/oauth/google")
async def google_oauth_start(request: Request):
    if not settings.google_client_id:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED,
                            detail="Google OAuth not configured. Set GOOGLE_CLIENT_ID in .env")
    return await oauth.google.authorize_redirect(request, settings.google_redirect_uri)


@router.get("/oauth/google/callback")
async def google_oauth_callback(request: Request, db: Session = Depends(get_db)):
    """
    Google OAuth callback — exchange code for tokens, upsert user, issue our JWT.

    Learning note: after Google confirms identity we stop using Google's token.
    We issue our OWN JWT. From this point the frontend talks only to our API.
    """
    if not settings.google_client_id:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED,
                            detail="Google OAuth not configured")

    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"OAuth failed: {e}")

    user_info = token.get("userinfo") or {}
    email     = user_info.get("email")
    if not email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Could not retrieve email from Google")

    google_id = user_info.get("sub", "")
    full_name = user_info.get("name", "")
    is_new    = False

    user = (
        db.query(User).filter(User.oauth_provider == "google", User.oauth_id == google_id).first()
        or db.query(User).filter(User.email == email).first()
    )

    if user is None:
        user = User(id=str(uuid.uuid4()), email=email, full_name=full_name,
                    oauth_provider="google", oauth_id=google_id, is_verified=True)
        db.add(user)
        is_new = True
    else:
        if not user.oauth_provider:
            user.oauth_provider = "google"
            user.oauth_id = google_id
            user.is_verified = True

    db.commit()
    db.refresh(user)

    event_type = AuthEventType.register_google if is_new else AuthEventType.login_google
    _log_event(db, event_type, request, user_id=user.id, provider="google")

    return RedirectResponse(url=_welcome_redirect(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
        provider="Google",
        name=user.full_name or user.email,
    ))


# ── GitHub OAuth ──────────────────────────────────────────────────────────────

@router.get("/oauth/github")
async def github_oauth_start(request: Request):
    if not settings.github_client_id:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED,
                            detail="GitHub OAuth not configured. Set GITHUB_CLIENT_ID in .env")
    return await oauth.github.authorize_redirect(request, settings.github_redirect_uri)


@router.get("/oauth/github/callback")
async def github_oauth_callback(request: Request, db: Session = Depends(get_db)):
    """
    GitHub OAuth callback.

    GitHub-specific quirks:
    - Users can hide their email on GitHub. If /user returns no email,
      we fetch /user/emails and use the primary verified one.
    - GitHub uses integer IDs, not UUIDs. We store them as strings.
    """
    if not settings.github_client_id:
        raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED,
                            detail="GitHub OAuth not configured")

    try:
        token = await oauth.github.authorize_access_token(request)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"GitHub OAuth failed: {e}")

    # Fetch GitHub user profile
    resp = await oauth.github.get("user", token=token)
    github_user = resp.json()
    github_id = str(github_user.get("id", ""))
    full_name = github_user.get("name") or github_user.get("login", "")
    email = github_user.get("email")

    # GitHub may not include email in the profile — fetch from /user/emails
    if not email:
        emails_resp = await oauth.github.get("user/emails", token=token)
        emails = emails_resp.json()
        email = next(
            (e["email"] for e in emails if e.get("primary") and e.get("verified")),
            next((e["email"] for e in emails if e.get("verified")), None),
        )

    if not email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Could not retrieve a verified email from GitHub. "
                                   "Please make your primary email public in GitHub settings.")

    is_new = False
    user = (
        db.query(User).filter(User.oauth_provider == "github", User.oauth_id == github_id).first()
        or db.query(User).filter(User.email == email).first()
    )

    if user is None:
        user = User(id=str(uuid.uuid4()), email=email, full_name=full_name,
                    oauth_provider="github", oauth_id=github_id, is_verified=True)
        db.add(user)
        is_new = True
    else:
        if not user.oauth_provider:
            user.oauth_provider = "github"
            user.oauth_id = github_id
            user.is_verified = True

    db.commit()
    db.refresh(user)

    event_type = AuthEventType.register_github if is_new else AuthEventType.login_github
    _log_event(db, event_type, request, user_id=user.id, provider="github")

    return RedirectResponse(url=_welcome_redirect(
        access_token=create_access_token(user.id),
        refresh_token=create_refresh_token(user.id),
        provider="GitHub",
        name=user.full_name or user.email,
    ))
