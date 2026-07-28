from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from typing import Generator
from app.config import get_settings

settings = get_settings()

# connect_args only needed for SQLite (allows multi-thread access)
connect_args = {"check_same_thread": False} if "sqlite" in settings.database_url else {}

engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    echo=settings.debug,  # logs all SQL in dev
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency — yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables. Called once on app startup."""
    # Import models here so SQLAlchemy registers them before create_all
    from app.models import user, document, job, chat, pipeline_run  # noqa: F401
    Base.metadata.create_all(bind=engine)
