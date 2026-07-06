"""
EmbeddingJob — tracks the embed stage for a document so it is auditable and
batch-resumable.

Migration note (no Alembic in this project — init_db uses Base.metadata.create_all):
  Fresh and test DBs get this schema automatically. An existing dev app.db that
  still has the OLD embedding_jobs schema should DROP the table so create_all
  rebuilds it (no embedding jobs exist yet, so nothing is lost):
      python -c "from app.database import engine; \
                 from app.models.job import EmbeddingJob; \
                 EmbeddingJob.__table__.drop(engine, checkfirst=True)"
  then restart the app (create_all recreates it).
"""
import uuid
import enum
from datetime import datetime

from sqlalchemy import String, Integer, Boolean, DateTime, ForeignKey, JSON, Enum as SAEnum, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class JobStatus(str, enum.Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    completed_with_failures = "completed_with_failures"
    failed = "failed"


class EmbeddingJob(Base):
    __tablename__ = "embedding_jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    doc_id: Mapped[str] = mapped_column(
        String, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # Which logical document version this job embedded. Lets stage-level resume ask
    # "is there a COMPLETED job for the CURRENT version?" (Part 0 doc_version).
    doc_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    total_chunks: Mapped[int] = mapped_column(Integer, default=0)
    completed_chunks: Mapped[int] = mapped_column(Integer, default=0)
    # chunk_ids that failed to embed (per-entry failures). JSON list[str].
    failed_chunk_ids: Mapped[list] = mapped_column(JSON, default=list)

    status: Mapped[JobStatus] = mapped_column(
        SAEnum(JobStatus), default=JobStatus.pending, nullable=False
    )
    # Index of the last batch that was fully embedded AND sunk (persisted). -1 means
    # no batch checkpoint yet. Only LARGE-tier jobs advance this; resume continues
    # from checkpoint_batch + 1.
    checkpoint_batch: Mapped[int] = mapped_column(Integer, default=-1, nullable=False)

    # Audit: what produced these vectors. A mismatch (e.g. an index built with a
    # different model/normalization) becomes detectable instead of silent.
    model_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    provider_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    normalize: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    def __repr__(self) -> str:
        return (
            f"<EmbeddingJob doc={self.doc_id} v{self.doc_version} status={self.status} "
            f"{self.completed_chunks}/{self.total_chunks} ckpt={self.checkpoint_batch}>"
        )
