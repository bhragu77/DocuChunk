import uuid
from datetime import datetime
from sqlalchemy import String, Integer, DateTime, ForeignKey, JSON, func, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column
import enum
from app.database import Base


class DocumentStatus(str, enum.Enum):
    uploaded = "uploaded"
    parsing = "parsing"
    chunking = "chunking"
    embedding = "embedding"
    ready = "ready"
    failed = "failed"


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    # File info
    filename: Mapped[str] = mapped_column(String(255), nullable=False)           # stored filename (UUID-based)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)  # user's original name
    file_path: Mapped[str] = mapped_column(String(512), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)              # bytes
    mime_type: Mapped[str] = mapped_column(String(100), nullable=False)

    # Pipeline state
    status: Mapped[DocumentStatus] = mapped_column(
        SAEnum(DocumentStatus), default=DocumentStatus.uploaded, nullable=False
    )
    error_message: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    # Results
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    chunker_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # Which vector store holds THIS document's embeddings. Chosen at upload and
    # then immutable: the vectors physically live in one backend, so changing this
    # value without re-indexing would point retrieval at an empty namespace.
    # Nullable so rows written before per-document routing existed keep working —
    # they are read as the configured default.
    vector_backend: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # Logical version of the document's content. Copied onto every Chunk at
    # chunk-creation time (see chunker.chunk_document(doc_version=...)) and used by
    # Phase 7 as the vector store's version field for incremental re-indexing.
    #
    # INCREMENT HOOK (re-upload not built yet): when re-upload of the SAME logical
    # document is implemented, increment this before re-running the pipeline —
    #     doc.version += 1
    # so the new run's chunks carry a higher doc_version and Phase 7 can tombstone
    # any chunk_ids from the previous version that are absent in the new one. Until
    # then every document stays at version 1.
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())

    def __repr__(self) -> str:
        return f"<Document id={self.id} name={self.original_filename} status={self.status}>"
