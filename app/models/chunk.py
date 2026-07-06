from sqlalchemy import String, Integer, Text, ForeignKey, JSON
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base


class ChunkRecord(Base):
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    doc_id: Mapped[str] = mapped_column(
        String, ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    # Content-addressable identity (Phase 4 amendment). content_hash changes iff the
    # chunk text changes; doc_version tags the document version that produced it.
    content_hash: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    doc_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    chunk_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    def to_chunk(self):
        from app.pipeline.types import Chunk
        return Chunk(
            chunk_id=self.id,
            doc_id=self.doc_id,
            text=self.text,
            chunk_index=self.chunk_index,
            page_number=self.page_number,
            char_start=self.char_start,
            char_end=self.char_end,
            token_count=self.token_count,
            content_hash=self.content_hash or "",
            doc_version=self.doc_version,
            metadata=self.chunk_metadata or {},
        )

    def __repr__(self) -> str:
        return f"<ChunkRecord doc={self.doc_id} idx={self.chunk_index} chars=[{self.char_start},{self.char_end})>"
