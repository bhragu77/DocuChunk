from pydantic import BaseModel, ConfigDict
from datetime import datetime
from app.models.document import DocumentStatus


class DocumentResponse(BaseModel):
    id: str
    original_filename: str
    file_size: int
    mime_type: str
    status: DocumentStatus
    chunk_count: int
    page_count: int
    error_message: str | None
    chunker_config: dict | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class DeleteResponse(BaseModel):
    message: str
    document_id: str
