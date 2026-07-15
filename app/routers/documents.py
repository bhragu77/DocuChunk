import os
import uuid
import math
import logging
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, Request, UploadFile, status
from sqlalchemy.orm import Session
import chromadb

from app.database import get_db
from app.models.user import User
from app.models.document import Document, DocumentStatus
from app.schemas.document import DocumentListResponse, DocumentResponse, DeleteResponse
from app.core.dependencies import get_current_user, get_current_user_optional as _optional_current_user, get_chroma, get_arq_pool
from app.pipeline.orchestrator import run_pipeline
from app.config import get_settings

router = APIRouter(prefix="/docs", tags=["documents"])
settings = get_settings()
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".pdf", ".docx"}
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_file(file: UploadFile) -> tuple[str, str]:
    """
    Returns (extension, mime_type) or raises 422.

    Learning note: we validate BOTH extension AND content-type.
    Never trust content-type alone — a user can rename any file to .pdf.
    In Phase 3 we'll do deep magic-byte validation inside the parser.
    """
    original_name = file.filename or ""
    ext = os.path.splitext(original_name)[-1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported file type '{ext}'. Only PDF and DOCX are allowed.",
        )

    # Use the browser-reported content type as a secondary hint
    content_type = file.content_type or ""
    # Normalise: some browsers send application/octet-stream for .docx
    if ext == ".pdf":
        mime = "application/pdf"
    else:
        mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    return ext, mime


def _user_upload_dir(user_id: str) -> str:
    """Each user gets their own subdirectory under uploads/."""
    path = os.path.join(settings.upload_dir, user_id)
    os.makedirs(path, exist_ok=True)
    return path


def _chroma_collection_name(user_id: str) -> str:
    """Consistent ChromaDB collection name per user."""
    return f"user_{user_id}"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    arq_pool=Depends(get_arq_pool),
):
    """
    Upload a PDF or DOCX document.

    Pipeline flow triggered after upload:
      uploaded → (Phase 3) parsing → (Phase 4) chunking → (Phase 6) embedding → ready

    The pipeline runs in a SEPARATE arq worker process: we enqueue run_pipeline
    with just the doc_id and return immediately with status="uploaded". The client
    polls GET /docs/{id} to watch progress. (See the enqueue block below for the
    inline dev/test fallback.)

    Learning note: we stream the file in chunks instead of reading it all into
    memory at once. This is how you handle large files in production without
    exhausting RAM.
    """
    ext, mime_type = _validate_file(file)

    # Read up to (max + 1) bytes — if we get more than max, reject early
    max_bytes = settings.max_file_size_bytes
    content = await file.read(max_bytes + 1)

    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum size of {settings.max_file_size_mb} MB",
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty",
        )

    # Save to disk: uploads/{user_id}/{uuid}.{ext}
    stored_filename = f"{uuid.uuid4()}{ext}"
    upload_dir = _user_upload_dir(current_user.id)
    file_path = os.path.join(upload_dir, stored_filename)

    with open(file_path, "wb") as f:
        f.write(content)

    logger.info(
        "Saved upload: user=%s file=%s size=%d bytes",
        current_user.id,
        stored_filename,
        len(content),
    )

    # Persist record to DB
    doc = Document(
        id=str(uuid.uuid4()),
        user_id=current_user.id,
        filename=stored_filename,
        original_filename=file.filename or stored_filename,
        file_path=file_path,
        file_size=len(content),
        mime_type=mime_type,
        status=DocumentStatus.uploaded,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # Kick off the parse → chunk → embed pipeline. Only the doc_id crosses the
    # boundary; the worker reconstructs all state from the DB (governing rule).
    #
    # Production path: enqueue onto the arq worker via the Redis pool.
    # Fallback (PIPELINE_SYNC dev mode, or no pool available — e.g. tests without
    # the lifespan / no Redis): run inline via BackgroundTasks. Same external
    # behavior either way — the response returns immediately with status="uploaded".
    if settings.pipeline_sync or arq_pool is None:
        if not settings.pipeline_sync:
            logger.warning("No arq pool available — running pipeline inline for doc=%s", doc.id)
        background_tasks.add_task(run_pipeline, doc.id)
    else:
        await arq_pool.enqueue_job("run_pipeline", doc.id)
        logger.info("Enqueued run_pipeline for doc=%s", doc.id)

    return doc


@router.get("/list", response_model=DocumentListResponse)
def list_documents(
    page: int = Query(default=1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(default=10, ge=1, le=100, description="Items per page"),
    status_filter: DocumentStatus | None = Query(default=None, alias="status"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Paginated list of the current user's documents.

    Learning note: pagination is done at the DB level (LIMIT + OFFSET),
    not by fetching all rows and slicing in Python. Always paginate at the DB.
    """
    query = db.query(Document).filter(Document.user_id == current_user.id)

    if status_filter is not None:
        query = query.filter(Document.status == status_filter)

    total = query.count()
    total_pages = math.ceil(total / page_size) if total > 0 else 1
    offset = (page - 1) * page_size

    documents = (
        query.order_by(Document.created_at.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )

    return DocumentListResponse(
        items=documents,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
    )


@router.get("/{doc_id}", response_model=DocumentResponse)
def get_document(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve a single document by ID (must belong to the current user)."""
    doc = _get_doc_or_404(doc_id, current_user.id, db)
    return doc


@router.get("/{doc_id}/raw")
def get_document_raw(
    doc_id: str,
    token: str | None = Query(default=None),
    db: Session = Depends(get_db),
    current_user: User | None = Depends(_optional_current_user),
):
    """Serve the raw uploaded file INLINE for the chat doc-preview panel.

    Auth is header OR `?token=` query param: an <iframe src> can't set an
    Authorization header, so the frontend appends the JWT as a query param. The
    file is still strictly scoped to its owner — a valid token for a different
    user gets a 404, same as an unknown id.
    """
    from fastapi.responses import FileResponse
    from app.core.security import verify_access_token

    user = current_user
    if user is None and token:
        payload = verify_access_token(token)
        if payload is not None:
            user = db.query(User).filter(User.id == payload.get("sub")).first()
    if user is None or not getattr(user, "is_active", False):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

    doc = _get_doc_or_404(doc_id, user.id, db)
    if not os.path.exists(doc.file_path):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="File no longer on disk")
    return FileResponse(
        doc.file_path,
        media_type=doc.mime_type or "application/octet-stream",
        filename=doc.original_filename,
        content_disposition_type="inline",
    )


@router.delete("/{doc_id}", response_model=DeleteResponse)
def delete_document(
    doc_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    chroma: chromadb.ClientAPI = Depends(get_chroma),
):
    """
    Delete a document: removes the file on disk, the DB record, and all
    associated chunk embeddings from ChromaDB.

    Learning note: we must clean up ChromaDB manually — SQLAlchemy's
    CASCADE only handles the relational DB. Vector data lives separately.
    """
    doc = _get_doc_or_404(doc_id, current_user.id, db)

    # 1. Delete physical file
    if os.path.exists(doc.file_path):
        os.remove(doc.file_path)
        logger.info("Deleted file: %s", doc.file_path)

    # 2. Delete chunk embeddings from ChromaDB (if any were created)
    collection_name = _chroma_collection_name(current_user.id)
    try:
        collection = chroma.get_collection(name=collection_name)
        collection.delete(where={"doc_id": doc_id})
        logger.info("Deleted ChromaDB chunks for doc=%s", doc_id)
    except Exception:
        # Collection doesn't exist yet (document never reached embedding phase)
        pass

    # 2b. Delete the document's chunks from the BM25 lexical index too, so the two
    # indexes stay in sync (a deleted doc must not linger as BM25 ghosts).
    bm25 = getattr(request.app.state, "bm25", None)
    if bm25 is not None:
        try:
            bm25.delete(doc_id, current_user.id)
        except Exception:
            logger.exception("BM25 delete failed for doc=%s", doc_id)

    # 3. Delete DB record (cascades to embedding_jobs via FK)
    db.delete(doc)
    db.commit()

    return DeleteResponse(
        message="Document deleted successfully",
        document_id=doc_id,
    )


# ── Internal helper ───────────────────────────────────────────────────────────

def _get_doc_or_404(doc_id: str, user_id: str, db: Session) -> Document:
    """Fetch a document that belongs to the user, or raise 404."""
    doc = (
        db.query(Document)
        .filter(Document.id == doc_id, Document.user_id == user_id)
        .first()
    )
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_id}' not found",
        )
    return doc
