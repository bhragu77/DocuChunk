"""
Analysis & visualisation endpoints.

Exposes the pipeline's chunk analyser, per-document embedding inspection, and a
UMAP 2-D projection of a document's embeddings to the frontend. These read data
that the ingestion pipeline already produced:

  * chunks     — the `chunks` table (ChunkRecord), analysed via pipeline.analyser
  * embeddings — the owner's per-user Chroma collection ("user_{user_id}")

All endpoints are owner-scoped: a user can only inspect their own documents.
"""
import logging
from typing import Any

import chromadb
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user, get_chroma
from app.database import get_db
from app.models.chunk import ChunkRecord
from app.models.document import Document
from app.models.user import User
from app.pipeline.analyser import analyse_chunks
from app.pipeline.vector_store import collection_name

router = APIRouter(prefix="/analysis", tags=["analysis"])
logger = logging.getLogger(__name__)

# How many leading vector components to expose as a "preview" in the embeddings view.
_EMBED_PREVIEW_DIMS = 8
# Characters of chunk text to surface in list/point views.
_EXCERPT_CHARS = 240


def _as_list(x) -> list:
    """None-safe coercion for Chroma's embeddings, which come back as a numpy
    array (truthiness is ambiguous) or None when the collection is empty."""
    if x is None:
        return []
    return list(x)


def _owned_doc(doc_id: str, user_id: str, db: Session) -> Document:
    """Fetch a document the current user owns, or 404."""
    doc = (
        db.query(Document)
        .filter(Document.id == doc_id, Document.user_id == user_id)
        .first()
    )
    if doc is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")
    return doc


def _doc_embeddings(chroma, user_id: str, doc_id: str) -> dict:
    """
    Pull all stored vectors + metadata for one document from the owner's Chroma
    collection. Returns Chroma's get() dict (keys: ids, embeddings, metadatas,
    documents). Missing collection → empty result rather than an error.
    """
    try:
        collection = chroma.get_collection(name=collection_name(user_id))
    except Exception:
        return {"ids": [], "embeddings": [], "metadatas": [], "documents": []}
    return collection.get(
        where={"doc_id": doc_id},
        include=["embeddings", "metadatas", "documents"],
    )


# ── Chunk Analyser ──────────────────────────────────────────────────────────────

@router.get("/{doc_id}/chunks")
def analyse_document_chunks(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Quality statistics + the chunk list for one document.

    `stats` is the pipeline analyser's output (size distribution, token stats,
    overlap coverage, per-page counts, anomaly warnings). `chunks` is the ordered
    list of chunks with offsets and a text excerpt for a browsable table.
    """
    doc = _owned_doc(doc_id, current_user.id, db)

    records = (
        db.query(ChunkRecord)
        .filter(ChunkRecord.doc_id == doc_id, ChunkRecord.doc_version == doc.version)
        .order_by(ChunkRecord.chunk_index)
        .all()
    )

    chunks = [r.to_chunk() for r in records]
    # pages=[] — cleaned pages are transient to the ingestion run and not persisted;
    # the analyser only uses them for one optional "dropped page" warning.
    stats = analyse_chunks(chunks, pages=[])

    return {
        "doc_id": doc_id,
        "filename": doc.original_filename,
        "status": doc.status.value if hasattr(doc.status, "value") else doc.status,
        "stats": stats,
        "chunks": [
            {
                "chunk_index": r.chunk_index,
                "page_number": r.page_number,
                "char_start": r.char_start,
                "char_end": r.char_end,
                "chars": r.char_end - r.char_start,
                "token_count": r.token_count,
                "text": r.text[:_EXCERPT_CHARS],
            }
            for r in records
        ],
    }


# ── Embeddings ──────────────────────────────────────────────────────────────────

@router.get("/{doc_id}/embeddings")
def document_embeddings(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    chroma: chromadb.ClientAPI = Depends(get_chroma),
) -> dict[str, Any]:
    """
    Per-document embedding inspection: vector count, dimensionality, the model
    that produced them, and a per-chunk preview (L2 norm + leading components).
    """
    doc = _owned_doc(doc_id, current_user.id, db)
    res = _doc_embeddings(chroma, current_user.id, doc_id)

    # Chroma returns `embeddings` as a numpy array (or None) — never use `or []`
    # on it (ambiguous truth value). metadatas/documents come back as plain lists.
    embeddings = _as_list(res.get("embeddings"))
    metadatas = res.get("metadatas") or []
    documents = res.get("documents") or []

    dim = len(embeddings[0]) if len(embeddings) else 0

    items = []
    for emb, meta, text in zip(embeddings, metadatas, documents):
        norm = sum(v * v for v in emb) ** 0.5
        items.append(
            {
                "chunk_index": (meta or {}).get("chunk_index"),
                "page_number": (meta or {}).get("page_number"),
                "norm": round(norm, 4),
                "preview": [round(float(v), 4) for v in emb[:_EMBED_PREVIEW_DIMS]],
                "text_excerpt": (text or "")[:_EXCERPT_CHARS],
            }
        )
    items.sort(key=lambda x: (x["chunk_index"] is None, x["chunk_index"]))

    model = metadatas[0].get("strategy") if metadatas else None  # placeholder if unset
    return {
        "doc_id": doc_id,
        "filename": doc.original_filename,
        "count": len(embeddings),
        "dimensions": dim,
        "model": model,
        "chunks": items,
    }


# ── Visualise (UMAP) ────────────────────────────────────────────────────────────

@router.get("/{doc_id}/visualise")
def visualise_embeddings(
    doc_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    chroma: chromadb.ClientAPI = Depends(get_chroma),
) -> dict[str, Any]:
    """
    Project the document's embeddings into 2-D for plotting.

    UMAP (cosine metric) is the primary reducer. UMAP needs a handful of points to
    be meaningful, so for very small documents we fall back to a PCA projection,
    and for 1-2 points to fixed coordinates. The response carries `method` so the
    frontend can label which reducer was used.
    """
    _owned_doc(doc_id, current_user.id, db)
    res = _doc_embeddings(chroma, current_user.id, doc_id)

    embeddings = _as_list(res.get("embeddings"))
    metadatas = res.get("metadatas") or []
    documents = res.get("documents") or []
    n = len(embeddings)

    if n == 0:
        return {"doc_id": doc_id, "method": None, "count": 0, "dimensions": 0, "points": []}

    coords, method = _project_2d(embeddings)

    # True nearest neighbours in the ORIGINAL high-dim space (cosine similarity),
    # not the 2-D projection — so "closest" reflects real semantic distance rather
    # than a projection artifact. Positions map to chunk_index via metadatas.
    nearest = _nearest_neighbors(embeddings, k=3)
    idx_to_chunk = [(m or {}).get("chunk_index") for m in metadatas]

    points = []
    for i, ((x, y), meta, text) in enumerate(zip(coords, metadatas, documents)):
        nn = [
            {"chunk_index": idx_to_chunk[j], "similarity": round(sim, 4)}
            for j, sim in nearest[i]
        ]
        points.append(
            {
                "x": round(float(x), 4),
                "y": round(float(y), 4),
                "chunk_index": (meta or {}).get("chunk_index"),
                "page_number": (meta or {}).get("page_number"),
                "source": (meta or {}).get("source"),
                "text_excerpt": (text or "")[:_EXCERPT_CHARS],
                "nearest": nn,
            }
        )
    points.sort(key=lambda p: (p["chunk_index"] is None, p["chunk_index"]))

    return {
        "doc_id": doc_id,
        "method": method,
        "count": n,
        "dimensions": len(embeddings[0]),
        "points": points,
    }


def _nearest_neighbors(embeddings, k: int = 3) -> list[list[tuple[int, float]]]:
    """
    For each embedding, return its top-k most similar OTHER embeddings as
    (position_index, cosine_similarity), sorted most-similar first.

    Cosine similarity is computed in the full embedding space (vectors are
    L2-normalised defensively first), so the ranking reflects true semantic
    proximity — independent of the 2-D projection used for plotting.
    """
    import numpy as np

    x = np.asarray(embeddings, dtype=float)
    n = x.shape[0]
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    xn = x / norms
    sim = xn @ xn.T
    np.fill_diagonal(sim, -np.inf)  # never a neighbour of itself

    kk = min(k, n - 1)
    out: list[list[tuple[int, float]]] = []
    for i in range(n):
        if kk <= 0:
            out.append([])
            continue
        order = np.argsort(-sim[i])[:kk]
        out.append([(int(j), float(sim[i, j])) for j in order])
    return out


def _project_2d(embeddings: list[list[float]]) -> tuple[list, str]:
    """Reduce embeddings to 2-D. Returns (coords, method_name)."""
    import numpy as np

    x = np.asarray(embeddings, dtype=float)
    n = x.shape[0]

    if n <= 2:
        # Nothing to project meaningfully — place points on a short line.
        return [[float(i), 0.0] for i in range(n)], "trivial"

    if n < 10:
        # Too few points for a stable UMAP manifold — use PCA (SVD) instead.
        return _pca_2d(x), "pca"

    try:
        import umap  # umap-learn

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(15, n - 1),
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        return reducer.fit_transform(x).tolist(), "umap"
    except Exception as exc:  # pragma: no cover - defensive fallback
        logger.warning("UMAP failed (%s) — falling back to PCA", exc)
        return _pca_2d(x), "pca"


def _pca_2d(x) -> list:
    """Top-2 principal components via SVD (mean-centred)."""
    import numpy as np

    centered = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return (centered @ vt[:2].T).tolist()
