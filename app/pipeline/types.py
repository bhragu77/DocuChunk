"""
Internal data structures that flow through the pipeline stages.

Learning note: these are plain Python dataclasses — not Pydantic models, not SQLAlchemy
models. They live only in memory during a pipeline run. They carry data FROM one stage
TO the next (parser → cleaner → chunker → embedder → vector store).
"""
from dataclasses import dataclass, field


@dataclass
class ParsedPage:
    """
    Raw text extracted from one page (PDF) or one paragraph group (DOCX).
    Text is dirty at this stage — whitespace issues, hyphenated lines, noise.
    """
    text: str
    page_number: int    # 1-indexed; for DOCX this is a logical grouping
    source: str         # original filename


@dataclass
class CleanedPage:
    """ParsedPage after the cleaner has normalised the text."""
    text: str
    page_number: int
    source: str
    original_char_count: int   # characters before cleaning
    cleaned_char_count: int    # characters after cleaning (always <= original)


@dataclass
class Chunk:
    """
    A segment of text produced by the chunker.
    One CleanedPage → one or more Chunks.

    Learning note: char_start and char_end are positions relative to the
    reconstructed full-document text (all pages joined). They let you map
    a chunk back to its exact location in the original document.
    """
    chunk_id: str
    doc_id: str
    text: str
    chunk_index: int          # 0-based position in the chunk list for this doc
    page_number: int          # which page this chunk originated from
    char_start: int           # character offset in full-document text
    char_end: int
    token_count: int
    # Content-addressable identity (Phase 4 amendment). See chunker.make_chunk_id /
    # make_content_hash. These make re-indexing stable: chunk_id is deterministic from
    # (doc_id, char_start, char_end) so a re-upload overwrites the right vectors instead
    # of duplicating stale content; content_hash lets Phase 7 skip re-embedding unchanged
    # chunks; doc_version tags which logical version of the document produced this chunk.
    content_hash: str = ""    # sha256(text)[:32]; changes iff the chunk text changes
    doc_version: int = 1      # Document.version at chunk-creation time
    metadata: dict = field(default_factory=dict)


@dataclass
class EmbeddedChunk:
    """
    A Chunk paired with its embedding vector, ready for ChromaDB.

    Identity pass-through: chunk_id / content_hash / doc_version are copied verbatim
    from the source Chunk in __post_init__ — the embedder NEVER mints new identity.
    Part 2 (vector store) uses chunk_id as the primary key and content_hash /
    doc_version as metadata for the incremental-index contract.
    """
    chunk: Chunk
    embedding: list[float]    # 384-dim for all-MiniLM-L6-v2
    chunk_id: str = ""
    content_hash: str = ""
    doc_version: int = 1

    def __post_init__(self):
        # Always mirror the source chunk's identity — do not trust caller-supplied
        # values, so identity can never drift from the chunk it belongs to.
        self.chunk_id = self.chunk.chunk_id
        self.content_hash = self.chunk.content_hash
        self.doc_version = self.chunk.doc_version
