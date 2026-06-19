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
    metadata: dict = field(default_factory=dict)


@dataclass
class EmbeddedChunk:
    """A Chunk paired with its embedding vector, ready for ChromaDB."""
    chunk: Chunk
    embedding: list[float]    # 384-dim for all-MiniLM-L6-v2
