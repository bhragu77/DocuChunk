"""
Token counting & token-aware splitting for the embedding model.

Why this module exists
----------------------
The embedding model (settings.hf_embed_model, default all-MiniLM-L6-v2) has a
HARD input limit of 256 tokens (its sentence-transformers max_seq_length). Any
text longer than that is SILENTLY TRUNCATED by the model at embed time — the tail
never influences the produced vector, so retrieval quality degrades invisibly.

The chunker sizes chunks by CHARACTERS, which does not bound tokens: a dense
"800-char" chunk (numbers, code, CJK, long words) can blow past 256 tokens. This
module is the safety net that guarantees every chunk fits the model's window.

Public API
----------
  count_tokens(text)                     -> exact model token count (incl. specials)
  split_by_tokens(text, max, overlap)    -> [(sub_text, char_start, char_end)]

Tokenizer selection (best → fallback), so the pipeline never hard-fails:
  1. The embedding model's OWN tokenizer (HuggingFace transformers AutoTokenizer).
     This is the correct counter — its tokens are exactly what the model sees, and
     its offset mapping gives char-accurate split points for free.
  2. tiktoken (cl100k_base) — approximate, but deterministic and dependency-light.
  3. A char/4 heuristic — last resort if neither the model nor tiktoken is present.
"""
from functools import lru_cache
import logging

from app.config import get_settings

logger = logging.getLogger(__name__)

# The model prepends [CLS] and appends [SEP]; reserve room so a content window of
# `max_tokens - _RESERVED_SPECIAL` still fits once the model adds its specials.
_RESERVED_SPECIAL = 2
# Approximate chars-per-token used only by the last-resort heuristic path.
_CHARS_PER_TOKEN = 4


@lru_cache(maxsize=1)
def _hf_tokenizer():
    """The embedding model's own tokenizer, or None if transformers/model absent."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(get_settings().hf_embed_model)
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("HF tokenizer unavailable (%s) — using fallback tokenizer", exc)
        return None


@lru_cache(maxsize=1)
def _tiktoken_encoder():
    """A tiktoken BPE encoder, or None if tiktoken is absent."""
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - environment dependent
        return None


def count_tokens(text: str) -> int:
    """
    Number of tokens the embedding model will see for `text`, including the two
    special tokens. This is what the 256-token limit is measured against.
    """
    if not text:
        return 0
    tok = _hf_tokenizer()
    if tok is not None:
        return len(tok.encode(text, add_special_tokens=True))
    enc = _tiktoken_encoder()
    if enc is not None:
        return len(enc.encode(text)) + _RESERVED_SPECIAL
    return len(text) // _CHARS_PER_TOKEN + _RESERVED_SPECIAL


def split_by_tokens(
    text: str,
    max_tokens: int,
    overlap_tokens: int = 0,
) -> list[tuple[str, int, int]]:
    """
    Split `text` into windows of at most `max_tokens` model tokens (specials
    included), overlapping consecutive windows by `overlap_tokens` content tokens.

    Returns (sub_text, char_start, char_end) with offsets INTO `text`, so callers
    can rebase them onto the document's global char offsets and keep provenance /
    content-addressable identity exact. If `text` already fits, returns a single
    (text, 0, len(text)) — the common case, so the cost is one token count.
    """
    if max_tokens <= 0 or count_tokens(text) <= max_tokens:
        return [(text, 0, len(text))]

    # Budget for CONTENT tokens per window, leaving room for the model's specials.
    content_budget = max(1, max_tokens - _RESERVED_SPECIAL)
    step = max(1, content_budget - max(0, overlap_tokens))

    tok = _hf_tokenizer()
    if tok is not None:
        offsets = tok(
            text, add_special_tokens=False, return_offsets_mapping=True
        )["offset_mapping"]
        return _window_from_offsets(text, offsets, content_budget, step)

    enc = _tiktoken_encoder()
    if enc is not None:
        ids = enc.encode(text)
        # Build per-token char offsets via prefix decoding (decode(ids[:k]) is a
        # prefix of text, so its length is the char offset of token k).
        offsets = []
        for k in range(len(ids)):
            start = len(enc.decode(ids[:k]))
            end = len(enc.decode(ids[: k + 1]))
            offsets.append((start, end))
        return _window_from_offsets(text, offsets, content_budget, step)

    # Last resort: approximate by characters (~4 chars/token).
    approx = content_budget * _CHARS_PER_TOKEN
    step_chars = step * _CHARS_PER_TOKEN
    windows: list[tuple[str, int, int]] = []
    i = 0
    while i < len(text):
        j = min(i + approx, len(text))
        windows.append((text[i:j], i, j))
        if j >= len(text):
            break
        i += step_chars
    return windows


def _window_from_offsets(
    text: str,
    offsets: list[tuple[int, int]],
    content_budget: int,
    step: int,
) -> list[tuple[str, int, int]]:
    """Slide a token window over per-token char offsets → char-accurate sub-spans."""
    n = len(offsets)
    windows: list[tuple[str, int, int]] = []
    i = 0
    while i < n:
        j = min(i + content_budget, n)
        c_start = offsets[i][0]
        c_end = offsets[j - 1][1]
        windows.append((text[c_start:c_end], c_start, c_end))
        if j >= n:
            break
        i += step
    return windows
