"""
Pipeline Stage 2 — Text Cleaning

Normalises raw extracted text before it reaches the chunker.

Why clean before chunking?
  The chunker splits on sentence/paragraph boundaries. If those boundaries are
  buried in whitespace noise or mangled by PDF line breaks, the chunker will
  produce bad splits. Clean first, chunk on clean text.

Learning note — what makes PDF text dirty:
  1. Hyphenated line breaks: "compre-\nhension" → "comprehension"
  2. Header/footer repetition: "Confidential | Page 3" appears on every page
  3. Multiple spaces from PDF column layout
  4. Unicode ligatures: "ﬁle" (single codepoint) instead of "file" (two)
  5. Null bytes / control characters from bad PDF encoders
"""
import re
import unicodedata
import logging

from app.pipeline.types import ParsedPage, CleanedPage

logger = logging.getLogger(__name__)

# Patterns for page number noise (applied before other cleaning)
_PAGE_NUM_PATTERNS = [
    re.compile(r'^\s*\d+\s*$', re.MULTILINE),           # standalone number on its own line
    re.compile(r'Page\s+\d+\s+of\s+\d+', re.IGNORECASE),  # "Page 3 of 45"
    re.compile(r'^\s*-\s*\d+\s*-\s*$', re.MULTILINE),   # "- 3 -" style page numbers
]

# Hyphenated line break: word-\nnextword → wordnextword
_HYPHEN_LINEBREAK = re.compile(r'(\w)-\n(\w)')

# Control characters (keep \t and \n)
_CONTROL_CHARS = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

# Multiple spaces / tabs on a single line → single space
_MULTI_SPACE = re.compile(r'[ \t]+')

# 3+ consecutive newlines → 2 newlines (paragraph break)
_MULTI_NEWLINE = re.compile(r'\n{3,}')


def clean_page(page: ParsedPage) -> CleanedPage:
    """
    Apply all cleaning steps to a single ParsedPage.
    Returns a CleanedPage with before/after character counts for diagnostics.
    """
    original_len = len(page.text)
    text = page.text

    # Step 1: Unicode normalisation — converts ligatures, fancy quotes, etc.
    # NFKC: compatibility decomposition followed by canonical composition.
    # Example: "ﬁle" (U+FB01) → "fi" + "le"
    text = unicodedata.normalize("NFKC", text)

    # Step 2: Remove null bytes and control characters
    text = _CONTROL_CHARS.sub('', text)

    # Step 3: Strip page number noise before fixing whitespace
    for pattern in _PAGE_NUM_PATTERNS:
        text = pattern.sub('', text)

    # Step 4: Fix hyphenated line breaks
    # "compre-\nhension" → "comprehension"
    # Use a loop because overlapping matches (e.g. "a-\nb-\nc") need multiple passes
    while _HYPHEN_LINEBREAK.search(text):
        text = _HYPHEN_LINEBREAK.sub(r'\1\2', text)

    # Step 5: Normalise horizontal whitespace (spaces/tabs) on each line
    text = _MULTI_SPACE.sub(' ', text)

    # Step 6: Strip trailing spaces from every line
    text = '\n'.join(line.rstrip() for line in text.split('\n'))

    # Step 7: Collapse 3+ blank lines → 1 blank line (paragraph separator)
    text = _MULTI_NEWLINE.sub('\n\n', text)

    # Step 8: Final strip
    text = text.strip()

    cleaned_len = len(text)
    reduction_pct = round((1 - cleaned_len / original_len) * 100, 1) if original_len > 0 else 0
    logger.debug(
        "Cleaned page %d: %d → %d chars (%.1f%% reduction)",
        page.page_number, original_len, cleaned_len, reduction_pct,
    )

    return CleanedPage(
        text=text,
        page_number=page.page_number,
        source=page.source,
        original_char_count=original_len,
        cleaned_char_count=cleaned_len,
    )


def clean_pages(pages: list[ParsedPage]) -> list[CleanedPage]:
    """
    Clean a list of ParsedPages and discard any that are empty after cleaning.
    """
    cleaned = []
    discarded = 0
    for page in pages:
        result = clean_page(page)
        if result.text:
            cleaned.append(result)
        else:
            discarded += 1
            logger.debug("Page %d became empty after cleaning — discarding", page.page_number)

    if discarded:
        logger.info("Discarded %d empty page(s) after cleaning", discarded)

    return cleaned
