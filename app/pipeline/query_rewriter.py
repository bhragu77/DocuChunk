"""
Multi-query expansion — retrieve for several phrasings, fuse the results.

THE PROBLEM IT SOLVES
=====================
A single query is one point in embedding space. If the user's wording differs from
the document's wording, dense retrieval misses — and BM25 misses too, because the
words genuinely are not there. Expansion generates a few alternative phrasings,
retrieves for each, and fuses the ranked lists with RRF (the same fusion already
used to combine dense and BM25, so no new ranking concept is introduced).

WHERE IT SHOULD HURT
--------------------
This is expected to REGRESS identifier queries, and the eval reports per category
specifically to catch it. "88-AZ-0097" is already the exact token BM25 needs; every
paraphrase of a part number is strictly worse than the original, and fusing four
bad lists with one good one demotes the correct hit. The hypothesis under test is
that expansion helps ambiguous-entity and multi-hop queries enough to justify that
loss — and if it does not, the negative result is the finding.

COST
----
One extra LLM call per query, before retrieval. That is a real latency and money
cost on the hot path (see eval/COST_LATENCY.md), which is why this is opt-in.
Rewrites are cached in-process so a repeated query does not pay twice.
"""
from __future__ import annotations

import logging
import re
from typing import Callable

logger = logging.getLogger(__name__)

DEFAULT_N_REWRITES = 3

REWRITE_PROMPT = """Rewrite the search query below into {n} alternative phrasings \
that would match the same information written in different words.

Rules:
- Keep every identifier, part number, model number, name and figure EXACTLY as written.
- Vary the wording and sentence shape, not the meaning.
- One rewrite per line. No numbering, no commentary.

Query: {query}"""

# Bounded cache: expansion is deterministic at temperature 0, and eval harnesses
# replay the same queries repeatedly. Unbounded would leak in a long-lived process.
_CACHE: dict[tuple[str, int], list[str]] = {}
_CACHE_MAX = 512


def _parse_rewrites(raw: str, original: str, n: int) -> list[str]:
    out: list[str] = []
    seen = {original.strip().lower()}
    for line in (raw or "").splitlines():
        # Strip list markers the model adds despite being told not to.
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"')
        if not line or len(line) < 3:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
        if len(out) >= n:
            break
    return out


def rewrite_query(
    query: str,
    llm_fn: Callable[[str], str],
    *,
    n: int = DEFAULT_N_REWRITES,
) -> list[str]:
    """Return `query` followed by up to `n` alternative phrasings.

    The original is ALWAYS first and always present. If the model fails, returns
    `[query]` — expansion degrading to plain retrieval is correct behaviour; a
    failed rewrite must never fail the search.
    """
    key = (query, n)
    if key in _CACHE:
        return _CACHE[key]

    variants = [query]
    try:
        raw = llm_fn(REWRITE_PROMPT.format(n=n, query=query))
        variants.extend(_parse_rewrites(raw, query, n))
    except Exception:
        logger.warning("query rewrite failed, using original query only", exc_info=True)

    if len(_CACHE) < _CACHE_MAX:
        _CACHE[key] = variants
    return variants


def fuse_ranked_lists(lists: list[list[str]], rrf_k: int = 60) -> list[str]:
    """Reciprocal Rank Fusion over several ranked ID lists.

    Deliberately the same fusion (and same k) used to combine dense and BM25, so
    expansion introduces no new ranking behaviour to reason about — an ID appearing
    high in several phrasings' results rises, which is exactly the signal wanted.
    """
    scores: dict[str, float] = {}
    for ranked in lists:
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    return [cid for cid, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def clear_cache() -> None:
    """Test hook — expansion is cached, so tests must be able to reset it."""
    _CACHE.clear()
