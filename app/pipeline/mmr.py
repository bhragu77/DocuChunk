"""
Maximal Marginal Relevance — trade a little relevance for coverage.

THE PROBLEM IT SOLVES
=====================
Hybrid retrieval plus a cross-encoder optimises each chunk INDEPENDENTLY for
relevance to the query. Nothing in that pipeline penalises redundancy, so when a
corpus contains near-duplicate passages — which this one deliberately does, the
fixture is built with distractor documents — the top-k can be five restatements of
the same fact from the same document.

For a single-fact lookup that is harmless: the answer is in slot one either way.
For a MULTI-HOP question it is fatal, because the second fact needed to complete the
chain never makes it into the budget. This is the failure mode the agent loop works
around by retrieving repeatedly; MMR attacks it directly in one pass.

    score(d) = λ · rel(d) − (1 − λ) · max_sim(d, already_selected)

λ = 1.0 is pure relevance (a no-op). λ = 0.0 is pure diversity, which happily
returns five irrelevant-but-different chunks. The useful range is 0.5–0.8.

COST
----
MMR needs chunk-to-chunk similarity, which needs chunk embeddings. Those are not
carried on ScoredChunk, so callers pass an `embed_fn` and the candidate texts are
embedded here. That is `len(candidates)` extra embedding calls per query — cheap
with a local MiniLM, and the reason this is opt-in rather than always on.
"""
from __future__ import annotations

import logging
from typing import Callable, Sequence

from app.pipeline.retrieval import ScoredChunk

logger = logging.getLogger(__name__)

DEFAULT_LAMBDA = 0.7


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _relevance(c: ScoredChunk) -> float:
    """Prefer the cross-encoder score; fall back to the RRF fused score."""
    return float(c.reranker_score if c.reranker_score is not None else c.fused_score)


def _normalise(values: list[float]) -> list[float]:
    """Min-max to [0, 1].

    Required, not cosmetic: cross-encoder scores are unbounded logits (routinely
    -10..+10) while the similarity term is a cosine in [-1, 1]. Mixing those raw
    means λ does not actually control the trade-off — the relevance term dominates
    at any λ and MMR silently degenerates into plain reranking.
    """
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def mmr_rerank(
    candidates: list[ScoredChunk],
    embed_fn: Callable[[str], list[float]],
    *,
    top_k: int = 5,
    lambda_mult: float = DEFAULT_LAMBDA,
) -> list[ScoredChunk]:
    """Re-order `candidates` for relevance AND mutual diversity, returning top_k.

    Input order is assumed to be best-first (post-rerank). Returns the original
    ScoredChunk objects unmodified — MMR changes ORDER and SELECTION, never scores,
    so downstream per-stage score reporting stays truthful about what produced them.
    """
    if not candidates:
        return []
    if top_k >= len(candidates):
        # Nothing to select between; MMR would only permute a set that is returned
        # whole anyway.
        return list(candidates)
    if lambda_mult >= 1.0:
        return list(candidates[:top_k])

    rel = _normalise([_relevance(c) for c in candidates])

    try:
        vectors = [embed_fn(c.text) for c in candidates]
    except Exception:
        # Diversity is an enhancement; failing to embed must not fail the query.
        logger.warning("MMR: embedding candidates failed, falling back to input order",
                       exc_info=True)
        return list(candidates[:top_k])

    selected: list[int] = [0]                      # always seed with the most relevant
    remaining = set(range(1, len(candidates)))

    while len(selected) < top_k and remaining:
        best_idx, best_score = None, float("-inf")
        for i in remaining:
            redundancy = max(_cosine(vectors[i], vectors[j]) for j in selected)
            score = lambda_mult * rel[i] - (1.0 - lambda_mult) * redundancy
            if score > best_score:
                best_idx, best_score = i, score
        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.discard(best_idx)

    return [candidates[i] for i in selected]
