# Retrieval improvements — measured

Same fixture, same ground truth, same embeddings; k = 5. Retrieval profile: **neural**. Rewriter: `gemini/gemini-flash-lite-latest`.

## Aggregate

| config | MRR | recall@k | precision@k | p50 retrieval |
|---|:--:|:--:|:--:|:--:|
| **baseline** | 0.967 | 0.978 | 0.316 | 2004 ms |
| **+rewrite** | 0.967 | 0.978 | 0.316 | 4158 ms |
| **+mmr** | 0.967 | 0.978 | 0.311 | 3611 ms |
| **+both** | 0.967 | 0.978 | 0.311 | 4431 ms |

## Per category — MRR

Aggregates hide the trade. Both techniques are expected to help some categories and hurt others; this table is where the decision is made.

| category | baseline | +rewrite | +mmr | +both |
|---|:--:|:--:|:--:|:--:|
| ambiguous_entity | 1.000 | 1.000 = | 1.000 = | 1.000 = |
| general | 0.857 | 0.857 = | 0.857 = | 0.857 = |
| identifier | 0.967 | 0.967 = | 0.967 = | 0.967 = |
| multi_hop | 1.000 | 1.000 = | 1.000 = | 1.000 = |

## Per category — recall@k

| category | baseline | +rewrite | +mmr | +both |
|---|:--:|:--:|:--:|:--:|
| ambiguous_entity | 1.000 | 1.000 = | 1.000 = | 1.000 = |
| general | 0.857 | 0.857 = | 0.857 = | 0.857 = |
| identifier | 1.000 | 1.000 = | 1.000 = | 1.000 = |
| multi_hop | 1.000 | 1.000 = | 1.000 = | 1.000 = |

## Verdict — no measurable gain, and the reason matters

Neither technique moved MRR or recall@k in any category. That is **not** evidence
they do not work; it is evidence this fixture cannot detect it.

Three of the four categories are already **at the ceiling** in the baseline —
`ambiguous_entity`, `multi_hop` and `identifier` recall all sit at 1.000, and
identifier MRR at 0.967. There is no headroom for a retrieval enhancement to
recover. The single category with room, `general` at 0.857, did not move either;
its remaining errors are not the kind expansion or diversification addresses.

This is the same class of problem that made the first agent-vs-RAG comparison
undecidable: an evaluation whose baseline saturates cannot measure an improvement.
That was fixed by growing the corpus from 10 to 42 documents with near-duplicate
distractors. Testing these two techniques properly needs the same treatment again —
a fixture where hybrid+rerank does **not** already score 0.967.

What the run does establish, and what it costs:

| | baseline | +rewrite | +mmr | +both |
|---|:--:|:--:|:--:|:--:|
| p50 retrieval | 2004 ms | **4158 ms** | 3611 ms | **4431 ms** |
| precision@k | 0.316 | 0.316 | 0.311 | 0.311 |
| extra LLM calls/query | 0 | **1** | 0 | **1** |

Expansion roughly **doubles retrieval latency** and adds an LLM call per query — real
money on the hot path (`eval/COST_LATENCY.md`). MMR slightly *reduces* precision
(0.316 → 0.311), which is the predicted cost of trading relevance for coverage on
single-fact lookups, showing up here with no compensating recall gain because recall
is already 1.000.

**Decision: both ship disabled** (`QUERY_REWRITE_ENABLED=false`, `MMR_ENABLED=false`).
On this corpus they are pure cost. The implementation and this harness stay so the
question can be re-asked on a corpus with headroom, where the trade may well go the
other way.

## Hypotheses under test

- **Expansion should hurt `identifier`.** A part number is already the exact token BM25 needs; every paraphrase is strictly worse, and fusing bad lists with the good one demotes the correct hit.
- **Expansion should help `ambiguous_entity` / `multi_hop`,** where the user's wording and the document's wording differ.
- **MMR should help `multi_hop` coverage** by breaking up blocks of near-duplicates so a second fact fits in the budget, and slightly hurt precision on single-fact lookups.

A negative result is a valid outcome and is published unchanged. Both features ship **disabled by default** (`QUERY_REWRITE_ENABLED`, `MMR_ENABLED`); the table above is what an operator would use to decide whether their query mix justifies the cost — expansion adds one LLM call per query, MMR adds one embedding per candidate.
