# Cost & latency — measured

Aggregated from **10 recorded pipeline runs** — real executions, not a synthetic benchmark. Every run already stores per-stage timing and per-call token usage and cost; this groups them by the dimensions that drive a decision.

## End-to-end

- **p50:** 8726 ms · **p95:** 22219 ms · **max:** 22219 ms
- **Cost per query (avg):** ₹0.0723
- **Cost per 1,000 queries:** ₹72.27

## By model tier

| model | runs | p50 | p95 | avg tokens in/out | cost/query | cost/1k queries |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| `gemini-flash-lite-latest` | 10 | 8726 ms | 22219 ms | 6004 / 436 | ₹0.0723 | ₹72.27 |

## By stage — where time and money go

| stage | n | p50 | p95 | max | share of p50 | avg cost |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| verify | 8 | 3179.7 ms | 15531.6 ms | 15531.6 ms | 28.1% | ₹0.0311 |
| generate | 10 | 2587.5 ms | 4643.4 ms | 4643.4 ms | 22.8% | ₹0.0565 |
| fallback | 1 | 4254.0 ms | 4254.0 ms | 4254.0 ms | 37.6% | unknown |
| agent | 11 | 1306.2 ms | 3150.5 ms | 3150.5 ms | 11.5% | ₹0.0055 |
| router | 10 | 0.0 ms | 0.1 ms | 0.1 ms | 0.0% | unknown |

## By vector backend

| backend | runs | p50 end-to-end | p95 end-to-end |
|---|:--:|:--:|:--:|
| pgvector | 4 | 8020 ms | 17366 ms |
| unrecorded | 6 | 8726 ms | 22219 ms |

> Retrieval is a small fraction of end-to-end time — see `eval/BACKEND_COMPARISON.md` for the vector call measured in isolation.

## How to read this

- **p95, not mean.** A RAG request is a chain of network calls; the mean hides the tail a user actually feels.
- **`unknown` cost is not free cost.** A model with no price entry reports unknown. Treating that as zero is how cost attribution silently disappears when a model name changes.
- **The LLM dominates.** Retrieval is single-digit milliseconds (`eval/BACKEND_COMPARISON.md`); generation and verification are seconds. Optimising retrieval further buys almost nothing — the levers that matter are model tier, caching, and cutting the number of LLM calls per answer.
- **The agent multiplies cost.** Each planner step is a full LLM call. `eval/AGENT_BASELINE.md` shows its recall benefit vanishes once the retrieval budget is generous, so routing every query through the agent pays for coverage that is already there.
