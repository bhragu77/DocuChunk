# Latency budget — measured

Computed from **11 recorded pipeline runs** — real executions, not a synthetic benchmark. Every run stores a per-stage `duration_ms` in its trace artifact; this aggregates them.

- **End-to-end p50:** 15219 ms
- **End-to-end p95:** 98173 ms

| stage | n | p50 (ms) | p95 (ms) | max (ms) | share of p50 |
|---|:--:|:--:|:--:|:--:|:--:|
| generate | 11 | 989.7 | 43621.1 | 43621.1 | 28.6% |
| agent | 31 | 1174.4 | 29322.2 | 42409.0 | 34.0% |
| fallback | 6 | 19.7 | 13911.8 | 13911.8 | 0.6% |
| verify | 5 | 1274.7 | 4101.5 | 4101.5 | 36.9% |
| router | 11 | 0.0 | 0.6 | 0.6 | 0.0% |

### How to read this

- **p95, not mean.** A RAG request is a chain of network calls; the mean hides the tail a user actually notices.
- **A stage whose p95 is many times its p50 is the latency budget's real cost centre** — that is where caching or a smaller model pays off, and it is usually an LLM call rather than retrieval.
- Regenerate with `python scripts/latency_report.py` after any change that could affect timing.
