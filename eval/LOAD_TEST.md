# Load test — executed, and INCONCLUSIVE

Run with `scripts/locustfile.py` at 50 / 100 / 200 concurrent users, 60s each,
against the local stack.

**These numbers do not characterise the application, and must not be quoted as if
they do.** Publishing them anyway, with the reason, because a load test that was run
and discarded is more honest than one that was never run.

## What was measured

| users | requests completed (60s) | error rate | p95 |
|---:|---:|---:|---:|
| 50 | 3 | 0% | 120 ms |
| 100 | 25 | 96% | 56,000 ms |
| 200 | 1 | 100% | 57,849 ms |

## Why it is invalid

**`GET /health` took 56.8 seconds at 100 users.** That endpoint does a database ping
and a vector-store heartbeat — single-digit milliseconds of work, no embedding, no
LLM, no reranking. An endpoint that trivial cannot take 57 seconds because of
anything in the retrieval path. The whole process was starved.

The host during the run:

```
Mem:  11Gi total, 1.2Gi free
Swap: 2.0Gi total, 1.6Gi used
```

The machine was paging. A load test in that state measures the swap device, not the
service. At 50 users only 3 requests completed and **not one of them was
`/search/semantic`** — Locust never got through ramp-up, so the hot path was never
exercised at all.

This was a predicted failure. `docs/GAP_CLOSURE_PLAN.md` §Constraints says
"benchmarking in that state measures swap, not the system." The test was run before
that constraint was resolved.

## What is nonetheless true

Two architectural limits, verified from configuration rather than from this run:

1. **The app serves from a single Uvicorn worker.** The container command is
   `uvicorn app.main:app --host 0.0.0.0 --port 8000` with no `--workers`. One
   process handles every request, so concurrency is bounded by one event loop.

2. **Query embedding is synchronous CPU work inside an async request path.**
   `embed_query` runs a sentence-transformers forward pass. Synchronous CPU work in
   an async endpoint blocks the event loop for its duration, so concurrent requests
   serialise behind it rather than interleaving.

Together these predict that throughput saturates at low concurrency and that the
first thing to saturate is CPU on the embedding call, not Postgres or the vector
index. **This run did not confirm that** — it only confirms the host had no capacity
to answer the question.

## What would make it valid

1. Run on the deployed VM (Oracle Always-Free ARM, 4 OCPU / 24 GB) rather than a
   laptop that is already swapping.
2. Serve with `--workers 4` so the measurement reflects a realistic deployment
   instead of a single event loop.
3. Keep hammering **retrieval only**. The Gemini free tier is 500 requests/day; a
   200-user run against the generation path would exhaust the daily quota in seconds
   and measure Google's rate limiter. LLM cost is already derived analytically from
   recorded per-call token usage in `eval/COST_LATENCY.md`.
4. Watch `docker stats` during the run to identify which container saturates first,
   turning the predicted bottleneck above into a measured one.

Until then, the honest claim is: **"load-test harness built with pass/fail gating;
first execution was invalidated by host resource exhaustion and is pending a run on
the deployed instance."** Not "the system handles N users."
