# Observability (LLMOps tracing)

DocuChunk traces every RAG query and every ingestion job through a small,
vendor-neutral **seam** (`app/observability/`). Phase 9A defined the seam and a
no-op tracer; Phase 9B wires **Langfuse** in behind it. Instrumentation call sites
never import a tracing SDK and never changed between the phases.

- **Off by default.** `TRACING_ENABLED` unset → `NullTracer`; the app behaves
  byte-identically and pulls in no tracing dependency.
- **Fail-open, always.** A missing SDK, absent keys, a dead Langfuse, or any SDK
  call raising can never change a request's result, status code, or (materially)
  its latency.

---

## 1. Span trees

### Query path — root `rag.request`

```
rag.request                      (trace; user_id = hashed, doc_id, top_k, scope, …)
├── retrieve                     k, candidate_count, returned_chunk_ids
│   ├── embed_query
│   ├── vector_search            n_results, returned
│   ├── bm25_search              returned
│   └── rrf_fuse                 rrf_k, fused
├── rerank                       model, candidates_in, kept_out
├── prompt_build                 source_count
├── generate  (generation)       model, input_tokens, output_tokens, cost, prompt_version, ttft_ms*
└── verify                       scores: groundedness, citation_validity, confidence
                                 (root also carries: abstained)
```
`*ttft_ms` is recorded on the streaming path only.

### Ingestion path — root `ingest.document`

```
ingest.document                  (trace; doc_id, status)   ← trace_id carried from the API
├── parse                        pages
├── clean                        cleaned_pages
├── chunk                        strategy, chunks
├── embed                        chunk_count, batch_count
└── store
```

Cross-process propagation: the API mints a `trace_id`, puts it in the arq job
kwargs alongside `doc_id` (primitive-only payload — idempotency/checkpoints
untouched), and the worker starts `ingest.document` under it.

---

## 2. Fields and scores recorded

`generate` is emitted as a Langfuse **generation** (so cost/usage/model are
first-class); every other span is a Langfuse **span**. Text (query/prompt) is
recorded only at capture level `full`.

| Span | Field | Unit | Expected range |
|------|-------|------|----------------|
| rag.request (root) | `user_id` | hashed id | 16-char hex, never raw |
| | `doc_id` | id | string or null (multi-doc) |
| | `top_k` | count | 1–50 |
| | `retrieval_scope` | enum | `local` / `global` |
| | `answer_task` | enum | answer/enumerate/table/… |
| | `abstained` | bool → score 0/1 | 0 or 1 |
| retrieve | `k` | count | 1–50 |
| | `candidate_count` | count | 0–50 |
| | `returned_chunk_ids` | ids | ≤200 ids, **ids only, never text** |
| vector_search | `n_results` / `returned` | count | 0–100 |
| bm25_search | `returned` | count | 0–50 |
| rrf_fuse | `rrf_k` / `fused` | const / count | 60 / 0–50 |
| rerank | `model` | name | cross-encoder model id |
| | `candidates_in` / `kept_out` | count | in ≥ out; out ≤ top_k |
| prompt_build | `source_count` | count | 0–top_k (or whole-doc) |
| generate | `model` | name | provider model id |
| | `input_tokens` / `output_tokens` | tokens | **from provider usage, never estimated** |
| | `cost_usd` | USD | ≥ 0 (stored) |
| | `cost_inr` | INR | ≥ 0 (display; `TRACE_USD_TO_INR`) |
| | `prompt_version` | id | e.g. `grounded-qa/v3+task:answer` |
| | `ttft_ms` | ms | ≥ 0 (streaming only) |
| verify (scores) | `groundedness` | score | 0.0–1.0 |
| | `citation_validity` | score | 0.0–1.0 (cited∧supported ratio) |
| | `confidence` | score | 0.0–1.0 |
| embed | `chunk_count` / `batch_count` | count | ≥ 0 |

**Scores** (`groundedness`, `citation_validity`, `confidence`) are attached to the
`verify` observation **and** mirrored to the trace level; `abstained` (a boolean
field on the root) is emitted as a trace-level score `abstained` ∈ {0,1}. Trace-level
scores are what Langfuse charts over time.

### Cost

`app/observability/pricing.py` holds a per-model USD price table (input/output per
1M tokens), overridable via `TRACE_MODEL_PRICES` (JSON). Cost is computed from the
**provider-returned** token usage — an unpriced model or missing usage records **no
cost** rather than a fabricated one. USD is stored; INR is a display conversion at
`TRACE_USD_TO_INR` (default 83). Aggregation "per hashed user per day" is a Langfuse
group-by on the trace `user_id` (hashed) and observation cost.

---

## 3. Fail-open guarantee & how it is tested

The seam guarantees:

1. Every call into a tracer/span is individually wrapped in `try/except` → DEBUG log.
2. `span()` always yields a usable span (a `NullSpan` proxy on failure) — callers
   never null-check.
3. Business exceptions are recorded on the span and **re-raised unchanged**; tracing
   never swallows or alters control flow.
4. On the request path, span calls only **buffer** in memory. The single SDK write
   per span and all HTTP flushing happen on Langfuse's background thread. `flush()`
   (the only blocking call) runs at shutdown and at the end of each worker job.

Tests (`tests/observability/`):

- `test_broken_tracer_does_not_change_endpoint` / `test_langfuse_broken_client_is_fail_open`
  — a tracer/SDK-client whose every method raises produces a **byte-identical**
  endpoint response and status.
- `test_build_tracer_null_without_keys` / `test_build_tracer_null_on_init_failure`
  — absent keys or a constructor that throws → `NullTracer`, never a startup failure.
- **KILL TEST** `test_kill_switch_20_queries_succeed_and_latency_bounded`.

### KILL TEST results

`TRACING_ENABLED=true`, real `LangfuseTracer`, 40 warm queries each against a
**reachable** mock Langfuse (traced baseline) vs an **unreachable** host (killed):

| | p50 | p95 |
|---|-----|-----|
| Traced (Langfuse reachable) | 25.4 ms | 35.34 ms |
| Killed (Langfuse unreachable) | 12.4 ms | 35.40 ms |

**All 40 killed-path queries returned 200. Killed p95 is +0.2% vs the traced
baseline — within the 5% budget.** The request path is identical whether Langfuse
is up or down (buffer-only); connection-refused fails fast on the background thread,
so tearing Langfuse down is invisible to callers. (Against real LLM latency of
hundreds of ms–seconds, the absolute buffering overhead is a fraction of a percent.)

Two distinct failure modes are covered:

- **Connection refused** ("door locked" — fast fail): the OS rejects the connect
  immediately. `test_kill_switch_20_queries_succeed_and_latency_bounded`.
- **Blackhole** ("door opens, nobody answers" — slow death): the socket is accepted
  but never replies, so the background flusher blocks until `timeout=5` gives up.
  `test_kill_switch_blackhole_host_does_not_hang_requests` runs 20 queries against a
  socket that accepts and never responds; all 20 return promptly (the flush hang is
  on the background thread, not the request path — this is exactly what `timeout=5`
  bounds). Connection-refused alone would NOT exercise this.

---

## 4. Capture-level policy & retention

`TRACE_CAPTURE`:

- `none` — record nothing (spans exist structurally but carry no fields/scores).
- `metadata` (**default**) — ids, counts, token usage, latencies, scores. **No
  document text, no prompt text, no completion text.**
- `full` — additionally records prompt/completion text. **Dev only.**

User ids are **always** hashed via `hash_user_id` (salted SHA-256, 16-char prefix;
salt from `TRACE_USER_SALT`, `TRACE_SALT` accepted as an alias) — raw ids are never
sent. `returned_chunk_ids` are ids only, never chunk text.

**Retention** is enforced at the Langfuse project level (data-retention setting on
Cloud, or your self-host policy), not by the app. Recommended: keep `metadata`
traces 30–90 days for trend analysis; keep `full`-capture traces (if ever enabled)
far shorter, as they contain source/answer text.

---

## 5. Why prompts stay in git

Prompt **templates live in `app/generation/prompt_builder.py`** and git is the
single source of truth for their bodies. We record only a **version identifier**
(`GROUNDED_PROMPT_VERSION`, e.g. `grounded-qa/v3+task:answer`) on each answer's
generate span. We deliberately **do not fetch prompts from Langfuse at request
time**:

- no network dependency on the request path (a prompt fetch would be exactly the
  blocking call fail-open exists to avoid);
- prompt changes go through code review, CI, and rollback like any other code;
- an answer can still be tied to the exact template that produced it, because the
  version travels to the trace. Bump the version constant when you edit a template.

---

## 6. Self-host stack (opt-in)

`docker compose --profile observability up -d` starts a full **Langfuse v3**
deployment. Default `docker compose up` (no profile) starts **none** of it.

> Budget roughly **4 CPU / 8 GB RAM**.

| Container | Stores / does |
|-----------|---------------|
| `langfuse-web` | UI + ingestion API (port 3000). Point `LANGFUSE_HOST` here. |
| `langfuse-worker` | Async ingestion: drains the queue → ClickHouse. |
| `langfuse-db` (Postgres) | Transactional data: projects, users, API keys, prompt/score **config**. |
| `langfuse-clickhouse` | **Traces, observations, and scores** — the analytical store. |
| `langfuse-minio` (S3) | Blob staging for large ingestion event batches / media. |
| `langfuse-redis` | Langfuse's **own** Redis — the ingestion queue. |

**Redis isolation:** Langfuse uses a dedicated `langfuse-redis` instance, never the
app's `redis` (arq + answer cache, DB 0). Trace-ingestion load can therefore never
evict answer-cache keys or arq jobs.

**Why ClickHouse:** the acceptance views — a per-span **latency waterfall**, **cost
per hashed user per day**, and **score trends** (groundedness/abstention over time)
— are `GROUP BY`/time-bucket aggregations over a high-volume, append-only event
stream. A row store (Postgres) can't serve those at trace volume; ClickHouse is a
columnar engine built exactly for them. Postgres keeps the small, transactional
config; ClickHouse keeps the big, analytical event data.

---

## 7. Viewing the acceptance dashboards

With a real Langfuse (`LANGFUSE_HOST` + keys, `TRACING_ENABLED=true`):

- **(a) Trace waterfall** — Traces → open a `rag.request` trace: the
  retrieve→(embed/vector/bm25/rrf), rerank, prompt_build, generate, verify tree with
  per-span latency, plus the generation's token counts and cost. In-repo, the exact
  emitted structure is asserted by `test_langfuse_query_trace_waterfall_cost_and_scores`.
- **(b) Cost** — the generation observation carries `cost_details.total` (USD) and
  `input`/`output` token usage; chart **Cost by `user_id` per day** (user_id is the
  hashed value). The pricing math is unit-tested in `test_pricing_from_config_not_hardcoded`.
- **(c) Scores over time** — Scores dashboard: `groundedness`, `confidence`,
  `citation_validity`, and `abstained` are trace-level scores, chartable as
  time series.
