# DocuChunk

A document question-answering system built to production shape: upload PDFs/DOCX →
parse → chunk → embed → **hybrid retrieval** (dense + BM25 + RRF, cross-encoder
rerank) → **agentic multi-hop planning** → **grounded generation** with citation
validation, abstention and a confidence breakdown — all of it **traced, cost-metered
and continuously evaluated**.

> Primary store is **Postgres + pgvector** — relational rows and embeddings in one
> database, one transaction boundary. Chroma (embedded) and Pinecone (managed) run
> behind the same adapter and are benchmarked against it on the same fixture. Scale
> today is a single application node with a real evaluation and observability story,
> not a distributed one.

## Measured results

Every number below is produced by a harness in this repo and committed as an
artifact — none of it is asserted. Negative and invalid results are published
unchanged.

| | result | artifact |
|---|---|---|
| Generation quality (real LLM generating **and** judging, 45 queries) | faithfulness **0.888** · answer-relevancy **1.000** · answer-correctness **0.978** | [`eval/GEN_BASELINE.md`](eval/GEN_BASELINE.md) |
| Retrieval, three modes head-to-head (42 docs, near-duplicate distractors) | MRR dense **0.777** → hybrid **0.883** → +rerank **0.967** | [`eval/phase8_comparison.md`](eval/phase8_comparison.md) |
| Agent vs single-shot RAG | **+0.431** context-recall at k=1 on multi-hop, decaying to 0 by k=5 | [`eval/AGENT_BASELINE.md`](eval/AGENT_BASELINE.md) |
| Vector backends — Chroma vs pgvector vs Pinecone (storage layer only varies) | identical quality (MRR **0.777**); p50 query **2.1 / 2.2 / 289.6 ms** | [`eval/BACKEND_COMPARISON.md`](eval/BACKEND_COMPARISON.md) |
| Cost & latency, from 10 recorded runs | e2e p50 **8.7 s** · p95 **22.2 s** · **₹0.072/query** (₹72.27 per 1k) | [`eval/COST_LATENCY.md`](eval/COST_LATENCY.md) |
| Synthesis: native vs LlamaIndex vs LangChain, identical evidence set | citation rate **80% / 0% / 40%** | [`eval/FRAMEWORK_COMPARISON.md`](eval/FRAMEWORK_COMPARISON.md) |
| Query expansion + MMR diversification | **no measurable gain** — fixture is saturated; both ship disabled | [`eval/RETRIEVAL_IMPROVEMENTS.md`](eval/RETRIEVAL_IMPROVEMENTS.md) |
| Per-stage latency budget | p50/p95/max per stage from real recorded runs | [`eval/LATENCY.md`](eval/LATENCY.md) |
| Load test (50/100/200 users) | **executed and INCONCLUSIVE** — host was swapping; published with the reason | [`eval/LOAD_TEST.md`](eval/LOAD_TEST.md) |
| RAGAS cross-check | **does not validate yet** — see the file for why | [`eval/RAGAS_CROSSCHECK.md`](eval/RAGAS_CROSSCHECK.md) |

Three findings worth reading the artifacts for:

- **The agent crossover.** Planned retrieval buys coverage, and coverage is only
  worth its extra LLM calls when the retrieval budget is scarce — at k=5 on this
  corpus a single pass already finds everything, so the agent costs precision for
  nothing. That crossover is what says *when* to route a query to the agent.
- **Backends differ operationally, not in quality.** Identical embeddings and metric
  → identical MRR; a gap would mean an adapter bug, not a better database. What does
  differ: Pinecone is **eventually consistent** and failed a read issued straight
  after ingestion, which pgvector and Chroma never do. Choose on ops burden and
  consistency, not on a quality table.
- **Frameworks lose the guarantee, not the fluency.** Given byte-identical evidence,
  LlamaIndex's default synthesizer emitted **zero** `[n]` citation markers. That is
  not a worse answer — it is one this system cannot verify, score or abstain on.

## What's inside

| Capability | Where |
|---|---|
| Hybrid retrieval (dense + BM25 + RRF) + cross-encoder rerank | `app/pipeline/retrieval.py`, `app/pipeline/bm25_index.py` |
| Pluggable vector backends — pgvector · Chroma · Pinecone, one adapter API | `app/pipeline/{pgvector_store,vector_store,pinecone_store}.py`, `app/core/vector_registry.py` |
| Per-document backend choice + search across a mixed corpus | `app/pipeline/multi_backend_store.py`, `app/core/dependencies.py` |
| Query expansion + MMR diversification (both off by default) | `app/pipeline/query_rewriter.py`, `app/pipeline/mmr.py` |
| Scope-aware routing — LOCAL (top-k) vs GLOBAL (whole document) + answer-task classes | `app/generation/query_classifier.py` |
| ReAct agent loop (hand-rolled; LangGraph-equivalent router/tool/state nodes, no framework lock-in) | `app/generation/agent/` |
| Grounded generation — citation parse + validate, groundedness verify, abstention, answer cache, token budgeting, model tiering | `app/generation/` |
| LLMOps — tracing, token + ₹ cost attribution, prompt versioning (Langfuse) | `app/observability/` |
| Framework interop — the retriever as a LlamaIndex / LangChain `BaseRetriever` | `app/integrations/` |
| Evaluation — retrieval, generation, agent-vs-RAG, failure taxonomy | `app/eval/` |
| Benchmarks — backends, cost/latency, frameworks, retrieval variants, load | `scripts/` |
| CI regression gate (hermetic — no API key needed) | `.github/workflows/eval.yml` |
| Pipeline dashboard — record-then-replay trace artifacts | `app/routers/pipeline.py`, `/pipeline` |

Evaluation runs in two profiles: a deterministic **surrogate** (no API, no torch)
that CI gates on, and a **neural** profile driven by a real model for the definitive
numbers. The failure taxonomy assigns each query exactly one class that names the
fix, so a regression points at the thing to change.

**Framework integrations are deliberately optional.** They live in
`requirements-integrations.txt` with lazy imports, so a missing package never breaks
the app and the CI gate stays hermetic — ragas 0.4.3 already pinned
`langchain-community<0.4` in this repo, and that conflict is not allowed near the
serving path.

## Run

```bash
uvicorn app.main:app --reload --port 8001      # web (serves API + HTML pages)
arq app.worker.WorkerSettings                  # ingestion worker (parse→chunk→embed)
```

If Redis is unavailable the web process ingests uploads inline (no separate worker
needed).

### Docker

```bash
docker compose up                              # app + worker + postgres/pgvector + redis
docker compose --profile chroma up             # add embedded Chroma (VECTOR_BACKEND=chroma)
docker compose --profile offline up            # add Ollama for the offline model tier
docker compose --profile observability up      # add self-hosted Langfuse (ClickHouse, MinIO, Redis)
```

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for the deployed topology.

## Configuration

Everything lives in `.env` (see `.env.example`, which documents each key inline).
The keys that change behaviour most:

```bash
# Generation
GEN_PROVIDER=gemini                 # stub | gemini | openai_compat
GEN_MODEL=gemini-3.1-flash-lite     # verified working; gemini-2.0-flash is quota-capped (429) on some keys
GEMINI_API_KEY=...                  # never commit a real key
VERIFY_PROVIDER=                    # empty reuses GEN_PROVIDER; set it to split verification onto a cheaper tier
CACHE_BACKEND=memory                # none | memory | redis — version-aware answer cache

# Storage
DATABASE_URL=postgresql+psycopg2://docuchunk:docuchunk@postgres:5432/docuchunk
VECTOR_BACKEND=pgvector             # pgvector | chroma | pinecone
PINECONE_API_KEY=                   # only for VECTOR_BACKEND=pinecone

# Retrieval enhancements — measured as pure cost on this corpus, so both ship off
QUERY_REWRITE_ENABLED=false
MMR_ENABLED=false

# Agent + streaming
AGENT_ENABLED=false                 # route always exists; safety is max-steps + loop detection + fallback-to-RAG
AGENT_MAX_STEPS=4
STREAM_ENABLED=true

# Observability (fail-open: disabled or misconfigured → NullTracer, app unchanged)
TRACING_ENABLED=false
TRACE_CAPTURE=metadata              # none | metadata | full (full = prompt/completion text, DEV ONLY)
LANGFUSE_HOST=https://cloud.langfuse.com
```

> **Model note:** if `/generate/answer` starts returning `error: "generation_failed"`
> with an underlying 429, the configured model has no quota on your key. Switch
> `GEN_MODEL` to another model your key can call (e.g. `gemini-3.1-flash-lite`).

## Key endpoints

| Endpoint | Purpose |
| --- | --- |
| `POST /generate/answer?stream=true` | SSE: `token` events → one `verification` event (answer, `cited_sources`, `dropped_sources`, `confidence`, `confidence_signals`, `abstained`, `verified`) |
| `POST /generate/answer?stream=false` | Same payload as a single JSON body |
| `GET  /generate/models` | Model tiers this deployment can actually serve (probes Ollama for the offline tier) |
| `POST /generate/agent` | ReAct multi-hop answer — plan, retrieve, repeat, then ground; falls back to RAG |
| `POST /search/semantic` | Ranked chunks, no LLM — spans every backend holding the user's documents |
| `POST /docs/upload` · `GET /docs/list` · `GET /docs/{id}` · `DELETE /docs/{id}` | Document lifecycle (upload picks the vector backend per document) |
| `GET  /docs/backends` | Vector backends this deployment can use, for the upload UI |
| `GET  /docs/{id}/raw` | Raw file inline (auth header **or** `?token=`) — powers the chat doc-preview panel |
| `GET  /analysis/{id}/chunks` · `/embeddings` · `/visualise` | Chunk, embedding and projection views behind the dashboard |
| `POST /pipeline/runs` · `GET /pipeline/runs` · `GET /pipeline/runs/{id}` | Record-then-replay pipeline trace artifacts (per-stage timing, tokens, ₹ cost) |
| `POST /api/chat/sessions` | Start a chat session (locks the user's prior active session) |
| `GET  /api/chat/sessions` · `GET /api/chat/sessions/{id}` | List / load session history |
| `POST /api/chat/sessions/{id}/messages` | Append a turn (409 if the session is locked) |
| `POST /api/chat/sessions/{id}/lock` · `DELETE /api/chat/sessions/{id}` | Lock / delete a session |
| `POST /auth/register` · `/login` · `/refresh` · `GET /auth/me` | JWT auth, plus Google and GitHub OAuth callbacks |
| `GET  /health` | DB ping + vector-store heartbeat |

### `confidence_signals`

The trust-metric breakdown returned alongside `confidence`:

- **retrieval** — relevance of the retrieved evidence (reranker top score, 0–1)
- **citation_coverage** — fraction of the model's `[n]` markers that survived validation
- **groundedness** — the verifier's confidence that claims are supported
- **verified** — whether the groundedness check actually ran (fail-closed → `false`)
- **capped** — `true` when groundedness outran the weaker retrieval/coverage evidence

## Frontend

- **Dashboard** (`/dashboard`) — documents, search, analysis. The profile is an
  in-page **popup** (avatar picker + email/password edit), opened from the sidebar
  user card. "Chat with Document" is in the sidebar.
- **Chat** (`/chat`, `/chat/{doc_id}`) — three panels: session sidebar · chat ·
  live document preview. Citations render as clickable `[n]` badges (dropped ones
  flagged), each message shows a confidence bar + expandable signals breakdown.
  Per-message model picker (cloud Gemini vs a local Ollama tier), plus **voice
  in/out** — browser speech recognition for input and speech synthesis for answers.
  Sessions are per-user and persist; leaving the chat locks the session and the next
  "New Chat" starts a fresh one.
- **Pipeline** (`/pipeline`) — replay a recorded run stage by stage with its timings,
  token usage and cost.

## Verify it yourself

```bash
# Full unit + integration suite (37 test modules, hermetic)
pytest -q

# Live end-to-end against a real Gemini key: auth → upload → ingest → grounded
# answer (cited, confident) → abstain on an unanswerable question → cache hit.
# Never prints the key. Exit 0 = all steps passed.
python scripts/smoke_test.py                                  # defaults to http://127.0.0.1:8001
BASE_URL=http://127.0.0.1:8000 python scripts/smoke_test.py   # override the target
```

Regenerate any published artifact:

```bash
python -m app.eval.harness --k 5                # retrieval metrics
python -m app.eval.gen_harness --provider offline --gate   # what CI enforces
python -m app.eval.agent_harness                # agent vs RAG
python scripts/backend_benchmark.py             # eval/BACKEND_COMPARISON.md
python scripts/cost_latency_report.py           # eval/COST_LATENCY.md
python scripts/latency_report.py                # eval/LATENCY.md
python scripts/framework_benchmark.py           # eval/FRAMEWORK_COMPARISON.md
python scripts/retrieval_improvements.py        # eval/RETRIEVAL_IMPROVEMENTS.md
locust -f scripts/locustfile.py                 # eval/LOAD_TEST.md
```

## Docs

| Document | What it covers |
|---|---|
| [`docs/TECHNICAL_DOSSIER.md`](docs/TECHNICAL_DOSSIER.md) | Full system walkthrough — every design decision and its evidence |
| [`docs/CAPABILITIES.md`](docs/CAPABILITIES.md) | Every capability mapped to the code and the artifact that proves it |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | Postgres + pgvector topology, compose profiles, production config |
| [`docs/OBSERVABILITY.md`](docs/OBSERVABILITY.md) | Trace schema, span types, cost attribution, self-hosted Langfuse |
| [`docs/GAP_CLOSURE_PLAN.md`](docs/GAP_CLOSURE_PLAN.md) | Known gaps, constraints, and what closing each one requires |

## Known limits

Stated here rather than discovered later:

- **Throughput is unmeasured.** One Uvicorn worker, and query embedding is
  synchronous CPU work inside an async request path — both predict saturation at low
  concurrency. The load test that would confirm it was invalidated by host resource
  exhaustion; see [`eval/LOAD_TEST.md`](eval/LOAD_TEST.md) for what makes it valid.
- **The LLM dominates the budget.** Retrieval is single-digit milliseconds against
  seconds of generation and verification, so further retrieval optimisation buys
  almost nothing — model tier, caching and fewer LLM calls per answer are the levers.
- **The eval fixture is saturating.** Three of four query categories already score at
  the ceiling, which is why the expansion/MMR comparison could not resolve. A larger
  corpus with harder distractors is the prerequisite for the next retrieval result.
- **RAGAS does not validate yet** — [`eval/RAGAS_CROSSCHECK.md`](eval/RAGAS_CROSSCHECK.md)
  records the run and the reason rather than dropping the check.
