# DocuChunk

A document question-answering system built to production shape: upload PDFs/DOCX →
parse → chunk → embed → **hybrid retrieval** (dense + BM25 + RRF, cross-encoder
rerank) → **agentic multi-hop planning** → **grounded generation** with citation
validation, abstention and a confidence breakdown — all of it **traced, cost-metered
and continuously evaluated**.

> Runs on SQLite + a single-node Chroma by default, which is honest about its scale
> today: this is a single-node system with a real evaluation and observability story,
> not a distributed one. Postgres/pgvector is the documented next step.

## Measured results

Every number below is produced by a harness in this repo and committed as an
artifact — none of it is asserted.

| | result | artifact |
|---|---|---|
| Generation quality (real LLM generating **and** judging, 45 queries) | faithfulness **0.888** · answer-relevancy **1.000** · answer-correctness **0.978** | [`eval/GEN_BASELINE.md`](eval/GEN_BASELINE.md) |
| Retrieval, three modes head-to-head (42 docs, near-duplicate distractors) | MRR dense **0.777** → hybrid **0.883** → +rerank **0.967** | [`eval/phase8_comparison.md`](eval/phase8_comparison.md) |
| Agent vs single-shot RAG | **+0.431** context-recall at k=1 on multi-hop, decaying to 0 by k=5 | [`eval/AGENT_BASELINE.md`](eval/AGENT_BASELINE.md) |
| Latency | per-stage p50/p95 from real recorded runs | [`eval/LATENCY.md`](eval/LATENCY.md) |
| RAGAS cross-check | **does not validate yet** — see the file for why | [`eval/RAGAS_CROSSCHECK.md`](eval/RAGAS_CROSSCHECK.md) |

**The agent result is the interesting one.** Planned retrieval buys coverage, and
coverage is only worth its extra LLM calls when the retrieval budget is scarce — at
k=5 on this corpus a single pass already finds everything, so the agent costs
precision for nothing. The crossover is the finding: it says *when* to route a query
to the agent instead of RAG.

## What's inside

| Capability | Where |
|---|---|
| Hybrid retrieval + reranking | `app/pipeline/retrieval.py`, `app/pipeline/bm25_index.py` |
| ReAct agent loop (hand-rolled; LangGraph-equivalent router/tool/state nodes, no framework lock-in) | `app/generation/agent/` |
| LLMOps — tracing, token + ₹ cost attribution, prompt versioning (Langfuse) | `app/observability/` |
| Evaluation — retrieval, generation, agent-vs-RAG, failure taxonomy | `app/eval/` |
| CI regression gate (hermetic — no API key needed) | `.github/workflows/eval.yml` |
| Pipeline dashboard — record-then-replay trace artifacts | `app/routers/pipeline.py`, `/pipeline` |

Evaluation runs in two profiles: a deterministic **surrogate** (no API, no torch)
that CI gates on, and a **neural** profile driven by a real model for the definitive
numbers. The failure taxonomy assigns each query exactly one class that names the
fix, so a regression points at the thing to change.

## Configuration (generation)

Set these in `.env` (see `.env.example`):

```
GEN_PROVIDER=gemini
GEN_MODEL=gemini-3.1-flash-lite     # verified working; gemini-2.0-flash is quota-capped (429) on some keys
GEMINI_API_KEY=...                  # never commit a real key
CACHE_BACKEND=memory                # memory | redis | none (enables the answer cache)
```

> **Model note:** if `/generate/answer` starts returning `error: "generation_failed"`
> with an underlying 429, the configured model has no quota on your key. Switch
> `GEN_MODEL` to another model your key can call (e.g. `gemini-3.1-flash-lite`).

## Run

```bash
uvicorn app.main:app --reload --port 8001      # web (serves API + HTML pages)
arq app.worker.WorkerSettings                  # ingestion worker (parse→chunk→embed)
```

If Redis is unavailable the web process ingests uploads inline (no separate worker needed).

## Gemini smoke test (real end-to-end)

Verifies the live stack against the real Gemini key: auth → upload → ingest →
grounded answer (cited, confident) → abstain on an unanswerable question → cache hit.
It never prints the key.

```bash
# requires GEN_PROVIDER=gemini + a valid GEMINI_API_KEY in .env, and the backend running
python scripts/smoke_test.py                    # defaults to http://127.0.0.1:8001
BASE_URL=http://127.0.0.1:8000 python scripts/smoke_test.py   # override the target
```

Exit `0` = all steps passed, `1` = any failure (the full response JSON is printed on failure).

## Key endpoints

| Endpoint | Purpose |
| --- | --- |
| `POST /generate/answer?stream=true` | SSE: `token` events → one `verification` event (answer, `cited_sources`, `dropped_sources`, `confidence`, `confidence_signals`, `abstained`, `verified`) |
| `POST /generate/answer?stream=false` | Same payload as a single JSON body |
| `POST /search/semantic` | Ranked chunks, no LLM |
| `GET  /docs/{id}/raw` | Raw file inline (auth header **or** `?token=`) — powers the chat doc-preview panel |
| `POST /api/chat/sessions` | Start a chat session (locks the user's prior active session) |
| `GET  /api/chat/sessions` · `GET /api/chat/sessions/{id}` | List / load session history |
| `POST /api/chat/sessions/{id}/messages` | Append a turn (409 if the session is locked) |
| `POST /api/chat/sessions/{id}/lock` | Lock a session (the chat UI calls this on leave) |

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
  Sessions are per-user and persist; leaving the chat locks the session and the
  next "New Chat" starts a fresh one.
