# DocuChunk

Production-grade document retrieval & RAG platform: upload PDFs/DOCX → parse →
chunk → embed → hybrid retrieval → grounded generation with citation validation
and a confidence breakdown.

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
