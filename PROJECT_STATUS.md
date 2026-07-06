# DocuChunk — Project Status & Build Summary

> Written: 2026-06-24

---

## What Is This?

DocuChunk is a production-grade **Retrieval-Augmented Generation (RAG) platform** built from scratch as a learning project. Users upload PDF and DOCX documents; the system parses, cleans, chunks, embeds, and stores them so they can be semantically searched and used to answer questions via an LLM.

The primary goal is to understand every layer deeply — from raw PDF text extraction, to custom chunking logic, to embedding vectors, to ChromaDB storage — while shipping something production-ready.

---

## Current State: What Is DONE

### Phase 0 — Bootstrap
- FastAPI app factory with lifespan events (DB init + ChromaDB init on startup)
- SQLAlchemy engine, session, and `Base` wired up
- `GET /health` endpoint
- All required directories auto-created on startup (`uploads/`, `chroma_db/`, `logs/`)
- Pydantic Settings (`config.py`) reading from `.env` — 12-factor compliant
- CORS + SessionMiddleware configured

### Phase 1a — JWT Authentication (COMPLETE)
- `POST /auth/register` — bcrypt-hashed password (SHA-256 pre-hash for >72 char passwords), stores user, returns access + refresh JWT
- `POST /auth/login` — verifies password, returns token pair
- `POST /auth/refresh` — rotates refresh token
- `GET /auth/me` — returns current user profile
- `get_current_user` FastAPI dependency — injects authenticated user into any protected route
- Access token: 30 min. Refresh token: 7 days.

### Phase 1b — OAuth (COMPLETE)
- `GET /auth/oauth/google` + callback — full Google OAuth 2.0 flow
- `GET /auth/oauth/github` + callback — full GitHub OAuth flow (handles hidden emails by fetching `/user/emails`)
- After OAuth: upserts user in DB, issues our own JWT (stops using the OAuth provider's token)
- Redirects to `/welcome` page with animated welcome screen after login

### Phase 1c — Password Reset (COMPLETE)
- `POST /auth/forgot-password` — generates a secure random token, stores only its SHA-256 hash, returns reset URL in dev mode
- `POST /auth/reset-password` — validates token hash, updates password, marks token as used
- Tokens expire in 1 hour and are single-use
- Secure: always returns 200 even for unknown emails (prevents user enumeration)

### Phase 1d — Auth Event Logging (COMPLETE)
- `auth_events` table — append-only log of every register/login/oauth/token-refresh event
- Stores: event type, user ID, provider, success flag, IP address, user-agent
- Used for breach detection and audit trail

### Phase 2 — Document Upload & Storage (COMPLETE)
- `POST /docs/upload` — validates extension (PDF/DOCX only) + MIME type, enforces 20 MB limit, stores file to `uploads/{user_id}/{uuid}.ext`, creates DB record, fires background pipeline
- `GET /docs/list` — paginated document list, filterable by status
- `GET /docs/{doc_id}` — single document detail
- `DELETE /docs/{doc_id}` — deletes file on disk, DB record, and ChromaDB embeddings
- Status field tracks pipeline progress: `uploaded → parsing → chunking → embedding → ready | failed`
- Each user gets their own upload subdirectory and ChromaDB collection

### Phase 3 — Parsing Pipeline (COMPLETE)
- `pipeline/parser.py`:
  - **PDF**: PyMuPDF page-by-page extraction, preserves page numbers, skips blank/image-only pages
  - **DOCX**: python-docx paragraph + table extraction, groups into 40-paragraph logical "pages", tables extracted separately as pipe-delimited rows
- `pipeline/cleaner.py` — 8-step cleaning:
  1. Unicode NFKC normalisation (ligatures, fancy quotes)
  2. Null byte / control character removal
  3. Page number noise stripping (`5`, `Page 3 of 45`, `- 3 -`)
  4. Hyphenated line-break fixing (`compre-\nhension` → `comprehension`)
  5. Multi-space collapse
  6. Trailing space stripping per line 
  7. 3+ consecutive newlines → single blank line
  8. Final strip
- `pipeline/types.py` — internal dataclasses: `ParsedPage`, `CleanedPage`, `Chunk`, `EmbeddedChunk`
- `pipeline/orchestrator.py` — background task runner; runs stages in sequence, updates document status at each stage, handles errors gracefully

### Frontend Pages (COMPLETE)
- `/` — Landing page (hero, feature list, CTAs)
- `/login` — Email/password login + Google + GitHub OAuth buttons
- `/register` — Registration form
- `/forgot-password` — Request reset link
- `/reset-password` — Enter new password via token link
- `/welcome` — Animated welcome screen after OAuth login
- `/dashboard` — Main app dashboard (document upload, list, status tracking)

### Tests (COMPLETE for Phases 0–3)
- `tests/test_health.py` — health endpoint
- `tests/test_auth.py` — register, login, token validation, duplicate email, wrong password
- `tests/test_documents.py` — upload, list, delete, auth enforcement, file type validation
- `tests/test_pipeline.py` — 18 tests covering parser (PDF + DOCX) and cleaner, plus a full integration test (upload → pipeline → `ready` status)

### Deployment Setup (COMPLETE)
- `Dockerfile` — Python 3.11-slim, layered for build cache efficiency
- `docker-compose.yml` — named volumes for uploads, chroma_db, sqlite data
- `railway.json` — Railway deployment config
- `Procfile` — for Render/Heroku
- `netlify.toml` — for landing page static hosting

---

## What Is NOT Built Yet

These phases are stubbed in `orchestrator.py` with comments but contain no real code:

| Phase | What | Status |
|-------|------|--------|
| Phase 4 | **Custom Chunker** (`pipeline/chunker.py`) — fixed, sentence-aware, paragraph strategies | Not started |
| Phase 5 | **Chunk Analyser** (`pipeline/analyser.py`) — token stats, distribution, overlap coverage | Not started |
| Phase 6 | **Embedding Generation** (`pipeline/embedder.py`) — HuggingFace API calls, batching, retry | Not started |
| Phase 7 | **ChromaDB Storage** (`pipeline/vector_store.py`) — upsert with metadata, query, delete | Not started |
| Phase 8 | **Search & RAG** (`/search/semantic`, `/generate/answer`) | Router not created |
| Phase 9 | **Debug / Visualisation** (`/debug/visualise`, `/debug/inspect`) — UMAP 2D plots | Not created |

The router stubs for chunks, embeddings, search, generate, and debug are referenced in `main.py` as comments ready to be uncommented once built.

---

## Architecture

```
Browser
  │
  ▼
FastAPI (app/main.py)
  ├── /auth        ← JWT + Google OAuth + GitHub OAuth + password reset
  ├── /docs        ← Upload, list, delete  [BackgroundTasks → pipeline]
  ├── /health      ← Liveness check
  ├── HTML pages   ← Jinja2 templates (landing, login, register, dashboard…)
  │
  ├── SQLite / PostgreSQL (SQLAlchemy)
  │     └── Tables: users, documents, auth_events, password_reset_tokens, embedding_jobs
  │
  └── ChromaDB (persistent, per-user collections)

Background Pipeline (run_pipeline)
  Stage 1: parse()       ← PyMuPDF / python-docx
  Stage 2: clean_pages() ← normalise text
  Stage 3: [STUB] chunk
  Stage 4: [STUB] analyse
  Stage 5: [STUB] embed
  Stage 6: [STUB] vector store
```

---

## Key Design Decisions Made

| Decision | Choice | Reason |
|----------|--------|--------|
| Task queue | FastAPI `BackgroundTasks` | Simple for MVP; Celery + Redis upgrade path documented |
| Embeddings | HuggingFace Inference API (free) | `sentence-transformers/all-MiniLM-L6-v2`, 384-dim |
| Vector DB | ChromaDB (local persistent) | Easy to learn internals; HNSW index |
| Relational DB | SQLite (dev) → PostgreSQL (prod) | SQLAlchemy makes migration trivial |
| Auth | JWT (python-jose) + OAuth (Authlib) | Industry standard; bcrypt with SHA-256 pre-hash |
| Chunking | Hand-written (three strategies planned) | Core learning objective |
| Frontend | Vanilla HTML/Jinja2 | No build step, easy to follow |

---

## Next Steps (Recommended Order)

**Step 1 — Build the Chunker (`app/pipeline/chunker.py`)**

This is the core learning objective. Implement three strategies:
1. Fixed-size character splitter (baseline)
2. Sentence-aware chunker (use `nltk.sent_tokenize`, group sentences to token target, carry overlap)
3. Paragraph-aware chunker (split on double newlines, merge small, split large)

Input: `list[CleanedPage]` → Output: `list[Chunk]` (the `Chunk` dataclass is already defined in `types.py`)

Then wire it into `orchestrator.py` Stage 3 (the stub is there, just uncomment + import).

**Step 2 — Build the Analyser (`app/pipeline/analyser.py`)**

Stats on the chunk list: total count, avg/min/max token count, token distribution histogram, overlap coverage %, chunks per page. Returns a dict — add a `/chunks/analyse/{doc_id}` endpoint to expose it.

**Step 3 — Build the Embedder (`app/pipeline/embedder.py`)**

Call HuggingFace Inference API (`HF_API_TOKEN` is already in config). Batch in groups of 32. Exponential backoff on 503. Returns `list[EmbeddedChunk]`.

**Step 4 — Build the Vector Store (`app/pipeline/vector_store.py`)**

ChromaDB wrapper: upsert chunks, query by vector + optional `doc_id` filter, delete by `doc_id`. ChromaDB client is already on `app.state.chroma` — the `get_chroma` dependency in `dependencies.py` surfaces it.

**Step 5 — Build Search + RAG endpoints**

`POST /search/semantic` — embed query, call `collection.query()`, return ranked chunks.
`POST /generate/answer` — top-k retrieval + prompt assembly + HuggingFace LLM call.

---

## Running the Project

```bash
# Local dev
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001

# Docker
docker compose up

# Tests
pytest tests/ -v
```

Environment variables: copy `.env.example` to `.env` and fill in `SECRET_KEY` (required) and optionally `HF_API_TOKEN`, `GOOGLE_CLIENT_ID`, `GITHUB_CLIENT_ID`.

---

## File Map (current actual state)

```
app/
  main.py              — app factory, lifespan, middleware, router registration
  config.py            — Pydantic Settings (reads .env)
  database.py          — SQLAlchemy engine + SessionLocal + Base

  models/
    user.py            — User ORM model (id, email, hashed_password, oauth_*)
    document.py        — Document ORM model + DocumentStatus enum
    job.py             — EmbeddingJob ORM model
    auth_event.py      — AuthEvent append-only log
    password_reset.py  — PasswordResetToken (stores hash only, not raw token)

  schemas/
    auth.py            — Pydantic request/response schemas for auth routes
    document.py        — Pydantic schemas for document routes

  routers/
    auth.py            — /auth/* (register, login, refresh, me, oauth, forgot/reset-password)
    documents.py       — /docs/* (upload, list, get, delete)
    health.py          — /health
    pages.py           — HTML page routes (landing, login, register, dashboard…)

  core/
    security.py        — bcrypt hash/verify, JWT create/verify
    oauth.py           — Authlib Google + GitHub OAuth client setup
    dependencies.py    — get_current_user, get_db, get_chroma FastAPI deps

  pipeline/
    types.py           — ParsedPage, CleanedPage, Chunk, EmbeddedChunk dataclasses
    parser.py          — Stage 1: PDF (PyMuPDF) + DOCX (python-docx) extraction
    cleaner.py         — Stage 2: 8-step text normalisation
    orchestrator.py    — Stage runner: calls stages in order, updates doc status
    [chunker.py]       ← MISSING — Phase 4
    [analyser.py]      ← MISSING — Phase 5
    [embedder.py]      ← MISSING — Phase 6
    [vector_store.py]  ← MISSING — Phase 7

  templates/
    landing.html, login.html, register.html, dashboard.html,
    forgot_password.html, reset_password.html, welcome.html

tests/
  test_health.py       — health endpoint tests
  test_auth.py         — auth flow tests
  test_documents.py    — document API tests
  test_pipeline.py     — parser + cleaner unit tests + integration test
```
