# Document Retrieval Platform — Implementation Plan

## Project Overview

A production-grade, SaaS-style Retrieval-Augmented Generation (RAG) platform where users upload
PDF/DOCX documents, the system parses, chunks, embeds, and stores them in ChromaDB, then serves
semantic search and generation APIs — all with JWT + OAuth authentication and a landing page UI.

**Primary goal**: Learn every layer deeply (chunking → embeddings → vector DB) while shipping
something production-ready.

---

## Architecture at a Glance

```
Browser / Client
      │
      ▼
┌─────────────────────────────────────────────────────┐
│                   Landing Page (HTML/JS)             │
│         Login · Register · Dashboard · Upload        │
└──────────────────────┬──────────────────────────────┘
                       │ HTTPS
                       ▼
┌─────────────────────────────────────────────────────┐
│              FastAPI Application Layer               │
│                                                     │
│  /auth      JWT issue · OAuth (Google)              │
│  /docs      Upload · List · Delete                  │
│  /chunks    View chunks · Analyse chunk stats       │
│  /embed     Trigger embedding · Status              │
│  /search    Semantic search (query → top-k chunks)  │
│  /generate  RAG answer generation                   │
│  /debug     Inspect embeddings · visualise UMAP     │
└───────┬──────────────────────────┬──────────────────┘
        │                          │
        ▼                          ▼
┌──────────────┐         ┌─────────────────────┐
│  SQLite /    │         │     ChromaDB         │
│  PostgreSQL  │         │  (vector store)      │
│  (users,     │         │  Collections per     │
│   documents, │         │  user / document     │
│   jobs)      │         └─────────────────────┘
└──────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────┐
│               Background Pipeline                    │
│                                                      │
│  1. Parse      (PyMuPDF for PDF, python-docx)        │
│  2. Clean      (normalise whitespace, remove noise)  │
│  3. Chunk      (YOUR custom chunker — learn here)    │
│  4. Analyse    (token count, avg size, overlap map)  │
│  5. Embed      (HuggingFace free model via API)      │
│  6. Store      (ChromaDB upsert with metadata)       │
└──────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Web framework | FastAPI | Async, auto OpenAPI docs, production-ready |
| Auth | python-jose (JWT) + Authlib (OAuth) | Industry standard |
| Document parse | PyMuPDF (PDF) + python-docx (DOCX) | Free, robust |
| Chunking | **Custom (hand-written)** | Core learning objective |
| Embeddings | HuggingFace Inference API (free tier) — `sentence-transformers/all-MiniLM-L6-v2` | Free, 768-dim, strong baseline |
| Vector DB | ChromaDB (local persistent) | Easy to learn internals |
| Relational DB | SQLite (dev) → PostgreSQL (prod) via SQLAlchemy | Easy migration path |
| Task queue | FastAPI BackgroundTasks → Celery + Redis (phase 2) | Start simple, scale later |
| Visualisation | UMAP + Matplotlib (debug endpoint returns image) | See embeddings in 2D |
| Frontend | Vanilla HTML/CSS/JS (Jinja2 templates) | No build step, easy to follow |
| Config | python-dotenv + Pydantic Settings | 12-factor app |

---

## Project Directory Structure

```
DocumentChunking/
│
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app factory, lifespan, middleware
│   ├── config.py                # Pydantic Settings (reads .env)
│   ├── database.py              # SQLAlchemy engine + session
│   │
│   ├── models/                  # SQLAlchemy ORM models
│   │   ├── user.py
│   │   ├── document.py
│   │   └── job.py
│   │
│   ├── schemas/                 # Pydantic request/response schemas
│   │   ├── auth.py
│   │   ├── document.py
│   │   ├── chunk.py
│   │   └── search.py
│   │
│   ├── routers/                 # One file per feature domain
│   │   ├── auth.py              # /auth/register, /auth/login, /auth/oauth/google
│   │   ├── documents.py         # /docs/upload, /docs/list, /docs/delete
│   │   ├── chunks.py            # /chunks/{doc_id}, /chunks/analyse/{doc_id}
│   │   ├── embeddings.py        # /embed/trigger, /embed/status/{job_id}
│   │   ├── search.py            # /search/semantic, /search/keyword
│   │   ├── generate.py          # /generate/answer (RAG endpoint)
│   │   └── debug.py             # /debug/visualise, /debug/inspect
│   │
│   ├── core/
│   │   ├── security.py          # JWT create/verify, password hashing
│   │   ├── oauth.py             # Google OAuth flow
│   │   └── dependencies.py      # get_current_user, get_db FastAPI deps
│   │
│   ├── pipeline/                # ← THE LEARNING HEART OF THE PROJECT
│   │   ├── parser.py            # PDF + DOCX text extraction
│   │   ├── cleaner.py           # Text normalisation
│   │   ├── chunker.py           # YOUR custom chunking logic
│   │   ├── analyser.py          # Chunk stats, overlap analysis
│   │   ├── embedder.py          # HuggingFace API calls + retry logic
│   │   └── vector_store.py      # ChromaDB wrapper (CRUD on collections)
│   │
│   └── templates/               # Jinja2 HTML templates
│       ├── base.html
│       ├── landing.html
│       ├── login.html
│       ├── register.html
│       └── dashboard.html
│
├── static/
│   ├── css/style.css
│   └── js/app.js
│
├── uploads/                     # Raw uploaded files (gitignored)
├── chroma_db/                   # ChromaDB persistent storage (gitignored)
├── logs/
│
├── tests/
│   ├── test_auth.py
│   ├── test_chunker.py          # Most important — unit test your chunker
│   ├── test_pipeline.py
│   └── test_search.py
│
├── .env.example
├── .gitignore
├── requirements.txt
└── IMPLEMENTATION_PLAN.md       ← this file
```

---

## Phase-by-Phase Build Plan

### Phase 0 — Project Bootstrap (Day 1)

- [ ] Create directory structure
- [ ] Write `requirements.txt`
- [ ] Write `.env.example` with all required keys
- [ ] Set up SQLAlchemy + SQLite, create tables
- [ ] FastAPI app factory with lifespan (DB init + ChromaDB init on startup)
- [ ] Health check endpoint `GET /health`

**Learning checkpoint**: Understand FastAPI lifespan events and dependency injection.

---

### Phase 1 — Authentication (Day 2–3)

#### 1a. JWT Auth
- `POST /auth/register` — hash password (bcrypt), store user in DB, return JWT
- `POST /auth/login` — verify password, return access token + refresh token
- Token middleware: `get_current_user` dependency injected into protected routes
- Access token: 30 min expiry; Refresh token: 7 days

#### 1b. OAuth (Google)
- Register app in Google Cloud Console (free)
- `GET /auth/oauth/google` — redirect to Google consent screen
- `GET /auth/oauth/google/callback` — exchange code for tokens, upsert user, issue JWT
- Store `oauth_provider` + `oauth_id` on user model

#### Key concepts to learn:
- JWT structure: header.payload.signature
- PKCE flow for OAuth
- Why we still issue our own JWT after OAuth (session ownership)

---

### Phase 2 — Document Upload & Storage (Day 4)

- `POST /docs/upload` — multipart form, validate extension (PDF/DOCX only), save to `uploads/`
- Store document record in DB: `id, user_id, filename, file_path, status, created_at`
- `GET /docs/list` — paginated list of user's documents
- `DELETE /docs/{doc_id}` — delete file + DB record + ChromaDB collection
- File size limit: 20 MB
- Status field: `uploaded → parsing → chunking → embedding → ready | failed`

---

### Phase 3 — Parsing Pipeline (Day 5)

#### `pipeline/parser.py`
```
PDF  → PyMuPDF (fitz) → extract text page by page → preserve page numbers in metadata
DOCX → python-docx   → extract paragraphs + tables → paragraph index in metadata
```

Output: `List[dict]` where each dict = `{ "text": str, "page": int, "source": str }`

**What you learn**: Raw text from documents is dirty. You'll see encoding issues,
hyphenated words across lines, headers/footers bleeding into body text.

#### `pipeline/cleaner.py`
- Remove excessive whitespace
- Fix hyphenation at line breaks
- Strip page numbers / headers detected by regex
- Unicode normalisation (NFKC)

---

### Phase 4 — Custom Chunker (Day 6–7) ← CORE LEARNING

This is where the platform teaches you the most. Build three strategies and compare them.

#### `pipeline/chunker.py`

**Strategy 1: Fixed-size character chunker**
```
Split text every N characters with M characters overlap.
Simple baseline. Breaks sentences arbitrarily.
```

**Strategy 2: Sentence-aware chunker**
```
Use nltk.sent_tokenize to split into sentences first.
Group sentences until chunk hits target token count.
Start new chunk, carry last K sentences as overlap.
```

**Strategy 3: Semantic paragraph chunker** (the interesting one)
```
Split on double newlines (paragraphs).
If paragraph > max_size  → split further using sentence chunker.
If paragraph < min_size  → merge with next paragraph.
Result: chunks that respect natural document boundaries.
```

**Chunker config object** (passed at runtime, stored with each job):
```python
{
  "strategy": "sentence_aware",   # fixed | sentence_aware | paragraph
  "chunk_size": 512,              # target tokens per chunk
  "chunk_overlap": 64,            # overlap tokens between consecutive chunks
  "min_chunk_size": 100,          # discard chunks smaller than this
}
```

Output: `List[Chunk]` where `Chunk` has:
- `chunk_id` (UUID)
- `doc_id`
- `text`
- `char_start`, `char_end` (position in original text)
- `page_number`
- `chunk_index`
- `token_count`
- `metadata` (dict, stored in ChromaDB)

---

### Phase 5 — Chunk Analyser (Day 8)

#### `pipeline/analyser.py` + `/chunks/analyse/{doc_id}`

Returns stats you can use to tune your chunker:
```json
{
  "total_chunks": 142,
  "avg_token_count": 487,
  "min_token_count": 103,
  "max_token_count": 512,
  "token_distribution": { "0-100": 2, "100-300": 18, ... },
  "overlap_coverage_pct": 12.5,
  "pages_covered": [1, 2, 3, ..., 28],
  "chunks_per_page": { "1": 5, "2": 7, ... }
}
```

**What you learn**: Why chunk size matters for retrieval quality.
Too small → context lost. Too large → noise in retrieval.

---

### Phase 6 — Embedding Generation (Day 9)

#### `pipeline/embedder.py`

**Free model**: HuggingFace Inference API
- Model: `sentence-transformers/all-MiniLM-L6-v2`
- Dimension: 384
- Free tier: 30,000 tokens/month (enough for learning)
- Sign up at huggingface.co → get `HF_API_TOKEN`

```python
# API call pattern
POST https://api-inference.huggingface.co/pipeline/feature-extraction/
     sentence-transformers/all-MiniLM-L6-v2
Headers: Authorization: Bearer {HF_API_TOKEN}
Body: { "inputs": ["chunk text 1", "chunk text 2", ...] }
Response: [[0.12, -0.34, ...], ...]   # list of embedding vectors
```

- Batch chunks in groups of 32 to stay within API limits
- Retry with exponential backoff on 503 (model loading)
- Store job progress in DB: `embedded_count / total_chunks`

**What you learn**: What an embedding vector actually is —
a point in 384-dimensional space where semantic similarity = cosine proximity.

---

### Phase 7 — ChromaDB Storage (Day 10)

#### `pipeline/vector_store.py`

One ChromaDB **collection per user** (named by `user_id`).
Each chunk is upserted with:
- `id`: chunk_id (UUID)
- `embedding`: the 384-dim vector
- `document`: chunk text
- `metadata`: `{ doc_id, page_number, chunk_index, strategy, chunk_size, ... }`

```python
# ChromaDB operations you'll learn
collection.upsert(ids=[...], embeddings=[...], documents=[...], metadatas=[...])
collection.query(query_embeddings=[...], n_results=5, where={"doc_id": "xyz"})
collection.delete(where={"doc_id": "xyz"})
collection.get(ids=[...], include=["embeddings", "documents", "metadatas"])
```

**What you learn**: ChromaDB stores embeddings on disk using HNSW index.
Query time is O(log n) not O(n) — that is why vector DBs are fast.

---

### Phase 8 — Search & RAG (Day 11–12)

#### `POST /search/semantic`
```json
Request:  { "query": "what is the refund policy?", "doc_id": "optional", "top_k": 5 }
Response: { "results": [ { "text": "...", "score": 0.92, "page": 3 }, ... ] }
```

Flow:
1. Embed the query using same model
2. `collection.query(query_embeddings=[query_vec], n_results=top_k)`
3. Return ranked chunks with similarity scores

#### `POST /generate/answer` (RAG)
```json
Request:  { "query": "...", "doc_id": "...", "top_k": 5 }
Response: { "answer": "...", "sources": [...] }
```

Flow:
1. Semantic search → top-k chunks
2. Build prompt: `Context:\n{chunks}\n\nQuestion: {query}\nAnswer:`
3. Call free LLM: HuggingFace `google/flan-t5-base` or `mistralai/Mistral-7B-Instruct-v0.2`
4. Return answer + source chunks (for citation)

---

### Phase 9 — Debug & Visualisation (Day 13)

#### `GET /debug/visualise/{doc_id}`
- Fetch all chunk embeddings from ChromaDB for a document
- Run UMAP to reduce 384-dim → 2-dim
- Plot with Matplotlib, colour by page number
- Return PNG image
- **What you see**: Chunks from the same topic cluster together visually

#### `GET /debug/inspect/{chunk_id}`
- Return raw embedding vector
- Cosine similarity to 5 nearest neighbours
- Token count, text, position in document

---

### Phase 10 — Landing Page & Dashboard (Day 14–15)

#### Pages
1. **Landing** (`/`) — Hero, feature list, "Get Started" CTA
2. **Register** (`/auth/register`) — Name, email, password form
3. **Login** (`/auth/login`) — Email/password + "Login with Google" button
4. **Dashboard** (`/dashboard`) — Upload document, view document list, pipeline status
5. **Document Detail** (`/dashboard/doc/{id}`) — Chunk viewer, analyser stats, search box

#### Auth flow in browser
- After login → store JWT in `httpOnly` cookie (more secure than localStorage)
- JS fetches APIs with `credentials: "include"`
- 401 response → redirect to login page

---

## Data Models (SQLAlchemy)

### User
```
id, email, hashed_password, full_name,
oauth_provider, oauth_id,
is_active, created_at
```

### Document
```
id, user_id (FK), filename, original_filename,
file_path, file_size, mime_type,
status (enum), error_message,
chunk_count, chunker_config (JSON),
created_at, updated_at
```

### EmbeddingJob
```
id, document_id (FK), user_id (FK),
status (enum), total_chunks, embedded_chunks,
model_name, started_at, completed_at, error
```

---

## Environment Variables (.env)

```bash
# App
SECRET_KEY=your-secret-key-min-32-chars
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# Database
DATABASE_URL=sqlite:///./app.db

# ChromaDB
CHROMA_PERSIST_DIR=./chroma_db

# HuggingFace (free)
HF_API_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx

# Google OAuth (free)
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/oauth/google/callback

# Storage
UPLOAD_DIR=./uploads
MAX_FILE_SIZE_MB=20

# CORS
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8000
```

---

## requirements.txt (planned)

```
fastapi>=0.111.0
uvicorn[standard]>=0.30.0
python-multipart>=0.0.9        # file uploads
python-jose[cryptography]>=3.3.0  # JWT
passlib[bcrypt]>=1.7.4         # password hashing
authlib>=1.3.0                 # OAuth
httpx>=0.27.0                  # async HTTP (OAuth + HF API)
sqlalchemy>=2.0.0
alembic>=1.13.0                # DB migrations
pydantic-settings>=2.0.0
python-dotenv>=1.0.0

# Document parsing
pymupdf>=1.24.0                # PDF (fitz)
python-docx>=1.1.0             # DOCX

# NLP / chunking
nltk>=3.8.0                    # sentence tokenisation
tiktoken>=0.7.0                # token counting (OpenAI tokeniser, free to use)

# Vector DB
chromadb>=0.5.0

# Visualisation / debug
umap-learn>=0.5.6
matplotlib>=3.9.0
numpy>=1.26.0

# Templates
jinja2>=3.1.0

# Testing
pytest>=8.0.0
pytest-asyncio>=0.23.0
httpx>=0.27.0                  # test client
```

---

## API Contract Summary

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | /health | None | Liveness check |
| POST | /auth/register | None | Create account |
| POST | /auth/login | None | Get JWT tokens |
| GET | /auth/oauth/google | None | Start Google OAuth |
| GET | /auth/oauth/google/callback | None | OAuth callback |
| POST | /auth/refresh | Refresh token | Get new access token |
| POST | /docs/upload | JWT | Upload PDF/DOCX |
| GET | /docs/list | JWT | List user documents |
| GET | /docs/{id} | JWT | Document details |
| DELETE | /docs/{id} | JWT | Delete document |
| GET | /chunks/{doc_id} | JWT | List chunks for doc |
| GET | /chunks/analyse/{doc_id} | JWT | Chunk stats |
| POST | /embed/trigger/{doc_id} | JWT | Start embedding job |
| GET | /embed/status/{job_id} | JWT | Job progress |
| POST | /search/semantic | JWT | Vector search |
| POST | /search/keyword | JWT | Keyword fallback |
| POST | /generate/answer | JWT | RAG answer |
| GET | /debug/visualise/{doc_id} | JWT | UMAP PNG |
| GET | /debug/inspect/{chunk_id} | JWT | Raw embedding |

---

## Build Order (recommended sequence)

```
Phase 0  →  Phase 1a  →  Phase 1b  →  Phase 2
   ↓
Phase 3 (parse)  →  Phase 4 (chunk)  →  Phase 5 (analyse)
   ↓
Phase 6 (embed)  →  Phase 7 (store)  →  Phase 8 (search + RAG)
   ↓
Phase 9 (debug)  →  Phase 10 (UI)
```

Test each phase before moving to the next.
Write at least one pytest test per pipeline module.

---

## Scalability Path (after MVP)

| Concern | MVP approach | Production upgrade |
|---|---|---|
| Task queue | BackgroundTasks | Celery + Redis |
| Database | SQLite | PostgreSQL |
| Storage | Local filesystem | S3 / GCS |
| ChromaDB | Local persistent | Chroma Cloud or Qdrant |
| Embeddings | HuggingFace free API | Self-hosted model (Ollama) |
| Auth | Single server JWT | Redis session store + token rotation |
| Rate limiting | None | slowapi + Redis |
| Logging | print / basicConfig | structlog → Loki / CloudWatch |

---

## Learning Milestones

After completing each phase you should be able to answer:

- Phase 4: Why does chunk overlap exist? What happens if you set overlap to 0?
- Phase 5: What is a good average token count per chunk for this document type?
- Phase 6: What does cosine similarity actually measure between two embedding vectors?
- Phase 7: What is HNSW and why does it make vector search fast?
- Phase 8: Why does RAG outperform a plain LLM on document-specific questions?
- Phase 9: When you visualise embeddings with UMAP, what do tight clusters mean?

---

*Plan written: 2026-06-19*
*Start with Phase 0 — bootstrap the project.*
