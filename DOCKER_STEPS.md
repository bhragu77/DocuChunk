# Dockerization Steps — DocuChunk

## Project Overview (before we begin)

- **Stack**: Python 3.11, FastAPI, Uvicorn
- **Storage**: SQLite (default) or PostgreSQL, ChromaDB (local persist), file uploads
- **Config**: Environment variables via `.env`
- **Port**: 8000

---

## Step 1 — Create `.dockerignore`

Create a `.dockerignore` file at the repo root to prevent bloating the image with files that don't belong inside the container.

**File: `.dockerignore`**
```
.git
__pycache__
**/__pycache__
*.pyc
*.pyo
.env
.pytest_cache
tests/
build/
chroma_db/
uploads/
*.db
logs/
*.egg-info
.venv
venv/
```

**Why**: Without this, `COPY . .` in the Dockerfile would bundle the local `uploads/`, `chroma_db/`, `.env` secrets, and hundreds of MB of test data into the image layer. Runtime data (uploads, chroma, db) belongs in Docker volumes, not the image.

---

## Step 2 — Create `Dockerfile`

Create a `Dockerfile` at the repo root.

**File: `Dockerfile`**
```dockerfile
FROM python:3.11-slim

# Prevents Python from writing .pyc files and buffers stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (separate layer — rebuilds only when requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ ./app/
COPY static/ ./static/

# Create directories the app expects at runtime
RUN mkdir -p uploads chroma_db logs

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Why each line**:
- `python:3.11-slim` — matches `runtime.txt` (3.11.9), slim keeps image small
- `COPY requirements.txt` before `COPY app/` — Docker layer cache means pip only reruns when `requirements.txt` changes, not on every code edit
- `RUN mkdir -p uploads chroma_db logs` — app creates these on startup too, but having them owned by the right user avoids permission errors on some systems
- `EXPOSE 8000` — documents the port; actual binding is in docker-compose

---

## Step 3 — Create `docker-compose.yml`

Create a `docker-compose.yml` at the repo root.

**File: `docker-compose.yml`**
```yaml
services:
  app:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    environment:
      - APP_ENV=development
      - UPLOAD_DIR=/app/uploads
      - CHROMA_PERSIST_DIR=/app/chroma_db
      - DATABASE_URL=sqlite:////app/data/app.db
    volumes:
      - uploads_data:/app/uploads
      - chroma_data:/app/chroma_db
      - sqlite_data:/app/data
    restart: unless-stopped

volumes:
  uploads_data:
  chroma_data:
  sqlite_data:
```

**Why named volumes instead of bind mounts**:
- Named volumes (`uploads_data`, etc.) are managed by Docker — they persist across `docker compose down` and survive even if you move the repo
- The `environment:` block inside the service overrides values from `.env` so the paths point to locations *inside* the container (`/app/uploads`) rather than the host's `./uploads`
- `DATABASE_URL` uses 4 slashes (`sqlite:////app/data/app.db`) — that's the SQLite syntax for an absolute path inside the container

---

## Step 4 — Update `.env` for container compatibility (optional but clean)

The `.env` file has paths like `./uploads` and `./chroma_db` which are relative to the host. In the container these are overridden by the `environment:` block in `docker-compose.yml` (Step 3), so `.env` doesn't need to change.

However, `SECRET_KEY` **must** be set in `.env` — the app will refuse to start without it.

Generate one if you haven't:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Then in `.env`:
```
SECRET_KEY=<paste the output here>
```

---

## Step 5 — Build the image

```bash
docker build -t docuchunk .
```

Watch for errors. Common issues:
- A package in `requirements.txt` needs a system library (e.g., `pymupdf` needs `libmupdf`) — fix by adding `apt-get install` lines in the Dockerfile before `pip install`
- If the build succeeds, `docker images | grep docuchunk` should show the image

---

## Step 6 — Start the stack

```bash
docker compose up
```

Or in detached (background) mode:
```bash
docker compose up -d
```

Check logs at any time with:
```bash
docker compose logs -f app
```

---

## Step 7 — Verify the app is running

Open in browser: `http://localhost:8000`

Or via curl:
```bash
curl http://localhost:8000/health
```

You should get a JSON health response.

---

## Step 8 — Stopping and cleaning up

Stop the containers (volumes are kept):
```bash
docker compose down
```

Stop AND delete all volumes (wipes uploads, chroma, db — destructive):
```bash
docker compose down -v
```

---

## Notes / What to watch for during the build

1. **`chromadb` and `umap-learn`** pull in large native dependencies (`hnswlib`, `numpy`, BLAS). The first build will be slow (~5–10 min). Subsequent builds are fast due to layer caching.

2. **`pymupdf`** ships pre-compiled wheels for `python:3.11-slim` on x86_64 — should install cleanly. On ARM (M1/M2 Mac), you may need `FROM python:3.11` (full image) instead of slim.

3. **`nltk` data**: The app uses `nltk` for tokenization. If the app throws an `LookupError` for `punkt` or similar, add this to the Dockerfile after `pip install`:
   ```dockerfile
   RUN python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger
   ```

4. **`tiktoken`** downloads tokenizer files on first use to a cache dir. Mount a volume or set `TIKTOKEN_CACHE_DIR=/app/.tiktoken` if you want to persist this.

