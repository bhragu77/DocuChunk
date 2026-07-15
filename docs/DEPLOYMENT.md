# Deployment Guide — DocuChunk (local embeddings + Gemini)

Exact, copy-paste steps to deploy the **same stack you run locally** — **local
sentence-transformers embeddings + Google Gemini generation + Chroma (HTTP server)
+ Redis + arq worker + SQLite** — live on a **free** Oracle Cloud VM using your
existing Docker Compose, with HTTPS and a free domain.

> **Cost:** $0/month (Oracle Always Free + DuckDNS + Gemini free tier).
> **Time:** ~60–90 min first time.

---

## 0. The stack you're deploying (identical to local)

```
                           ┌──────────── Oracle Always-Free VM (ARM, 12–24 GB RAM) ────────────┐
 Visitor ──HTTPS(443)────► │  caddy  ──►  app (FastAPI: UI + API)                                │
                           │                │  ├─ EMBED_PROVIDER=local  → loads all-MiniLM-L6-v2 │
                           │                │  │                          in-process (query time)│
                           │                │  ├─ GEN_PROVIDER=gemini    → calls Gemini API      │
                           │                │  ├─ redis   (arq queue + answer cache)             │
                           │                │  ├─ worker  (arq: parse→chunk→EMBED→index;         │
                           │                │  │            loads all-MiniLM at startup)         │
                           │                │  └─ chroma  (vector DB, HTTP server)               │
                           │   shared volumes: uploads · chroma_data · sqlite · hf_cache         │
                           └────────────────────────────────────────────────────────────────────┘
      External API the app calls: Google Gemini (generation).   Local model = embeddings (no API).
```

**What "local model" means here:** embeddings are computed **on the VM** by
`sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim) — no embedding API. The worker
loads it at startup; the web process loads it lazily on the first search. The
cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) also loads locally on first search.
**Only generation uses an external API (Gemini).**

| Component | Value | Runs where |
|---|---|---|
| Embeddings | `EMBED_PROVIDER=local`, `all-MiniLM-L6-v2` | on the VM (worker + web) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | on the VM (web, first search) |
| Generation | `GEN_PROVIDER=gemini`, `gemini-3.1-flash-lite` | Gemini API |
| Vector store | Chroma **HTTP server** (`CHROMA_MODE=http`) | `chroma` container |
| Queue + cache | Redis | `redis` container |
| Database | SQLite on a volume | `sqlite_data` volume |

> **First-run downloads (need internet, cached afterward in the `hf_cache` volume):**
> the MiniLM embedder (~90 MB) and the cross-encoder (~80 MB) are pulled from
> HuggingFace the first time. **RAM:** torch + both models load in the worker AND the
> web process → budget **~2–3 GB**. Use the **12–24 GB** Ampere shape (below), never
> the 1 GB micro.

---

## 1. Accounts (all free)

1. **Oracle Cloud** — https://cloud.oracle.com (card for identity check; Always Free is never charged).
2. **DuckDNS** — https://www.duckdns.org (free subdomain).
3. **Gemini API key** — https://aistudio.google.com/apikey → *Create API key*. Copy it.
4. *(Optional)* **Google OAuth** creds (only if you enable "Sign in with Google").

---

## 2. Create the Oracle Always-Free VM

1. Sign in → **Compute → Instances → Create instance**.
2. **Image:** Ubuntu 24.04 — **aarch64 (ARM)** build.
3. **Shape:** *Change shape* → **Ampere** → `VM.Standard.A1.Flex` → **2 OCPU / 12 GB RAM**
   (within Always Free: 4 OCPU / 24 GB total). *Out of capacity? Switch Availability
   Domain or region and retry.*
4. **SSH keys:** upload your public key (`ssh-keygen -t ed25519` if you need one).
5. Keep "Assign public IPv4 = Yes". **Create**. Note the **public IP**.

### 2.1 Open ports (TWO places — Oracle blocks by default)

**a) Cloud:** Instance → its VCN → **Security Lists** → default → **Add Ingress Rules**:
`0.0.0.0/0` TCP **80**, and `0.0.0.0/0` TCP **443**.

**b) OS firewall** (after you SSH in, step 3):
```bash
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 80  -j ACCEPT
sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save
```

---

## 3. Install Docker + get the code

SSH in (user `ubuntu`):
```bash
ssh ubuntu@YOUR_PUBLIC_IP
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker ubuntu
exit                       # log out so the docker group applies
```
Re-SSH, then clone:
```bash
ssh ubuntu@YOUR_PUBLIC_IP
git clone https://github.com/YOUR_USER/DocumentChunking.git
cd DocumentChunking
```
> Private repo → use a GitHub Personal Access Token in the URL, or `scp -r ./DocumentChunking ubuntu@YOUR_PUBLIC_IP:~/`.

---

## 4. Free domain (DuckDNS) → your VM

1. https://www.duckdns.org → sign in → create a subdomain, e.g. **`docuchunk`** → `docuchunk.duckdns.org`.
2. Set its IP to your VM's **public IP**, click **update ip**.
3. From your laptop: `ping docuchunk.duckdns.org` should show the VM IP.

---

## 5. Configure `.env` (secrets + your stack)

```bash
cp .env.production.example .env
python3 -c "import secrets; print(secrets.token_hex(32))"   # copy this
nano .env
```
Set these (the template documents every line):

```ini
# Security
SECRET_KEY=<paste the token_hex(32) you just generated>

# Your domain — CORS must include the backend domain (+ any separate frontend)
ALLOWED_ORIGINS=https://docuchunk.duckdns.org

# Generation → Gemini
GEN_PROVIDER=gemini
GEMINI_API_KEY=<your Gemini API key>
GEN_MODEL=gemini-3.1-flash-lite

# Embeddings → LOCAL model (default; explicit here)
EMBED_PROVIDER=local
HF_EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Google OAuth — ONLY if you use it (else leave blank)
GOOGLE_REDIRECT_URI=https://docuchunk.duckdns.org/auth/oauth/google/callback
```

> **Do NOT set** `DATABASE_URL`, `CHROMA_MODE/HOST`, `REDIS_HOST`, `UPLOAD_DIR` here —
> `docker-compose.yml` already wires those (SQLite volume + Chroma HTTP server + Redis).
> `CACHE_BACKEND=redis` and `APP_ENV=production` are set by `docker-compose.prod.yml`.

---

## 6. Point Caddy at your domain

```bash
nano Caddyfile
```
```caddy
docuchunk.duckdns.org {
    encode gzip
    reverse_proxy app:8000
}
```

---

## 7. Launch 🚀

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```
- The build installs `sentence-transformers`, `torch`, `chromadb`, `google-genai`, etc. (a few minutes on ARM).
- On **first startup the worker downloads the MiniLM model** (~90 MB) into the `hf_cache` volume — watch for it:

```bash
docker compose ps                      # app, worker, chroma, redis, caddy all "running"
docker compose logs -f worker          # look for the sentence-transformers model load
docker compose logs -f app             # "Generation wired [gen=gemini ...]" confirms Gemini
```

### Verify end to end
1. `curl -sk https://docuchunk.duckdns.org/health` → healthy JSON.
2. Open `https://docuchunk.duckdns.org` → login page, valid HTTPS padlock (Caddy auto-gets the cert; give it ~30 s on first hit).
3. **Register** a user → **upload a PDF/DOCX** → wait for status **`ready`** (the worker parses → chunks → embeds locally → indexes).
4. Ask a question → you should get a streamed, **cited** answer from Gemini. (First search also downloads the cross-encoder ~80 MB — one-time.)

---

## 8. Google OAuth (skip if unused)

https://console.cloud.google.com → APIs & Services → Credentials → your OAuth client:
- **Authorized redirect URIs** → add `https://docuchunk.duckdns.org/auth/oauth/google/callback` (must byte-match `GOOGLE_REDIRECT_URI`).
- **Authorized JavaScript origins** → add `https://docuchunk.duckdns.org`.
- If the OAuth app is in "Testing", add your account as a test user.

---

## 9. Operations

### Update after a code change
```bash
cd ~/DocumentChunking
git pull
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
```

### Back up the data (volumes hold everything)
```bash
docker volume ls          # confirm exact names (prefix documentchunking_)
docker run --rm \
  -v documentchunking_sqlite_data:/db -v documentchunking_chroma_data:/vec \
  -v documentchunking_uploads_data:/up -v "$PWD":/backup alpine \
  tar czf /backup/backup-$(date +%F).tgz /db /vec /up
```
(Or use an Oracle boot-volume snapshot.)

### Logs / restart
```bash
docker compose logs -f worker      # ingestion + local embedding
docker compose logs -f app         # API + Gemini generation
docker compose logs -f caddy       # TLS / proxy
docker compose restart app worker
```

### Keep the Always-Free VM from being reclaimed
Oracle can reclaim **idle** Always-Free compute. Normal use keeps it active; if it'll
sit unused, a small cron `curl https://…/health` every few minutes keeps it above the
idle threshold.

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `Out of host capacity` creating the VM | Retry; switch Availability Domain/region (ARM capacity is intermittent). |
| Site unreachable but `docker compose ps` healthy | Re-check **both** the Oracle Security List AND the OS `iptables` rules (2.1). |
| HTTPS cert not issued | DNS must point at the VM first (step 4); ports 80+443 open; `docker compose logs caddy`. |
| Uploads never reach `ready` | `docker compose logs worker` — the **first run must download the MiniLM model** (needs outbound internet). Wait for it; ensure `redis` + `chroma` are up. |
| Answers say `generation_not_configured` | `GEN_PROVIDER=gemini` + a valid `GEMINI_API_KEY` in `.env`, then `up -d`. |
| Search returns nothing for a ready doc | Confirm the worker logged the model load (local embeddings) and `chroma` is running; check `docker compose logs worker`. |
| Out of memory / OOM-killed | Use the **12–24 GB** Ampere shape — torch + MiniLM + cross-encoder load in both worker and web. |
| Google login `redirect_uri_mismatch` | `GOOGLE_REDIRECT_URI` must byte-match the URI registered in Google Console (step 8). |

---

## Appendix — files this guide uses (all in the repo)

- **`docker-compose.yml`** — base stack: `app`, `worker` (arq), `chroma` (HTTP server), `redis`. Unchanged.
- **`docker-compose.prod.yml`** — production overlay: adds Caddy, prod mode, Redis answer cache, and the `hf_cache` volume that persists the downloaded local models.
- **`Caddyfile`** — reverse proxy + automatic HTTPS (edit the hostname).
- **`.env.production.example`** — annotated secrets template → copy to `.env`.

**One-line launch:** `docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build`
