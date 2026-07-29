from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    # App
    app_name: str = "DocuChunk"
    app_version: str = "0.1.0"
    app_env: str = "development"
    debug: bool = True

    # Security
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # Database
    database_url: str = "sqlite:///./app.db"
    # Vector backend: "chroma" (default, embedded/http) | "pgvector" (Postgres).
    # pgvector keeps relational rows and embeddings in ONE database, which removes
    # the dual-store consistency problem and is the path to a managed deployment.
    vector_backend: str = "chroma"
    pgvector_dsn: str = ""          # defaults to DATABASE_URL when empty
    # Pinecone — the MANAGED backend. Selected with VECTOR_BACKEND=pinecone. One
    # index serves every user; per-user isolation is by namespace, which matters on
    # the free tier where you get exactly one index.
    # Retrieval enhancements — both OFF by default. Each costs something on the hot
    # path (an LLM call, or N embeddings per query) and each is expected to help some
    # query categories while hurting others, so enabling them is a measured decision
    # per deployment, not a default. See eval/RETRIEVAL_IMPROVEMENTS.md.
    query_rewrite_enabled: bool = False
    query_rewrite_n: int = 3
    mmr_enabled: bool = False
    mmr_lambda: float = 0.7

    pinecone_api_key: str = ""
    pinecone_index: str = "docuchunk"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # ChromaDB — client mode is a single, explicit config switch. Nothing else in
    # the codebase decides persistent-vs-http; app/core/chroma.build_chroma_client
    # is the one construction site and reads only these settings.
    #   persistent → chromadb.PersistentClient(path=chroma_persist_dir)   [local/dev/test]
    #   http       → chromadb.HttpClient(host=chroma_host, port=chroma_port) [deployed server]
    chroma_mode: str = "persistent"   # "persistent" | "http"
    # Persistent-mode path. Accepts CHROMA_PATH or CHROMA_PERSIST_DIR as the env var.
    chroma_persist_dir: str = Field(
        default="./chroma_db",
        validation_alias=AliasChoices("CHROMA_PATH", "CHROMA_PERSIST_DIR"),
    )
    # Http-mode target (only used when chroma_mode == "http").
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    # HNSW index tuning (applied at collection creation — see app/core/chroma.py).
    #   m                → bi-directional links per node. ↑ recall + memory, slower build.
    #   construction_ef  → candidate list size while BUILDING. ↑ graph quality, slower ingest.
    #   search_ef        → candidate list size while QUERYING. THE recall⇄latency dial:
    #                      ↑ = higher recall (fewer missed chunks), slower query.
    # Defaults raise search_ef well above Chroma's default (10) for better recall.
    hnsw_m: int = 16
    hnsw_construction_ef: int = 200
    hnsw_search_ef: int = 100
    # Max ids per single Chroma upsert. Chroma caps how many records one upsert call
    # may carry; the batch-upsert sink (Part 2) splits any sink batch larger than this
    # into multiple upsert calls. Keep comfortably under Chroma's own limit.
    chroma_upsert_max: int = 4000

    # HuggingFace
    hf_api_token: str = ""
    hf_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_gen_model: str = "google/flan-t5-base"

    # Generation (Story 1) — the LLM layer on top of retrieval. GEN_PROVIDER selects
    # the backend; app/generation/factory.build_gen_provider is the one construction
    # site. Default "stub" keeps generation OFFLINE/safe until a real key is configured.
    #   stub          → deterministic offline provider (tests + default)
    #   gemini        → google-genai SDK (needs GEMINI_API_KEY)
    #   openai_compat → any OpenAI-compatible /chat/completions (DeepSeek/Groq/Ollama)
    gen_provider: str = "stub"          # "stub" | "gemini" | "openai_compat"
    gemini_api_key: str = ""            # env-only — NEVER commit a real key
    gen_model: str = "gemini-3.1-flash-lite"
    gen_temperature: float = 0.3        # low for grounded QA
    gen_max_tokens: int = 1024
    gen_timeout_s: float = 30.0
    # OpenAI-compatible fallback provider target (only used when gen_provider == "openai_compat")
    openai_compat_base_url: str = ""    # e.g. https://api.deepseek.com/v1 or http://localhost:11434/v1
    openai_compat_api_key: str = ""
    openai_compat_model: str = ""

    # ── Chat model picker: the "offline" tier ─────────────────────────────────
    # The chat UI lets the user pick per-message between "gemini" (accurate cloud)
    # and "offline" (a small local LLM served by Ollama over its OpenAI-compatible
    # API, so it reuses OpenAICompatProvider — no new provider code). This is
    # INDEPENDENT of GEN_PROVIDER (the default answer path): both can be wired at
    # once, and only the GENERATION model swaps per request — retrieval, reranking,
    # prompt assembly and verification are unchanged. When OFFLINE_ENABLED is false
    # or Ollama is unreachable, the UI shows "offline" as unavailable and the
    # endpoint falls back to the default generation seam.
    offline_enabled: bool = False
    offline_base_url: str = "http://ollama:11434/v1"   # Ollama OpenAI-compat endpoint
    offline_api_key: str = "ollama"                    # Ollama ignores it; the client needs non-empty
    offline_model: str = "qwen2.5:1.5b"                # ~1 GB, CPU-friendly local LLM
    # Local CPU inference is 10-50x slower than a cloud call, so the offline tier
    # gets its OWN timeout instead of sharing GEN_TIMEOUT_S (tuned for Gemini).
    # A 1.5B model on ~4 CPU cores runs ~25 tok/s prefill and ~13 tok/s decode, so a
    # ~500-token agent planner prompt alone takes ~20s — and each agent step re-sends
    # the evidence gathered so far, making later prompts several times larger. At 30s
    # every agent run died on step 1 while the same run on Gemini succeeded.
    offline_timeout_s: float = 180.0

    # Model tiering (Story 7) — the VERIFY_* tier. Generation (the creative call)
    # uses GEN_*; verification (citation Tier-2 judge + groundedness — mechanical
    # yes/no work) uses a CHEAPER, lower-temperature, hard-capped model. The seam
    # from Story 1 makes this free: build_verify_provider reads these and the
    # endpoint routes generate→llm_fn, validate+groundedness→verify_fn.
    #   VERIFY_PROVIDER unset ("") → reuse the generation provider (backward
    #   compatible: a single model still generates AND verifies, exactly as before).
    verify_provider: str = ""            # "" (reuse gen) | "stub" | "gemini" | "openai_compat"
    verify_model: str = ""               # e.g. a cheaper gemini-2.0-flash-lite
    verify_api_key: str = ""             # may share GEMINI_API_KEY or be separate
    verify_temperature: float = 0.1      # lower than gen — verification is mechanical
    verify_max_tokens: int = 64          # verification replies are short ("0.85"/"supported")

    # Answer cache (Story 7) — skip the ENTIRE chain (retrieval + generation +
    # validation + groundedness) for a repeated (user, doc, doc_version, query).
    #   CACHE_BACKEND: "none" (default — pre-Story-7 behavior, no caching) |
    #                  "memory" (in-process LRU, dev/tests) | "redis" (shared).
    #   Key is version-aware, so a re-upload (doc.version += 1) busts it
    #   automatically; TTL is only the safety net for edge cases.
    cache_backend: str = "none"          # "none" | "memory" | "redis"
    cache_ttl: int = 3600                # seconds a cached answer stays valid
    cache_max_memory_entries: int = 1000  # InMemoryAnswerCache LRU capacity

    # Token budgeting (Story 7) — cap the prompt SIZE. Over-budget prompts trim the
    # lowest-ranked expanded chunks first (Story 5's middle-truncation), applied to
    # the WHOLE prompt (system + labels + question), not just the context block.
    # Output caps are GEN_MAX_TOKENS (generation) and VERIFY_MAX_TOKENS (verify).
    gen_max_input_tokens: int = 4000

    # Generation-quality fix — thin-result retry (Part 4). When rerank returns fewer
    # than THIN_RESULT_THRESHOLD chunks for a DEFINITIONAL/INFORMATIONAL query, the
    # endpoint re-runs retrieval ONCE with a broadened query and appends the deduped
    # hits, so the model has enough material to synthesize instead of parroting or
    # abstaining. THIN_RESULT_RETRY=false disables the fallback entirely.
    thin_result_threshold: int = 3
    thin_result_retry: bool = True

    # Scope-Aware Retrieval Routing (Keebler reasoning fix). A GLOBAL query
    # (enumerate / table / classify / summarize / critique / infer) needs the WHOLE
    # document, not the top-k slice — top-k drops the chunks a complete answer
    # requires. When enabled and a single doc_id is in scope, such queries bypass
    # hybrid_search/rerank and feed the full document (retrieval.fetch_whole_document).
    # SCOPE_ROUTING_ENABLED=false restores pure top-k for every query.
    scope_routing_enabled: bool = True
    # Hard cap on how many chunks a whole-document GLOBAL fetch will feed the prompt.
    # A safety rail for very large docs (the prompt builder's GEN_MAX_INPUT_TOKENS
    # budget still trims text); 0 = no cap. Map-reduce for huge docs is a later phase.
    scope_whole_doc_max_chunks: int = 400

    # Agentic retrieval (ReAct) — an OPTIONAL layer over /generate/answer that lets
    # the model plan multi-hop retrieval before the grounded-answer step runs. It is
    # exposed on its OWN endpoint (POST /generate/agent) and ships DARK: AGENT_ENABLED
    # is only a hint the endpoint echoes — the route exists regardless — while the
    # loop's real safety comes from AGENT_MAX_STEPS + loop detection + fallback-to-RAG.
    #   AGENT_MODE          — "react" (JSON-action loop; the only mode today). Native
    #                         function-calling is a later drop-in behind the same loop.
    #   AGENT_ALLOWED_TOOLS — comma list restricting which tools may run ("" = all).
    agent_enabled: bool = False
    agent_max_steps: int = 5
    agent_mode: str = "react"
    agent_allowed_tools: str = ""

    @property
    def agent_allowed_tools_list(self) -> list[str]:
        return [t.strip() for t in self.agent_allowed_tools.split(",") if t.strip()]

    # Generation-quality fix — anti-verbatim guard (Part 5). A post-generation string
    # check (no LLM). A sentence whose longest contiguous token run against any source
    # chunk exceeds VERBATIM_THRESHOLD of its length is flagged VERBATIM. When the
    # fraction of flagged sentences exceeds VERBATIM_WARNING_RATIO, the response carries
    # quality_warning="high_verbatim". Monitoring only — the answer is never modified.
    verbatim_threshold: float = 0.8
    verbatim_warning_ratio: float = 0.5

    # Streaming (Story 6) — GLOBAL kill switch for SSE token streaming on
    # POST /generate/answer. When true (default), a request may stream (tokens as
    # they generate, then a verification patch) unless it opts out with ?stream=false.
    # When FALSE, every request is served the synchronous JSON response regardless
    # of ?stream — the emergency lever to disable streaming platform-wide without a
    # client change. (Retrieval and verification logic are identical either way.)
    stream_enabled: bool = True

    # Citation validation (Story 3) — Tier 2 (LLM judge) toggle. Tier 1 (lexical)
    # always runs and is free. VALIDATE_USE_LLM=true lets the judge adjudicate the
    # SUSPECT citations Tier 1 flags; false makes those citations "unverified"
    # (fail-open, flagged) instead of judged. Default true (prod); set false in
    # fast/offline test runs to keep them hermetic and LLM-free.
    validate_use_llm: bool = True

    # Context expansion (Story 5) — widen each retrieved chunk to its neighbors so
    # answers that span a chunk boundary have the full passage to work from.
    #   window   → neighbors per side (±N by chunk_index within the same doc).
    #   max_tokens → total budget across all expanded chunks; over-budget context
    #                is middle-truncated from the lowest-ranked chunks first (the
    #                "lost in the middle" guard). Rough len//4 token estimate.
    #   enabled  → kill switch / A-B lever. False = pre-Story-5 behavior (raw
    #              chunk text, no neighbor merge, no Chroma doc-chunk fetch).
    # NOTE: expansion needs char_start/char_end in Chroma metadata (Story 5 fix in
    # vector_store._metadata_for) — documents ingested earlier must be re-uploaded.
    context_expansion_window: int = 1
    context_max_tokens: int = 3000
    context_expansion_enabled: bool = True

    # Embedding (Phase 6 / Part 1)
    #   local  → sentence-transformers in-process (model loaded once in worker on_startup)
    #   hf_api → huggingface_hub InferenceClient (no local model)
    embed_provider: str = "local"      # "local" | "hf_api"
    embed_max_retries: int = 4         # hf_api retry budget on 429/503/network
    embed_small_max: int = 100         # <= this many chunks → SMALL tier
    embed_large_min: int = 1000        # >= this many chunks → LARGE tier
    embed_batch_size: int = 32         # chunks per embedding batch
    embed_batch_delay_s: float = 0.5   # inter-batch delay — hf_api ONLY (not local)
    # Token budget of the embedding model. all-MiniLM-L6-v2 truncates inputs over
    # 256 tokens, so the chunker splits any chunk exceeding this into token-bounded
    # sub-chunks (see app/pipeline/tokenization.py). Lower this if you swap to a
    # model with a smaller window; raise it (e.g. 512) for a longer-context model.
    embed_max_tokens: int = 256
    embed_overlap_tokens: int = 32     # token overlap when a chunk must be split

    # Google OAuth
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = "http://localhost:8001/auth/oauth/google/callback"

    # GitHub OAuth
    github_client_id: str = ""
    github_client_secret: str = ""
    github_redirect_uri: str = "http://localhost:8001/auth/oauth/github/callback"

    # Password reset
    password_reset_expire_hours: int = 1
    # In production set this to your domain. In dev the reset link is returned in API response.
    frontend_url: str = "http://localhost:8001"

    # Storage
    upload_dir: str = "./uploads"
    max_file_size_mb: int = 20

    # Redis / arq worker — the document pipeline runs in a separate worker process
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    # Pipeline execution substrate:
    #   False (default) → enqueue run_pipeline to the arq/Redis worker (production path)
    #   True            → run the pipeline INLINE in the web process via BackgroundTasks.
    # PIPELINE_SYNC=true is a DEV-ONLY escape hatch for local debugging without a
    # worker/Redis. It is NOT the production path. (Note: the upload endpoint also
    # falls back to inline automatically when no arq pool is available, e.g. in tests.)
    pipeline_sync: bool = False

    # Chunking defaults (Phase 4)
    default_chunk_strategy: str = "sentence"
    fixed_chunk_size: int = 500
    fixed_overlap: int = 50
    sentence_max_chars: int = 800
    sentence_overlap: int = 2
    paragraph_max_chars: int = 1000
    paragraph_overlap: int = 1

    # CORS
    allowed_origins: str = "http://localhost:8000"

    # Evaluation harness — GET /eval/run is admin-gated by email allowlist. Empty
    # (the default) means the endpoint is DISABLED for everyone; set a comma-separated
    # list of admin emails to enable it. The harness itself is always runnable as a
    # module (python -m app.eval.harness) and from tests regardless of this setting.
    eval_admin_emails: str = ""

    @property
    def eval_admin_emails_list(self) -> list[str]:
        return [e.strip().lower() for e in self.eval_admin_emails.split(",") if e.strip()]

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    def build_redis_settings(self):
        """
        Single source of truth for how we connect to Redis. Used by both the arq
        WorkerSettings (worker process) and the app's arq pool (web process), so the
        two can never point at different Redis instances. Imported lazily so config
        stays importable even if arq isn't installed.
        """
        from arq.connections import RedisSettings
        return RedisSettings(
            host=self.redis_host,
            port=self.redis_port,
            database=self.redis_db,
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
