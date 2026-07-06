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
    # Max ids per single Chroma upsert. Chroma caps how many records one upsert call
    # may carry; the batch-upsert sink (Part 2) splits any sink batch larger than this
    # into multiple upsert calls. Keep comfortably under Chroma's own limit.
    chroma_upsert_max: int = 4000

    # HuggingFace
    hf_api_token: str = ""
    hf_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_gen_model: str = "google/flan-t5-base"

    # Embedding (Phase 6 / Part 1)
    #   local  → sentence-transformers in-process (model loaded once in worker on_startup)
    #   hf_api → huggingface_hub InferenceClient (no local model)
    embed_provider: str = "local"      # "local" | "hf_api"
    embed_max_retries: int = 4         # hf_api retry budget on 429/503/network
    embed_small_max: int = 100         # <= this many chunks → SMALL tier
    embed_large_min: int = 1000        # >= this many chunks → LARGE tier
    embed_batch_size: int = 32         # chunks per embedding batch
    embed_batch_delay_s: float = 0.5   # inter-batch delay — hf_api ONLY (not local)

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
