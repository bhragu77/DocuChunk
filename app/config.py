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

    # ChromaDB
    chroma_persist_dir: str = "./chroma_db"

    # HuggingFace
    hf_api_token: str = ""
    hf_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_gen_model: str = "google/flan-t5-base"

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

    # CORS
    allowed_origins: str = "http://localhost:8000"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",")]

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
