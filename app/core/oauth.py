from authlib.integrations.starlette_client import OAuth
from app.config import get_settings

settings = get_settings()

oauth = OAuth()

# ── Google ────────────────────────────────────────────────────────────────────
oauth.register(
    name="google",
    client_id=settings.google_client_id,
    client_secret=settings.google_client_secret,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={
        "scope": "openid email profile",
        "prompt": "select_account",
    },
)

# ── GitHub ────────────────────────────────────────────────────────────────────
# GitHub doesn't have an OIDC discovery URL — we specify endpoints manually.
oauth.register(
    name="github",
    client_id=settings.github_client_id,
    client_secret=settings.github_client_secret,
    access_token_url="https://github.com/login/oauth/access_token",
    access_token_params=None,
    authorize_url="https://github.com/login/oauth/authorize",
    authorize_params=None,
    api_base_url="https://api.github.com/",
    client_kwargs={
        "scope": "user:email",
        "token_endpoint_auth_method": "client_secret_post",
    },
)
