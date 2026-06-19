import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import app
from app.database import Base, get_db

# ── In-memory SQLite DB per test session ─────────────────────────────────────
TEST_DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSession()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True, scope="module")
def setup_db():
    """Create tables once for the module, drop them after."""
    from app.models import user, document, job  # noqa
    Base.metadata.create_all(bind=engine)
    app.dependency_overrides[get_db] = override_get_db
    yield
    Base.metadata.drop_all(bind=engine)
    app.dependency_overrides.clear()


client = TestClient(app)

# ── Helpers ───────────────────────────────────────────────────────────────────

def register_user(email="test@example.com", password="password123", full_name="Test User"):
    return client.post("/auth/register", json={"email": email, "password": password, "full_name": full_name})

def login_user(email="test@example.com", password="password123"):
    return client.post("/auth/login", json={"email": email, "password": password})


# ── Register ──────────────────────────────────────────────────────────────────

def test_register_success():
    res = register_user()
    assert res.status_code == 201
    body = res.json()
    assert "access_token" in body
    assert "refresh_token" in body
    assert body["token_type"] == "bearer"


def test_register_duplicate_email():
    register_user(email="dup@example.com")
    res = register_user(email="dup@example.com")
    assert res.status_code == 409
    assert "already exists" in res.json()["detail"].lower()


def test_register_weak_password():
    res = client.post("/auth/register", json={"email": "weak@example.com", "password": "short"})
    assert res.status_code == 422  # Pydantic validation error


def test_register_invalid_email():
    res = client.post("/auth/register", json={"email": "not-an-email", "password": "password123"})
    assert res.status_code == 422


# ── Login ─────────────────────────────────────────────────────────────────────

def test_login_success():
    register_user(email="login@example.com")
    res = login_user(email="login@example.com")
    assert res.status_code == 200
    body = res.json()
    assert "access_token" in body
    assert "refresh_token" in body


def test_login_wrong_password():
    register_user(email="wrongpass@example.com")
    res = client.post("/auth/login", json={"email": "wrongpass@example.com", "password": "wrongpassword"})
    assert res.status_code == 401
    assert "Invalid" in res.json()["detail"]


def test_login_unknown_email():
    res = client.post("/auth/login", json={"email": "nobody@example.com", "password": "password123"})
    assert res.status_code == 401
    # Same error message — no user enumeration
    assert "Invalid" in res.json()["detail"]


# ── /auth/me ─────────────────────────────────────────────────────────────────

def test_get_me_authenticated():
    register_user(email="me@example.com", full_name="Me User")
    tokens = login_user(email="me@example.com").json()
    res = client.get("/auth/me", headers={"Authorization": f"Bearer {tokens['access_token']}"})
    assert res.status_code == 200
    body = res.json()
    assert body["email"] == "me@example.com"
    assert body["full_name"] == "Me User"
    assert "id" in body


def test_get_me_unauthenticated():
    res = client.get("/auth/me")
    assert res.status_code == 401


def test_get_me_invalid_token():
    res = client.get("/auth/me", headers={"Authorization": "Bearer this.is.fake"})
    assert res.status_code == 401


# ── Refresh token ─────────────────────────────────────────────────────────────

def test_refresh_token_success():
    register_user(email="refresh@example.com")
    tokens = login_user(email="refresh@example.com").json()
    res = client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert res.status_code == 200
    new_tokens = res.json()
    assert "access_token" in new_tokens
    assert "refresh_token" in new_tokens
    # Verify the new access token actually works for protected endpoints
    me_res = client.get("/auth/me", headers={"Authorization": f"Bearer {new_tokens['access_token']}"})
    assert me_res.status_code == 200
    assert me_res.json()["email"] == "refresh@example.com"


def test_refresh_with_access_token_fails():
    register_user(email="badrefresh@example.com")
    tokens = login_user(email="badrefresh@example.com").json()
    # Passing access token as refresh token should fail
    res = client.post("/auth/refresh", json={"refresh_token": tokens["access_token"]})
    assert res.status_code == 401


def test_refresh_with_garbage_fails():
    res = client.post("/auth/refresh", json={"refresh_token": "not.a.real.token"})
    assert res.status_code == 401


# ── Token flow end-to-end ─────────────────────────────────────────────────────

def test_full_auth_flow():
    """Register → Login → access /me → refresh → access /me again with new token."""
    # Register
    reg = register_user(email="flow@example.com", full_name="Flow User")
    assert reg.status_code == 201

    # Login
    login_res = login_user(email="flow@example.com")
    tokens = login_res.json()

    # Access protected endpoint
    me_res = client.get("/auth/me", headers={"Authorization": f"Bearer {tokens['access_token']}"})
    assert me_res.status_code == 200
    assert me_res.json()["email"] == "flow@example.com"

    # Refresh
    refresh_res = client.post("/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert refresh_res.status_code == 200
    new_tokens = refresh_res.json()

    # Access protected endpoint with new token
    me_res2 = client.get("/auth/me", headers={"Authorization": f"Bearer {new_tokens['access_token']}"})
    assert me_res2.status_code == 200
    assert me_res2.json()["email"] == "flow@example.com"
