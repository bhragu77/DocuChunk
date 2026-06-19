import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_structure():
    response = client.get("/health")
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "environment" in data
    assert "checks" in data


def test_health_all_checks_ok():
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "healthy"
    for key, val in data["checks"].items():
        assert val == "ok", f"Check '{key}' failed: {val}"


def test_docs_available():
    assert client.get("/api/docs").status_code == 200
    assert client.get("/api/redoc").status_code == 200
