"""
Chroma client-mode + collection-space tests.

Covers the config-driven client construction (app/core/chroma.py):
  - persistent mode builds a working local client against a temp dir (no server)
  - an unknown CHROMA_MODE fails loudly
  - collections are created with cosine space, NOT Chroma's default L2
  - http mode is documented via a SKIPPED-by-default integration test
"""
import os

import chromadb
import pytest

from app.core.chroma import build_chroma_client, get_or_create_collection


class _FakeSettings:
    """Stand-in for the pydantic Settings object build_chroma_client reads."""
    def __init__(self, **kw):
        self.chroma_mode = kw.get("chroma_mode", "persistent")
        self.chroma_persist_dir = kw.get("chroma_persist_dir", "./chroma_db")
        self.chroma_host = kw.get("chroma_host", "localhost")
        self.chroma_port = kw.get("chroma_port", 8000)


def test_persistent_mode_builds_working_client(tmp_path, monkeypatch):
    """persistent mode → a real local client, no external server required."""
    monkeypatch.setattr(
        "app.core.chroma.get_settings",
        lambda: _FakeSettings(chroma_mode="persistent", chroma_persist_dir=str(tmp_path)),
    )
    client = build_chroma_client()
    assert client.heartbeat() > 0
    assert client.list_collections() == []


def test_persistent_mode_is_case_insensitive(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.core.chroma.get_settings",
        lambda: _FakeSettings(chroma_mode="Persistent", chroma_persist_dir=str(tmp_path)),
    )
    assert build_chroma_client().heartbeat() > 0


def test_invalid_mode_raises(monkeypatch):
    monkeypatch.setattr(
        "app.core.chroma.get_settings",
        lambda: _FakeSettings(chroma_mode="bogus"),
    )
    with pytest.raises(ValueError, match="CHROMA_MODE"):
        build_chroma_client()


def test_collection_created_with_cosine_space(tmp_path):
    """
    The whole point for the embedder work: collections must use cosine space so the
    L2-normalized vectors Part 1 produces are ranked by cosine == dot product, not L2.
    """
    client = chromadb.PersistentClient(path=str(tmp_path))
    coll = get_or_create_collection(client, "user_alice_docs")
    assert coll.metadata is not None
    assert coll.metadata.get("hnsw:space") == "cosine", (
        f"Collection must be created with cosine space, not the default L2. "
        f"Got metadata: {coll.metadata}"
    )


def test_get_or_create_collection_is_idempotent(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path))
    a = get_or_create_collection(client, "user_bob_docs")
    b = get_or_create_collection(client, "user_bob_docs")
    assert a.name == b.name
    assert len(client.list_collections()) == 1


@pytest.mark.skipif(
    not os.getenv("CHROMA_HOST"),
    reason="http-mode integration test — set CHROMA_HOST (and optionally CHROMA_PORT) "
           "to run against a live Chroma server. Skipped in CI by default.",
)
def test_http_mode_against_live_server(monkeypatch):
    """Documents the deployed http path without breaking CI (skipped unless CHROMA_HOST set)."""
    monkeypatch.setattr(
        "app.core.chroma.get_settings",
        lambda: _FakeSettings(
            chroma_mode="http",
            chroma_host=os.getenv("CHROMA_HOST"),
            chroma_port=int(os.getenv("CHROMA_PORT", "8000")),
        ),
    )
    client = build_chroma_client()
    assert client.heartbeat() > 0
