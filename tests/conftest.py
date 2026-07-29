"""
Test-suite isolation from the developer's local environment.

The app reads its settings from `.env` via pydantic-settings, and
`app/database.py` builds its engine at MODULE IMPORT time. That means whatever
`DATABASE_URL` happens to be in a developer's `.env` decides what the test suite
connects to — and once that file points at the Docker-network host `postgres`
(which does not resolve outside Compose), collection fails before a single test
runs.

Tests must not depend on local configuration. This pins a hermetic environment
BEFORE any `app.*` module is imported:

  * DATABASE_URL   -> a throwaway SQLite file
  * VECTOR_BACKEND -> chroma (embedded; needs no server)
  * CHROMA_MODE    -> persistent, in a temp directory

Anything a test needs differently, it overrides explicitly — which is the point:
the dependency becomes visible instead of ambient.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Applied at import, before pytest collects any module that touches `app.*`.
_TMP = Path(tempfile.mkdtemp(prefix="docuchunk_tests_"))

os.environ["DATABASE_URL"] = f"sqlite:///{_TMP / 'test.db'}"
os.environ["VECTOR_BACKEND"] = "chroma"
os.environ["CHROMA_MODE"] = "persistent"
os.environ["CHROMA_PERSIST_DIR"] = str(_TMP / "chroma")
# Never let a real key be picked up from .env and spend quota during a test run.
os.environ.setdefault("GEN_PROVIDER", "stub")


import pytest


@pytest.fixture(scope="session", autouse=True)
def _create_schema():
    """Create the ORM schema on the throwaway database.

    Some endpoint tests query through the default session rather than injecting
    their own, and previously happened to work because a developer's `app.db` was
    already sitting in the repo root with tables in it. That is an accidental
    dependency on local state: the same tests fail on a clean checkout or in CI.
    Creating the schema here makes the suite self-contained.
    """
    from app.database import Base, engine
    import app.models  # noqa: F401 — registers every model on Base

    Base.metadata.create_all(bind=engine)
    yield
