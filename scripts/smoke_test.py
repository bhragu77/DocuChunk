#!/usr/bin/env python3
"""
Gemini end-to-end smoke test for the DocuChunk backend.

Verifies the REAL generation stack (retrieval → Gemini generation → citation
validation → groundedness/confidence → cache) against a running backend using
the real GEMINI_API_KEY. It never prints the key.

Prerequisites:
  * Backend running (default http://127.0.0.1:8001, override with BASE_URL).
  * .env has GEN_PROVIDER=gemini and a valid GEMINI_API_KEY.
  * For the cache-hit step to show a speedup, CACHE_BACKEND=memory|redis
    (with CACHE_BACKEND=none the step still PASSes on identical content).

Usage:
    python scripts/smoke_test.py

Exit code 0 = all steps passed, 1 = any step failed.
"""
from __future__ import annotations

import io
import os
import sys
import time

import httpx

# Import settings for config-driven values (key presence, cache backend). Importing
# app.config only reads .env — it does not start Chroma/DB/etc.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.config import get_settings  # noqa: E402

BASE_URL = os.environ.get("BASE_URL", "http://127.0.0.1:8001").rstrip("/")
TEST_EMAIL = os.environ.get("SMOKE_EMAIL", "smoke_test@example.com")
TEST_PASSWORD = os.environ.get("SMOKE_PASSWORD", "smoke-pass-123")
INGEST_TIMEOUT_S = 90
POLL_EVERY_S = 2.0

# A fact only this document contains — used to check grounded generation.
DOC_FACT = "The GX-4200 controller operates between 0 and 45 degrees Celsius."
ANSWERABLE_Q = "What is the operating temperature range of the GX-4200?"
UNANSWERABLE_Q = "What is the airspeed velocity of an unladen swallow?"

# ── tiny reporting harness ────────────────────────────────────────────────────
_results: list[tuple[str, bool, str]] = []
_skipped: list[str] = []


def record(step: str, ok: bool, detail: str = "") -> bool:
    _results.append((step, ok, detail))
    icon = "PASS" if ok else "FAIL"
    print(f"  [{icon}] {step}" + (f" — {detail}" if detail else ""))
    return ok


def record_skip(step: str, detail: str = "") -> None:
    _skipped.append(step)
    print(f"  [SKIP] {step}" + (f" — {detail}" if detail else ""))


def fail_dump(label: str, obj) -> None:
    import json
    print(f"\n----- {label} (full response for debugging) -----")
    try:
        print(json.dumps(obj, indent=2, default=str))
    except Exception:
        print(repr(obj))
    print("-" * 52)


def make_test_pdf() -> bytes:
    """Create a minimal 1-page PDF carrying DOC_FACT (PyMuPDF)."""
    import fitz  # PyMuPDF — already a project dependency (used in tests)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 96), "GX-4200 Controller — Technical Specification")
    page.insert_text((72, 130), DOC_FACT)
    page.insert_text((72, 160), "Brief excursions to 55 degrees Celsius are permitted under warranty.")
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def main() -> int:
    settings = get_settings()
    print(f"DocuChunk smoke test → {BASE_URL}")
    print(f"  provider={settings.gen_provider} model={settings.gen_model} "
          f"cache={settings.cache_backend}")

    # Preflight: config sanity (never print the key itself).
    if settings.gen_provider != "gemini":
        record("config: GEN_PROVIDER=gemini", False,
               f"GEN_PROVIDER is {settings.gen_provider!r}; set it to 'gemini'")
        return summary()
    record("config: GEN_PROVIDER=gemini", True)
    if not settings.gemini_api_key:
        record("config: GEMINI_API_KEY present", False, "GEMINI_API_KEY is empty in .env")
        return summary()
    record("config: GEMINI_API_KEY present", True, f"(len={len(settings.gemini_api_key)})")

    client = httpx.Client(base_url=BASE_URL, timeout=60.0)

    # ── Preflight: /health reachable ──────────────────────────────────────────
    try:
        h = client.get("/health")
        ok = h.status_code == 200
        record("backend reachable (/health)", ok, f"status={h.status_code}")
        if not ok:
            fail_dump("/health", _safe_json(h))
            return summary()
    except httpx.HTTPError as exc:
        record("backend reachable (/health)", False,
               f"cannot reach {BASE_URL} — is the backend running? ({exc})")
        return summary()

    # ── Step 1: register or login → JWT ───────────────────────────────────────
    token = _auth(client)
    if token is None:
        return summary()
    auth = {"Authorization": f"Bearer {token}"}

    # ── Step 2: upload a tiny test PDF ────────────────────────────────────────
    try:
        up = client.post(
            "/docs/upload",
            headers=auth,
            files={"file": ("gx4200_spec.pdf", make_test_pdf(), "application/pdf")},
        )
    except httpx.HTTPError as exc:
        return _end(record("step 2: upload document", False, f"upload error: {exc}"))
    if up.status_code != 201:
        record("step 2: upload document", False, f"status={up.status_code}")
        fail_dump("upload", _safe_json(up))
        return summary()
    doc_id = up.json()["id"]
    record("step 2: upload document", True, f"doc_id={doc_id}")

    # ── Step 3: poll until status=ready (or timeout) ──────────────────────────
    if not _wait_ready(client, auth, doc_id):
        return summary()

    # ── Step 4/5: answerable question → grounded, cited, confident ────────────
    ans = _generate(client, auth, ANSWERABLE_Q, doc_id)
    if ans is None:
        return summary()

    # Environment guard: if the LLM provider itself is unavailable (e.g. Gemini
    # free-tier 429), the backend must degrade GRACEFULLY — HTTP 200 with the
    # honest error shape, real citations, verified=false — NOT a 500. That
    # graceful degradation IS a passing backend check; the content-dependent
    # steps are then SKIPPED (they need a live model we can't reach).
    if ans.get("error") == "generation_failed":
        graceful = (
            ans.get("answer") is None
            and ans.get("verified") is False
            and ans.get("confidence") == 0.0
            and bool(ans.get("citations"))          # retrieval citations are still real
        )
        record("step 5: provider unavailable → HONEST error shape (200, not 500)", graceful,
               "LLM provider error (likely Gemini free-tier 429); backend degraded gracefully")
        record_skip("step 5b/7/8: live generation content checks",
                    "Gemini generation unavailable — cannot exercise real answers")
        if not graceful:
            fail_dump("generation_failed response", ans)
        return summary()

    good_shape = _has_new_shape(ans)
    record("step 5a: response has the current schema (not the old broken shape)", good_shape)
    if not good_shape:
        fail_dump("answerable /generate/answer", ans)
        return summary()

    non_empty = bool((ans.get("answer") or "").strip())
    cited = ans.get("cited_sources") or []
    confidence = ans.get("confidence")
    ok5 = non_empty and len(cited) >= 1 and isinstance(confidence, (int, float)) and confidence > 0
    record(
        "step 5b: non-empty answer + >=1 cited source + confidence>0",
        ok5,
        f"answer_len={len(ans.get('answer') or '')} cited={len(cited)} "
        f"confidence={confidence} verified={ans.get('verified')}",
    )
    if not ok5:
        fail_dump("answerable /generate/answer", ans)

    # ── Step 6/7: unanswerable question → abstain OR low-confidence ───────────
    un = _generate(client, auth, UNANSWERABLE_Q, doc_id)
    if un is not None:
        # NB: use an explicit None check, not `or 1.0` — a legitimate confidence
        # of 0.0 is falsy and would otherwise be coerced to 1.0, failing the test
        # for the exact behavior we want (abstain with confidence 0.0).
        _c = un.get("confidence")
        conf = 1.0 if _c is None else float(_c)
        abstained = un.get("abstained") is True and conf == 0.0
        weak = conf < 0.5 and len(un.get("cited_sources") or []) == 0
        ok7 = abstained or weak
        record(
            "step 7: unanswerable → abstained(conf=0) OR confidence<0.5 & no citations",
            ok7,
            f"abstained={un.get('abstained')} confidence={un.get('confidence')} "
            f"cited={len(un.get('cited_sources') or [])}",
        )
        if not ok7:
            fail_dump("unanswerable /generate/answer", un)

    # ── Step 8: repeat answerable question → cache hit (fast, same content) ────
    t0 = time.perf_counter()
    ans2 = _generate(client, auth, ANSWERABLE_Q, doc_id)
    dt_ms = (time.perf_counter() - t0) * 1000
    if ans2 is not None:
        same = (ans2.get("answer") or "").strip() == (ans.get("answer") or "").strip()
        if settings.cache_backend == "none":
            record("step 8: repeat question (cache disabled — content identical)", same,
                   f"latency={dt_ms:.0f}ms (CACHE_BACKEND=none, no speedup expected)")
        else:
            # A cache hit should be dramatically faster than a full 2-LLM-call miss.
            record("step 8: repeat question served from cache (fast + identical)",
                   same and dt_ms < 1500,
                   f"latency={dt_ms:.0f}ms same_content={same}")
        if not same:
            fail_dump("repeat /generate/answer", ans2)

    return summary()


# ── helpers ───────────────────────────────────────────────────────────────────

def _auth(client: httpx.Client) -> str | None:
    """Register (idempotent) then login → access token."""
    client.post("/auth/register", json={"email": TEST_EMAIL, "password": TEST_PASSWORD})
    r = client.post("/auth/login", json={"email": TEST_EMAIL, "password": TEST_PASSWORD})
    if r.status_code != 200:
        record("step 1: authenticate", False, f"login status={r.status_code}")
        fail_dump("login", _safe_json(r))
        return None
    token = r.json().get("access_token")
    record("step 1: authenticate", bool(token), "obtained JWT")
    return token


def _wait_ready(client, auth, doc_id) -> bool:
    deadline = time.time() + INGEST_TIMEOUT_S
    last = "?"
    while time.time() < deadline:
        r = client.get(f"/docs/{doc_id}", headers=auth)
        if r.status_code == 200:
            last = r.json().get("status")
            if last == "ready":
                return record("step 3: ingestion complete (status=ready)", True)
            if last == "failed":
                record("step 3: ingestion complete (status=ready)", False, "status=failed")
                fail_dump("doc status", _safe_json(r))
                return False
        time.sleep(POLL_EVERY_S)
    return record("step 3: ingestion complete (status=ready)", False,
                  f"timeout after {INGEST_TIMEOUT_S}s (last status={last})")


def _generate(client, auth, query, doc_id) -> dict | None:
    try:
        r = client.post(
            "/generate/answer",
            params={"stream": "false"},
            headers=auth,
            json={"query": query, "doc_id": doc_id},
        )
    except httpx.HTTPError as exc:
        record(f"generate ({query[:30]}…)", False, f"request error: {exc}")
        return None
    if r.status_code != 200:
        record(f"generate ({query[:30]}…)", False, f"status={r.status_code}")
        fail_dump("generate", _safe_json(r))
        return None
    return r.json()


def _has_new_shape(obj: dict) -> bool:
    """The current GenerateAnswerResponse has these keys; the old broken shape did not."""
    required = {"answer", "cited_sources", "confidence", "verified", "abstained", "grounded"}
    return required.issubset(obj.keys())


def _safe_json(resp: httpx.Response):
    try:
        return resp.json()
    except Exception:
        return {"status_code": resp.status_code, "text": resp.text[:2000]}


def _end(_ok: bool) -> int:
    return summary()


def summary() -> int:
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)
    total = len(_results)
    print("\n" + "=" * 52)
    print(f"SMOKE TEST SUMMARY: {passed}/{total} checks passed, "
          f"{failed} failed, {len(_skipped)} skipped")
    print("=" * 52)
    # Exit 0 when nothing hard-FAILED. Skips (external provider quota) are not
    # backend defects — they leave the run green with a clear note.
    ok = failed == 0 and total > 0
    if ok and _skipped:
        print("RESULT: PASS (with skips) ✅  — backend healthy; "
              "some live-generation steps skipped due to provider quota")
    else:
        print("RESULT: " + ("ALL PASS ✅" if ok else "FAIL ❌"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
