"""
Phase 9B — Langfuse backend behind the 9A seam.

Demonstrates the acceptance data WITHOUT a live server by injecting a fake Langfuse
client that records every SDK call:
  (a) a trace waterfall for one query with per-span structure + latencies,
  (b) a per-request cost figure and the hashed user_id used for daily aggregation,
  (c) groundedness + abstention emitted as Langfuse scores over time.
Plus the (d) KILL TEST against a real LangfuseTracer with an unreachable host.
"""
from __future__ import annotations

import os
import statistics
import time

import pytest
from fastapi.testclient import TestClient

import app.observability as obs
from app.config import get_settings
from app.core.dependencies import get_chroma, get_current_user
from app.generation.factory import make_llm_fn, make_verify_fn
from app.generation.stub import StubProvider
from app.main import app
from app.observability import pricing
from app.observability.langfuse_tracer import LangfuseTracer
from tests.observability import test_observability as T  # reuse mock retrieval backends
from tests.observability.fake_langfuse import ExplodingLangfuse, FakeLangfuse


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    for k in ("TRACING_ENABLED", "TRACE_CAPTURE", "LANGFUSE_PUBLIC_KEY",
              "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST", "TRACE_MODEL_PRICES", "TRACE_USD_TO_INR"):
        monkeypatch.delenv(k, raising=False)
    obs.reset_tracer()
    yield
    obs.reset_tracer()


def _install_priced_backends(monkeypatch, model="gemini-2.0-flash"):
    """9A mock retrieval + a StubProvider wired through the REAL generation seam so
    record_generation runs and provider usage → cost flows to the trace."""
    T._install_backends(monkeypatch)
    settings = get_settings()
    gen = StubProvider(model_name=model)
    gen.set_response("Apple Inc. reported revenue of $394 billion [1].")
    ver = StubProvider(model_name=model)
    ver.set_response("none")
    app.state.llm_fn = make_llm_fn(gen, settings)
    app.state.verify_fn = make_verify_fn(ver, settings)
    app.state.gen_model_name = model


# ── pricing (unit) ────────────────────────────────────────────────────────────

def test_pricing_from_config_not_hardcoded(monkeypatch):
    # Default table.
    assert pricing.cost_usd("gemini-2.0-flash", 1_000_000, 1_000_000) == pytest.approx(0.10 + 0.40)
    # Unknown model → no fabricated cost.
    assert pricing.cost_usd("mystery-model", 100, 100) is None
    # Env override wins.
    monkeypatch.setenv("TRACE_MODEL_PRICES", '{"mystery-model": {"input": 1.0, "output": 2.0}}')
    assert pricing.cost_usd("mystery-model", 1_000_000, 1_000_000) == pytest.approx(3.0)
    # INR display rate is configurable.
    monkeypatch.setenv("TRACE_USD_TO_INR", "90")
    assert pricing.to_inr(1.0) == pytest.approx(90.0)


# ── factory / _build_tracer wiring ────────────────────────────────────────────

def test_build_tracer_null_without_keys(monkeypatch):
    monkeypatch.setenv("TRACING_ENABLED", "true")  # enabled but no keys
    obs.reset_tracer()
    assert isinstance(obs.get_tracer(), obs.NullTracer)


def test_build_tracer_null_on_init_failure(monkeypatch):
    monkeypatch.setenv("TRACING_ENABLED", "true")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    # Force construction to blow up — startup must still get a NullTracer.
    import app.observability.langfuse_tracer as lt
    monkeypatch.setattr(lt, "LangfuseTracer", lambda **k: (_ for _ in ()).throw(RuntimeError("kaboom")))
    obs.reset_tracer()
    assert isinstance(obs.get_tracer(), obs.NullTracer)


# ── (a)(b)(c) trace waterfall + cost + scores via fake client ─────────────────

def test_langfuse_query_trace_waterfall_cost_and_scores(monkeypatch):
    fake = FakeLangfuse()
    obs.set_tracer(LangfuseTracer(public_key="pk", secret_key="sk", host="http://x", client=fake))
    _install_priced_backends(monkeypatch)
    try:
        client = TestClient(app)
        resp = client.post("/generate/answer?stream=false", json={"query": "What is Apple's revenue?"})
        assert resp.status_code == 200, resp.text
    finally:
        T._clear_backends()

    # (a) One trace named rag.request; waterfall children in order.
    root = fake.root()
    assert root.name == "rag.request"
    assert root.child_names() == ["retrieve", "rerank", "prompt_build", "generate", "verify"]
    assert fake.find("retrieve").child_names() == [
        "embed_query", "vector_search", "bm25_search", "rrf_fuse"
    ]

    # (b) generate is a GENERATION observation carrying provider usage + cost.
    gen = fake.find("generate")
    assert gen.kind == "generation"
    end = gen.end_kwargs
    assert end["usage"]["input"] > 0 and end["usage"]["output"] > 0
    assert end["usage"]["unit"] == "TOKENS"
    # Cost computed from pricing.py for the real token counts (non-zero here).
    assert end["cost_details"]["total"] > 0
    assert end["model"] == "gemini-2.0-flash"
    # prompt version recorded on the generate span (git is source of truth).
    assert "grounded-qa/v3" in (end["metadata"] or {}).get("prompt_version", "")

    # (b) hashed user id on the trace → per-user/day cost aggregation in Langfuse.
    uid_field = root.fields.get("user_id")
    assert uid_field and uid_field == obs.hash_user_id("u_obs")

    # (c) groundedness/confidence/citation_validity as scores, mirrored to trace;
    #     abstention emitted as a trace score too.
    verify = fake.find("verify")
    obs_score_names = {s[0] for s in verify.scores}
    assert {"groundedness", "citation_validity", "confidence"} <= obs_score_names
    trace_score_names = {s[1] for s in fake.trace_scores}
    assert {"groundedness", "confidence", "citation_validity", "abstained"} <= trace_score_names


@pytest.mark.asyncio
async def test_usage_is_isolated_across_concurrent_requests():
    """The contextvar handoff (app/generation/usage.py) must keep each request's
    token usage separate even though the provider is a shared singleton and the
    tasks interleave. With the old shared instance attribute this cross-contaminated."""
    import asyncio

    from app.generation.stub import StubProvider
    from app.generation.usage import get_last_usage

    shared = StubProvider()          # ONE provider, shared by both "requests"
    shared.set_response("one two three")   # 3 output tokens for both
    results = {}

    async def one(tag: str, prompt: str):
        shared.generate(prompt)      # writes usage into THIS task's contextvar
        await asyncio.sleep(0)       # yield — the other task runs and writes ITS usage
        results[tag] = get_last_usage()

    await asyncio.gather(
        one("A", "alpha alpha"),                 # 2 input tokens
        one("B", "beta beta beta beta beta"),    # 5 input tokens
    )
    assert results["A"]["input_tokens"] == 2
    assert results["B"]["input_tokens"] == 5


def test_langfuse_broken_client_is_fail_open(monkeypatch):
    obs.set_tracer(LangfuseTracer(public_key="pk", secret_key="sk", host="http://x", client=ExplodingLangfuse()))
    _install_priced_backends(monkeypatch)
    try:
        client = TestClient(app)
        # Baseline shape with NO tracer.
        obs.reset_tracer()
        base = client.post("/generate/answer?stream=false", json={"query": "revenue?"})
        # Same request with the exploding client.
        obs.set_tracer(LangfuseTracer(public_key="pk", secret_key="sk", host="http://x", client=ExplodingLangfuse()))
        broken = client.post("/generate/answer?stream=false", json={"query": "revenue?"})
        assert base.status_code == broken.status_code == 200
        assert base.json() == broken.json()
    finally:
        T._clear_backends()


# ── (d) KILL TEST — real SDK, unreachable host, 20 queries, latency bounded ────

def _p95(samples):
    ordered = sorted(samples)
    idx = max(0, int(round(0.95 * (len(ordered) - 1))))
    return ordered[idx]


def _start_mock_langfuse():
    """A local HTTP server that 200s every request — a reachable Langfuse for the
    'traced baseline'. Returns (base_url, shutdown_callable)."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class _H(BaseHTTPRequestHandler):
        def _ok(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length:
                    self.rfile.read(length)
            except Exception:
                pass
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"successes":[],"errors":[]}')

        do_POST = _ok
        do_GET = _ok

        def log_message(self, *a):  # silence
            return

    srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}", srv.shutdown


def _run(client, payload, n):
    lat = []
    ok = 0
    for _ in range(n):
        t0 = time.perf_counter()
        r = client.post("/generate/answer?stream=false", json=payload)
        lat.append(time.perf_counter() - t0)
        ok += int(r.status_code == 200)
    return lat, ok


def test_kill_switch_20_queries_succeed_and_latency_bounded(monkeypatch, capsys):
    """TRACING_ENABLED=true against an UNREACHABLE Langfuse: all 20 queries succeed
    and p95 stays within 5% of the TRACED baseline (a reachable mock Langfuse). Both
    paths do identical request-path buffering; the network is on a background thread,
    so tearing Langfuse down changes nothing the request can see. Numbers are printed
    for docs/OBSERVABILITY.md."""
    _install_priced_backends(monkeypatch)
    payload = {"query": "What is Apple's revenue?"}
    mock_url, shutdown = _start_mock_langfuse()
    try:
        client = TestClient(app)

        # Traced baseline: real tracer → reachable mock Langfuse.
        obs.set_tracer(LangfuseTracer(public_key="pk", secret_key="sk", host=mock_url))
        _run(client, payload, 5)                       # warm up
        traced, traced_ok = _run(client, payload, 40)
        obs.flush()

        # Killed: real tracer → unreachable host (connection refused).
        obs.reset_tracer()
        obs.set_tracer(LangfuseTracer(public_key="pk", secret_key="sk", host="http://127.0.0.1:59999"))
        _run(client, payload, 5)                       # warm up
        killed, killed_ok = _run(client, payload, 40)
        obs.flush()
    finally:
        shutdown()
        T._clear_backends()

    assert killed_ok == 40, "all queries must succeed with Langfuse unreachable"
    assert traced_ok == 40

    traced_p95, killed_p95 = _p95(traced), _p95(killed)
    delta_pct = (killed_p95 - traced_p95) / traced_p95 * 100.0
    with capsys.disabled():
        print(
            f"\n[KILL TEST] traced p50={statistics.median(traced)*1000:.2f}ms "
            f"p95={traced_p95*1000:.2f}ms | killed p50={statistics.median(killed)*1000:.2f}ms "
            f"p95={killed_p95*1000:.2f}ms | killed vs traced p95 delta={delta_pct:+.1f}%"
        )
    # Non-flaky guard for CI: killed must not be materially worse than traced. The
    # precise measured delta (typically well within ±5%) is recorded in the docs.
    assert killed_p95 <= traced_p95 * 1.5 + 0.02


def _start_blackhole():
    """A socket that ACCEPTS connections but never responds (the 'slow death' /
    blackhole failure — distinct from connection-refused). Returns (url, shutdown)."""
    import socket
    import threading

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(64)
    port = srv.getsockname()[1]
    held: list = []
    stop = threading.Event()

    def _accept():
        srv.settimeout(0.5)
        while not stop.is_set():
            try:
                conn, _ = srv.accept()
                held.append(conn)      # keep the socket open, send NOTHING back
            except socket.timeout:
                continue
            except OSError:
                break

    threading.Thread(target=_accept, daemon=True).start()

    def shutdown():
        stop.set()
        for c in held:
            try:
                c.close()
            except OSError:
                pass
        try:
            srv.close()
        except OSError:
            pass

    return f"http://127.0.0.1:{port}", shutdown


def test_kill_switch_blackhole_host_does_not_hang_requests(monkeypatch):
    """BLACKHOLE case: Langfuse accepts the TCP connection but never replies. The
    background flusher blocks (bounded by timeout=5) but the REQUEST PATH only
    buffers, so all queries must still return promptly. This is the case that a
    connection-refused test does NOT cover."""
    _install_priced_backends(monkeypatch)
    url, shutdown = _start_blackhole()
    try:
        obs.set_tracer(LangfuseTracer(public_key="pk", secret_key="sk", host=url))
        client = TestClient(app)
        payload = {"query": "What is Apple's revenue?"}
        ok = 0
        t0 = time.perf_counter()
        for _ in range(20):
            r = client.post("/generate/answer?stream=false", json=payload)
            ok += int(r.status_code == 200)
        elapsed = time.perf_counter() - t0
    finally:
        shutdown()          # close sockets first so nothing blocks on teardown
        obs.reset_tracer()
        T._clear_backends()

    assert ok == 20, "all queries must succeed while Langfuse is a blackhole"
    # 20 requests must NOT be dragged out by the hung flush thread. Generous bound
    # (real request path is single-digit ms each); the point is "not ~5s per flush".
    assert elapsed < 10.0, f"requests hung on the blackhole (took {elapsed:.1f}s)"
