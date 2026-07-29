"""Load test for the retrieval and generation paths.

Answers the question a system-design interview actually asks: what happens to p95
under concurrency, and which component saturates first?

Deliberately weights the mix toward RETRIEVAL rather than generation. Generation
latency is dominated by an external LLM's queue, so hammering it measures the
provider's rate limit rather than this system; retrieval and document listing are
where *our* database, vector index and connection pool are exercised.

Setup (once):
    pip install locust
    python scripts/seed_load_test.py          # creates the load-test user + corpus

Run headless, three concurrency levels:
    locust -f scripts/locustfile.py --headless -u 50  -r 10 -t 2m --host http://localhost:8000
    locust -f scripts/locustfile.py --headless -u 100 -r 20 -t 2m --host http://localhost:8000
    locust -f scripts/locustfile.py --headless -u 200 -r 40 -t 2m --host http://localhost:8000

Web UI:
    locust -f scripts/locustfile.py --host http://localhost:8000
"""
from __future__ import annotations

import os
import random

from locust import HttpUser, between, events, task

EMAIL = os.getenv("LOADTEST_EMAIL", "loadtest@example.com")
PASSWORD = os.getenv("LOADTEST_PASSWORD", "loadtest-password-123")

QUERIES = [
    "which replacement battery does the GX-4200 use",
    "what firmware version is recommended for the HX-9000",
    "how much can someone growing apples earn from an acre",
    "what was Apple's total annual revenue",
    "which solar inverter comes with the longer warranty",
    "88-AZ-0097",
    "E-4521",
    "what is the default admin password on the GX-4200",
    "which oranges sell for the most",
    "how many trees should I plant per acre in a modern apple orchard",
]


@events.quitting.add_listener
def _assert_healthy(environment, **_kw):
    """Fail the run on a bad result, so this is usable as a CI gate rather than a
    number someone has to eyeball."""
    stats = environment.stats.total
    if stats.num_requests == 0:
        environment.process_exit_code = 1
        return
    fail_ratio = stats.fail_ratio
    p95 = stats.get_response_time_percentile(0.95)
    if fail_ratio > 0.01:
        print(f"FAIL: error rate {fail_ratio:.2%} > 1%")
        environment.process_exit_code = 1
    elif p95 > 5000:
        print(f"FAIL: p95 {p95:.0f}ms > 5000ms")
        environment.process_exit_code = 1
    else:
        print(f"PASS: {stats.num_requests} reqs, {fail_ratio:.2%} errors, p95 {p95:.0f}ms")


class DocuChunkUser(HttpUser):
    # Think time: a real user reads an answer before asking again. Zero wait would
    # measure how fast we can be DoS'd, not how the system behaves under load.
    wait_time = between(1, 3)

    def on_start(self):
        """Authenticate once per simulated user."""
        resp = self.client.post(
            "/auth/login",
            json={"email": EMAIL, "password": PASSWORD},
            name="/auth/login",
            catch_response=True,
        )
        self.token = None
        if resp.status_code == 200:
            try:
                self.token = resp.json().get("access_token")
                resp.success()
            except Exception:
                resp.failure("login returned no token")
        else:
            resp.failure(f"login failed: {resp.status_code}")

    @property
    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    @task(10)
    def semantic_search(self):
        """The hot path: embed the query, hit the vector index + BM25, fuse, rerank."""
        self.client.post(
            "/search/semantic",
            json={"query": random.choice(QUERIES), "top_k": 5},
            headers=self.headers,
            name="POST /search/semantic",
        )

    @task(4)
    def list_documents(self):
        """Pure relational read — isolates Postgres from the vector index."""
        self.client.get("/docs/list", headers=self.headers, name="GET /docs/list")

    @task(1)
    def health(self):
        self.client.get("/health", name="GET /health")
