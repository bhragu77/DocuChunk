# DocuChunk — Capability Reference

> **What this document is for.** Every feature in this system, what engineering
> decision sits behind it, the hiring requirement it proves, and the measured
> evidence that it works. Written so that any claim made on a résumé or in an
> interview can be traced to a file or a committed artifact in this repository.
>
> **Scale of the codebase:** 14,356 LOC application · 9,383 LOC tests ·
> 421 test functions across 34 files · 21 distinct trace span types ·
> 42-document / 45-query evaluation fixture.
>
> **Rule applied throughout:** if a number appears here, a harness in this repo
> produced it and the artifact is committed. Nothing is asserted.

---

## 0. What the system actually does

Upload a PDF or DOCX → it is parsed (with OCR fallback for scanned pages), cleaned,
chunked with page-accurate offsets, embedded, and indexed into **both** a vector
store and a BM25 lexical index. A question then goes through:

```
query
  ├─ classification (query type + answer task)
  ├─ retrieval:  dense ─┐
  │               BM25 ─┴─ RRF fusion → cross-encoder rerank
  ├─ agent (optional): plans multi-hop retrieval, decides when it has enough
  ├─ generation: grounded prompt with numbered sources → answer with [n] citations
  └─ verification: citation validation + groundedness check → confidence + abstention
```

Every stage of that is traced, cost-metered, and covered by an evaluation harness
with a CI regression gate.

---

## 1. Document ingestion

| Feature | Where | Decision behind it |
|---|---|---|
| PDF / DOCX parsing | `app/pipeline/parser.py` | PyMuPDF for text layers, **Tesseract OCR fallback** per-page when a page has no text layer — scanned PDFs are the common real-world failure and silently returning empty text is worse than being slow |
| Cleaning | `app/pipeline/cleaner.py` | Normalises whitespace/hyphenation before chunking, so chunk boundaries aren't decided by PDF layout artifacts |
| Chunking with offsets | `app/pipeline/chunker.py` | Sentence-aware splitting that **preserves char offsets and page spans**, so every chunk can cite an exact page. Content-hash chunk IDs make re-ingestion idempotent |
| Token budgeting | `app/pipeline/tokenization.py` | tiktoken-based, so chunk sizes are measured in model tokens rather than characters |
| Async ingestion | `app/workers/tasks.py`, arq + Redis | Upload returns immediately; parse→chunk→embed runs in a worker. **Falls back to inline ingestion when Redis is absent**, so local dev needs no broker |

**Proves:** unstructured data handling, semantic chunking, async job processing.

---

## 2. Retrieval — the measured core

| Feature | Where |
|---|---|
| Dense vector search (Chroma) | `app/pipeline/vector_store.py`, `app/core/chroma.py` |
| BM25 lexical index, persisted per user | `app/pipeline/bm25_index.py` |
| Reciprocal Rank Fusion | `app/pipeline/retrieval.py` |
| Cross-encoder reranking | `app/pipeline/retrieval.py` (sentence-transformers) |
| Context expansion (neighbour chunks) | `app/generation/context_expander.py` |
| Scope-aware routing | `app/generation/query_classifier.py` |
| Pluggable embedding providers | `app/pipeline/embedding_providers.py` |

### Measured — three modes head-to-head, same ingested store, same ground truth

| metric | dense-only | hybrid | hybrid + rerank |
|---|:--:|:--:|:--:|
| MRR | 0.777 | 0.883 | **0.967** |
| recall@5 | 0.867 | 1.000 | 0.978 |

Artifact: [`eval/phase8_comparison.md`](../eval/phase8_comparison.md)

**The honest finding this produced** — and the one worth telling in an interview:
hybrid retrieval *recovered 6 queries* dense-only missed entirely (bare alphanumeric
identifiers like `88-AZ-0097`, which a neural embedding blurs), but it also
**regressed 2 semantic queries** by injecting a lexical distractor. The cross-encoder
rerank repaired one of them. So the pipeline isn't "hybrid is better" — it's *hybrid
wins on identifiers, costs you on semantic queries, and rerank pays for the damage*.

**Proves:** hybrid search, RRF, reranking, embeddings, vector DBs, retrieval evaluation.

---

## 3. Agentic retrieval (ReAct)

`app/generation/agent/` — 775 LOC: `loop.py`, `react.py`, `tools.py`, `state.py`

An explicit while-loop — LangGraph's router/tool/LLM nodes written out plainly, with
no framework dependency — driven as a **generator**, so the streaming and
non-streaming request paths consume identical steps.

**Tools:** `retrieve_docs` (focused sub-query search) · `fetch_document` (whole-document
reads for summarise/enumerate tasks that top-k structurally truncates).

**Five guardrails**, each recorded as an error step rather than a crash:

1. `MAX_STEPS` cap
2. allowed-tool check — a hallucinated tool name is recorded, not raised
3. required-argument validation
4. loop detection on a repeated `(tool, args)` signature + a no-progress guard
5. **fallback-to-RAG** — if the loop ends with no evidence, run one plain retrieval so
   the agent is *never worse* than single-shot RAG

A planner LLM failure (local-model timeout, cloud 504, rate limit) is caught **inside**
the loop. Letting it escape the generator skipped the fallback and returned zero
evidence — strictly worse than plain RAG, the one thing the loop promises never to be.

### Measured — and this is the differentiated result

| k (retrieval budget) | Δ context-recall, multi-hop | Δ context-precision |
|:--:|:--:|:--:|
| 1 | **+0.431** | −0.014 |
| 2 | **+0.180** | −0.065 |
| 3 | +0.083 | −0.092 |
| 5 | +0.000 | −0.095 |

Artifact: [`eval/AGENT_BASELINE.md`](../eval/AGENT_BASELINE.md)

**Read it as:** planned retrieval buys coverage, and coverage is only worth its extra
LLM calls when the budget is scarce. At k=5 on this corpus one pass already finds
everything, so the agent costs precision for nothing. **The crossover is the finding** —
it tells you *when* to route a query to the agent instead of RAG.

Getting there required fixing the measurement first: the original fixture had 26
chunks, so a k=5 retrieval returned ~19% of the entire corpus in a single pass and
**no strategy could lose**. 32 schema-matched distractor documents were added to make
the retrieval budget actually bind.

**Proves:** agentic AI, tool use, multi-step reasoning, state management, and — rarer —
the ability to design an experiment that can actually falsify your own feature.

---

## 4. Grounded generation

| Feature | Where | Decision |
|---|---|---|
| Provider abstraction | `app/generation/factory.py`, `base.py` | One construction site; `stub` / `gemini` / `openai_compat` behind a protocol, so tests never touch a network |
| Per-request model picker | `app/generation/factory.py` | Chat can pick "gemini" (cloud) vs "offline" (local Ollama). Tiers are built **independently** — a missing key hides one, not both — and the offline tier is only offered if Ollama answers a startup probe |
| Model tiering | `VERIFY_*` settings | Generation uses a strong model; verification (mechanical yes/no work) uses a cheaper, hard-capped, low-temperature one |
| Numbered-source prompting | `app/generation/prompt_builder.py` | Sources rendered `[n] (file, p.N)` so citations are checkable |
| Citation parsing + validation | `citation_parser.py`, `citation_validator.py` | Two-tier: structural parse, then an LLM judge for borderline claims. Invalid markers are **dropped and surfaced**, not hidden |
| Abstention | `quality_guard.py` | The system refuses when evidence is thin, rather than producing a confident wrong answer |
| Streaming (SSE) | `streaming.py` | Token events then one verification event |
| Answer caching | `cache.py` | memory / Redis / off |

**Proves:** prompt engineering, structured output, LLM API integration, streaming, caching.

---

## 5. LLMOps / observability

`app/observability/` — 904 LOC

| Component | Purpose |
|---|---|
| `base.py` | Tracer protocol + `NullTracer` — instrumentation is free when tracing is off |
| `context.py` | Contextvar span stack, trace-id propagation, **user-id hashing** so no raw IDs reach a third party |
| `langfuse_tracer.py` | Langfuse v2 backend; **every SDK call individually guarded** so a dead or slow Langfuse cannot change a response |
| `pricing.py` | Per-model input/output pricing → USD/INR cost per call |

**21 distinct span types** covering the whole request: `parse`, `store`,
`vector_search`, `rrf_fuse`, `rerank`, `prompt_build`, `agent`, `agent.step`,
`agent.plan`, `tool.retrieve_docs`, `tool.fetch_document`, `generate`, `verify`,
`rag.request`, `pipeline.request` …

**Design principle:** fail-open. Observability that can break the product is worse
than no observability.

**Proves:** LLMOps, tracing, cost monitoring, prompt versioning, production monitoring.

---

## 6. Evaluation — the strongest asset

`app/eval/` — 2,000+ LOC across three harnesses over one fixture and one ground truth.

### 6.1 Retrieval harness (`harness.py`)
recall@k, MRR, precision@k, per-category breakdown, three-mode comparison.

### 6.2 Generation harness (`gen_harness.py`)
Five RAGAS-aligned metrics: faithfulness, answer-relevancy, context-precision,
context-recall, answer-correctness.

**Two profiles**, which is the key design decision:
- **surrogate** — deterministic extractive generator + lexical judge. No API, no
  torch. This is what CI gates on, so a regression turns the build red with no key
  and no cost.
- **neural** — the real model generating *and* judging. The definitive numbers.

**Failure taxonomy** — every query gets exactly one class that *names the fix*:

| class | routes to |
|---|---|
| `retrieval_miss` | search / embeddings |
| `over_refusal` | confidence threshold |
| `hallucination` | grounding / prompt |
| `off_topic` | query understanding |
| `partial_answer` | generation quality |

That is what converts evaluation from a scoreboard into a debugging tool.

### 6.3 Agent harness (`agent_harness.py`)
Agent vs RAG with a **retrieval-budget sweep** (`--sweep 1,2,3,5`). Both arms share
generator and judge, so every delta is attributable to retrieval strategy alone.

### 6.4 RAGAS interoperability (`ragas_compat.py`)
Exports runs in RAGAS schema and cross-checks against the reference library.
**Current status: does not validate** (max deviation 0.449 on a degraded run) — and
[`eval/RAGAS_CROSSCHECK.md`](../eval/RAGAS_CROSSCHECK.md) says so explicitly rather
than presenting the numbers as a pass.

### Measured — neural profile, real model generating and judging, 45 queries

| metric | score |
|---|:--:|
| faithfulness | 0.888 |
| answer-relevancy | 1.000 |
| context-precision | 0.959 |
| context-recall | 0.948 |
| answer-correctness | 0.978 |

Artifact: [`eval/GEN_BASELINE.md`](../eval/GEN_BASELINE.md)

### CI regression gate
`.github/workflows/eval.yml` — hermetic (no API key), fails the build if quality drops
below a floor or a failure class exceeds its ceiling. Ceilings are **profile-specific**:
the extractive generator is grounded by construction so any hallucination is a pure
regression (0.0), while a real LLM writes prose and the strict "not directly supported"
test counts ordinary elaboration — demanding 0% there is a permanently red light, and a
gate that is always red stops being a signal.

**Proves:** evaluation as a discipline, RAGAS metrics, CI/CD, regression testing.

---

## 7. The bug the evaluation caught — the best interview story here

Running the generation harness against a real LLM judge for the first time produced
faithfulness **0.713** with a **33%** hallucination rate. Before publishing that, the
failures were triaged — and most were false.

`q-gx4200-firmware-nat` answered *"The recommended firmware version for the GX-4200 is
4.0.2 [1, 2]"* — verbatim correct — and scored **0.0**. Capturing the raw verifier
output showed why:

> "the citation `[1, 2]` is inaccurate because the source excerpts do not use a
> numbering system"

**Two real production defects:**

1. The generation prompt numbered its sources `[n]`, but the **verification prompt
   passed an unnumbered blob** — so the verifier flagged the answer's own citation
   markers as unsupported claims.
2. The parser counted verifier lines that said a claim ***is* supported** as evidence
   against it, inverting their meaning.

Both drove the **user-visible confidence badge and the abstention path**, not just the
eval. Fixed → faithfulness **0.713 → 0.888**, false hallucination rate **0.333 → 0.156**.
Regression tests use the verbatim captured verifier output.

The remaining 15.6% is **real and deliberately left visible**: answers that are
factually correct but add unsupported elaboration.

---

## 8. Pipeline dashboard — record-then-replay

`app/routers/pipeline.py` + `app/templates/pipeline.html`, served at `/pipeline`.

`POST /pipeline/runs` executes one query through the real pipeline **once**, times
every stage, meters token and rupee cost on every LLM call, captures the retrieved
evidence and the groundedness verdict, and serialises all of it into a durable
artifact the dashboard replays step by step.

**Faithful by construction:** it records what executed; nothing is synthesised.

This is the single best demo surface in the project — it makes an invisible pipeline
watchable, with real per-stage timings and costs.

---

## 9. Latency & cost engineering

- **Cost:** per-call token usage → USD/INR via `app/observability/pricing.py`,
  attached to the active span and to every recorded run.
- **Latency:** `scripts/latency_report.py` → [`eval/LATENCY.md`](../eval/LATENCY.md),
  per-stage p50/p95/max from real recorded runs. **p95 rather than mean** — a RAG
  request is a chain of network calls and the mean hides the tail a user notices.

Also required real engineering to make live evaluation possible at all: a shared
requests-per-minute pacer with retry on transient failures (429 **and** 5xx). ~180
calls per run against a 15 rpm free tier died partway through, which is precisely why
the neural profile had never produced a committed number before.

---

## 10. Platform & security

JWT + OAuth (`app/core/security.py`, `oauth.py`), bcrypt password hashing, password
reset tokens, auth event auditing, per-user data isolation enforced at the query layer,
Docker Compose (app / worker / Redis / Chroma, with opt-in Ollama and Langfuse
profiles), Caddy HTTPS deployment guide, SQLAlchemy + Alembic.

---

# Résumé material

## Ready-to-use bullets

Lead with evaluation and observability — those are the scarcest skills relative to
demand. Keep the numbers; they are what separate this from every other RAG project.

> **DocuChunk — Agentic RAG platform with LLMOps** · Python, FastAPI, Chroma, BM25,
> Gemini, Langfuse, Docker
>
> - Built a **generation-quality evaluation harness** (RAGAS-aligned faithfulness,
>   answer-relevancy, context-precision/recall, answer-correctness) with a **failure
>   taxonomy** that routes each failure to its fix; wired it into a **hermetic CI gate**
>   that fails the build on quality regression with no API key required.
> - Evaluation caught two defects in the **verification layer** that were zeroing
>   confidence on correct answers in production; measured faithfulness **0.713 → 0.888**
>   and false-hallucination rate **0.333 → 0.156** after the fix.
> - Implemented a **ReAct agent loop** (tool use, multi-step planning, 5 guardrails,
>   fallback-to-RAG) and **quantified when it is worth its cost** — +0.431 context-recall
>   at k=1 on multi-hop queries, decaying to zero by k=5.
> - Engineered **hybrid retrieval** (dense + BM25 + RRF + cross-encoder rerank), lifting
>   **MRR 0.777 → 0.967**; documented the trade-off where the lexical leg regresses
>   semantic queries and rerank repairs it.
> - Instrumented the full request with **21 trace span types**, per-call **token + ₹ cost
>   attribution** and prompt versioning via Langfuse, designed **fail-open** so tracing
>   can never break the product.
> - **421 tests** across retrieval, generation, agent, observability and API layers.

## Interview questions this prepares you for

| They ask | You have |
|---|---|
| "How do you know your RAG system is any good?" | Three harnesses, two profiles, a CI gate, committed baselines |
| "How do you debug a bad answer?" | The failure taxonomy — each class names the fix |
| "Why an agent? Isn't it just slower RAG?" | The budget sweep and its crossover point |
| "Have you found a real bug with evals?" | Section 7, with the raw verifier output |
| "How do you control LLM cost?" | Per-call pricing, model tiering, caching, cost per recorded run |
| "What would you do differently?" | Postgres/pgvector, load testing, a tightened generation prompt for the residual 15.6% |

## Do NOT claim

- ~~"RAGAS-validated"~~ — the cross-check does not validate yet (Δ 0.449). Say
  "RAGAS-aligned metrics, cross-check in progress."
- ~~"Production-grade at scale"~~ — SQLite and single-node Chroma. Say "production
  *shape*: auth, workers, tracing, CI gates."
- ~~"LangChain/LangGraph"~~ unless you frame it honestly: "hand-rolled ReAct loop,
  LangGraph-equivalent nodes, no framework lock-in."

---

# The precise path to ₹20 LPA+

The work above supports **₹15–20**. Moving the *ceiling* past ₹20 needs a different
class of evidence — not more features in this same repo.

## What ₹20+ actually requires that you do not yet have

| # | Requirement | Current state | Exact action | Effort |
|---|---|---|---|---|
| 1 | **A live, reachable demo** | deployment guide only | Deploy to Oracle Always-Free per `docs/DEPLOYMENT.md`; URL in README line 1 | ~4 hrs |
| 2 | **Scale & system design evidence** | SQLite, single-node Chroma, no load test | Migrate to **Postgres + pgvector**; load-test with Locust at 50/100/200 concurrent; publish p95 + a bottleneck analysis | ~3 days |
| 3 | **A modeling artifact** | none — you consume models, never train one | **Fine-tune the cross-encoder reranker** on your own eval fixture; publish before/after MRR | ~4 days |
| 4 | **RAGAS validated** | Δ 0.449, not validated | One clean run: matched judge + full contexts | ~2 hrs |
| 5 | **Framework fluency, provably** | hand-rolled | Port **one** path to LangGraph behind the existing seam; keep both, benchmark them | ~2 days |

## Ranked by ₹ per hour of effort

**1. Deploy it (~4 hrs).** Highest return of anything on this list. It converts the
entire repo from "code someone must read" into "a system I can click." Do this first
regardless of everything else.

**2. Fine-tune the reranker (~4 days) — the single biggest ceiling-raiser.** This is
the one gap that structurally separates ₹15–20 from ₹22–25. Every candidate in this
band consumes LLM APIs; very few have trained anything. You are unusually well
positioned because **you already have the hard part**: a labelled fixture (45 queries
with ground-truth chunk IDs) and a harness that measures MRR. Fine-tune
`ms-marco-MiniLM` on your own retrieval ground truth, publish before/after. That single
artifact converts "AI engineer who uses models" into "AI engineer who improves models."

**3. Postgres + pgvector + load test (~3 days).** Answers the system-design interview
you currently cannot. Also lets you honestly say "production-grade."

**4. Clean RAGAS run (~2 hrs).** Removes your last unvalidated claim.

**5. LangGraph port (~2 days).** Do this *last* and only for ATS. Your hand-rolled loop
is better engineering; the port exists so a keyword filter and a sceptical interviewer
both get what they want. Benchmarking the two against each other is itself a strong
talking point.

## The honest ceiling statement

- **₹15–20:** achieved by what exists today, *conditional on items 1 and 4*.
- **₹20–23:** add item 2 or 3. **Item 3 (fine-tuning) has the higher ceiling** because
  it changes your category, not just your depth.
- **₹23–25+:** items 2 **and** 3, plus company selection — GCCs and AI-native product
  companies pay 40–70% more than services firms for identical experience. Applying
  broadly to services caps you at ₹12–14 regardless of what this repo contains.

**Total to a defensible ₹20+: roughly 8 working days**, ordered 1 → 4 → 3 → 2 → 5.

The sequencing matters. Deploy first because it multiplies the value of everything
already built. Fine-tune before scaling, because a modeling artifact raises the ceiling
while infrastructure work only removes an objection.
