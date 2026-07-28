# DocuChunk → "Exclusive" Portfolio Roadmap

> **Purpose:** turn DocuChunk from a strong RAG project into a portfolio piece that
> clears **AI / Applied-AI / GenAI Engineer** roles in the **₹5–25 LPA, 1–2 YoE**
> bracket — by closing the six gaps identified in review with **real, shippable
> features**, each mapped to a real hiring requirement.
>
> **How to read this:** Part A is the market (what's actually being hired for, with
> sources). Part B maps each gap → the feature that closes it. Part C is the
> phase-wise build plan with acceptance criteria and effort. Parts D–F are the
> interview ammunition (system-design decisions + ML/DL principles) so the project
> *and* the person hold up under questioning.
>
> _Written 2026-07-23. Effort estimates assume focused part-time work (2–3 hrs/day)._

---

## Part A — The real market (grounded in current listings)

### A.1 Salary bands vs what they actually demand (India, 1–2 YoE)

| Band | Typical titles | What clears the bar | DocuChunk today |
|---|---|---|---|
| **₹5–10 LPA** (services / entry product / GCC junior) | AI Engineer I, ML Engineer (Jr), GenAI Developer | Python, one LLM API, **RAG basics**, a vector DB (FAISS/Chroma), can ship a FastAPI service, Git, Docker | ✅ **Already exceeds this.** You have RAG + auth + Docker + tests. |
| **₹10–18 LPA** (product companies, GCCs, funded startups) | GenAI Engineer, Applied AI Engineer, LLM Engineer | Everything above **+ hybrid retrieval, reranking, evaluation, agentic/tool-use, LangChain/LangGraph, prompt+context+memory management, production concerns** | 🟡 **Mostly there on retrieval/eval; missing agentic, LLMOps, gen-eval, scale story.** |
| **₹18–25+ LPA** (strong startups, senior-leaning) | Sr. Applied AI, Founding AI Engineer | Everything above **+ LLMOps/observability, cost & latency engineering, system design at scale, some model-training/fine-tuning, evals as a discipline** | 🔴 **This roadmap's target — close all six gaps + one modeling artifact.** |

**Market facts that shaped this plan (2026):**
- LLM/RAG **+ evaluation** skills move the ceiling "from ₹12 LPA to ₹25–40 LPA even at 2 years." Evaluation is the single biggest multiplier — and it's your existing strength. **Lead with it.**
- Hands-on **fine-tuning, RAG, or LLMOps** command a **20–40% premium** over generalist ML engineers.
- Agentic pipelines (**LangGraph / CrewAI / AutoGen**), **RAG + prompt/context/memory/state management**, and vector search are now *named requirements*, not nice-to-haves.
- **Observability is "no longer optional"** for production LLM systems — tracing, eval, prompt management, cost monitoring are the four LLMOps pillars employers expect.
- **Voice AI** is a distinct, well-paid track (globally $100K–140K at 1–2 YoE) with its own named skills: STT/TTS APIs, WebSockets/streaming, low-latency architecture, interruption handling.

### A.2 The named-skill checklist (harvested from live JDs)

Tick = DocuChunk proves it today; blank = a phase below adds it.

- [x] Python, FastAPI, multi-file project, Docker
- [x] LLM API integration (Gemini), provider abstraction
- [x] RAG pipeline (parse → chunk → embed → store → retrieve → generate)
- [x] Vector DB (Chroma), hybrid search (dense + BM25 + RRF), cross-encoder rerank
- [x] Retrieval **evaluation** (recall@k, MRR, precision@k, hardened fixture)
- [x] Grounded generation, citation validation, abstention, confidence
- [x] Auth (JWT/OAuth), async worker (arq/Redis), caching
- [ ] **Agentic / tool-use / multi-step reasoning** → Phase 10
- [ ] **LLMOps: tracing, cost, prompt versioning, prod monitoring** → Phase 9
- [x] **Generation-quality eval (RAGAS-style: faithfulness, answer-relevancy, context precision/recall, answer-correctness)** → Phase 11 ✅ *(app/eval/gen_harness.py + ragas_compat.py; failure taxonomy → debugging; eval/GEN_BASELINE.md; CI gate .github/workflows/eval.yml)*
- [ ] **Voice: STT → LLM → TTS streaming, barge-in, latency budget** → Phase 12
- [ ] **Scale & system design (Postgres/pgvector, batch ingest, load test)** → Phase 13
- [ ] **A model-training / fine-tuning artifact (LoRA/embeddings/reranker)** → Phase 14

---

## Part B — Gap → feature mapping

| # | Gap (from review) | Feature that closes it | Phase | JD skill it proves |
|---|---|---|---|---|
| 1 | No LLMOps / observability | Self-hosted **Langfuse**: trace every RAG request, cost & token dashboards, **prompt versioning**, prod hallucination-rate monitor | **9** | "Observability is no longer optional" |
| 2 | No agentic / tool-use | **Agent loop** with tool-calling (retrieve, web-fetch, calculator, multi-hop) + **stateful graph** router built on your existing scope-router | **10** | "Agentic pipelines, memory/state management" |
| 3 | Generation not evaluated | **RAGAS-style gen eval** (faithfulness, answer-relevancy, context precision/recall) wired into **CI as a regression gate** | **11** | "LLM evaluation frameworks" — your biggest multiplier |
| 4 | (New track) Voice | **Voice mode**: streaming STT → RAG → streaming TTS, barge-in, <500ms target; ElevenLabs **and** a free stack | **12** | Distinct Voice-AI JD track |
| 5 | Scale story thin | **Postgres + pgvector** migration path, **batch/concurrent ingestion**, **load test**, written **system-design doc** | **13** | "How does this scale to 10M docs / 100 users?" |
| 6 | Zero model training | **One modeling artifact**: LoRA fine-tune a reranker/embedding **or** train a small query classifier, measured on *your own* eval set | **14** | "Fine-tuning… 20–40% premium" |

> Gap "it's one project" is dissolved by making these **self-contained, separately-demoable modules** (each gets its own README section + metrics) — so one repo reads as five deliverables. Phase 14 optionally spins out as a standalone repo for range.

---

## Part C — Phase-wise implementation plan

Phases are ordered by **ROI-per-effort**. Each is independently shippable and demoable.
Your existing anchors are referenced so this grafts onto real code, not a rewrite.

---

### Phase 9 — LLMOps & Observability  ⭐ *do first, highest ROI*

**Why first:** cheapest to add, instantly visible in a demo, and it's a named
requirement at every band ≥₹10 LPA. It also *instruments* everything you build next.

**Build:**
1. Self-host **Langfuse** (open-source, Docker) — add a `langfuse` service to `docker-compose.yml`. $0.
2. Wrap the RAG path (`app/generation/factory.py`, `app/pipeline/retrieval.py`,
   `app/generation/streaming.py`) with trace spans: `retrieve → rerank → prompt-build → generate → verify`. Each span logs latency + token counts.
3. **Cost tracking:** per-request token → ₹ cost, aggregated per user/day on a dashboard.
4. **Prompt versioning:** move prompts from `app/generation/prompt_builder.py` into Langfuse-managed, versioned prompts; log which version served each answer.
5. **Prod hallucination monitor:** you already compute groundedness/`confidence_signals` — pipe them to Langfuse as scores so you can chart hallucination/abstention rate over time.

**Files:** `docker-compose.yml`, `app/observability/` (new), wrap points above.
**Acceptance:** a live trace waterfall for a real query; a cost dashboard; two prompt versions A/B'd; a groundedness-over-time chart.
**Proves:** all four LLMOps pillars (tracing, eval, prompt mgmt, cost).
**Effort:** 3–4 days.

---

### Phase 10 — Agentic layer (tool-use + stateful multi-step)  ⭐

**Why:** the most-named 2026 gap. You already have scope-aware routing — promote it to a real agent.

**Build:**
1. **Tool registry** with function-calling: `retrieve_docs`, `web_search` (or `fetch_url`), `calculator`, `list_user_docs`, `get_document_summary`.
2. **Agent loop**: LLM decides tool → execute → observe → repeat → final grounded answer. Guardrails: max steps, timeout, loop detection.
3. **Multi-hop retrieval**: decompose a complex question into sub-queries, retrieve each, synthesize (directly hits "agentic RAG").
4. **Stateful graph**: model the flow as an explicit state machine (router → retrieve → maybe-tool → generate → verify). Build it yourself for depth, **or** wire **LangGraph** so the JD keyword is literally true — recommend building your own first, then a thin LangGraph adapter to name-drop it honestly.
5. **Conversation memory**: you have chat sessions (`app/models/chat.py`) — add summarized long-term memory so the agent uses prior turns.

**Files:** `app/agent/` (new: `tools.py`, `loop.py`, `graph.py`), router in `app/routers/chat.py`.
**Acceptance:** a query needing 2+ tools returns a correct, cited answer with a visible trace of the steps (via Phase 9). A multi-hop question that dense retrieval alone fails, the agent gets right.
**Proves:** "architecting agentic pipelines, tool-use, memory/state management."
**Effort:** 5–7 days.

---

### Phase 11 — Generation-quality evaluation (close the eval loop)  ⭐

**Why:** completes your strongest narrative — "I evaluate *everything*, not just retrieval." Evaluation is the #1 salary multiplier in the data.

**Build:**
1. Add **RAGAS** (or **DeepEval** for the CI ergonomics) alongside your `app/eval/harness.py`.
2. Score generated answers on **faithfulness, answer-relevancy, context-precision, context-recall** over a curated Q/A set (extend `tests/fixtures/eval_set.json`).
3. **LLM-as-judge** with a rubric for answer quality; report inter-metric correlation.
4. **CI regression gate:** a GitHub Action fails the build if faithfulness drops below a threshold (e.g. 0.85) — this is what "evals as a discipline" looks like.
5. Publish a **generation baseline** doc next to your existing `eval/BASELINE.md` and `eval/phase8_comparison.md` (keep that same before/after table format — it's excellent).

**Files:** `app/eval/gen_harness.py` (new), `eval/GEN_BASELINE.md`, `.github/workflows/eval.yml`.
**Acceptance:** a table of faithfulness/relevancy/context scores; a CI run that goes red on a deliberately bad prompt change.
**Proves:** "LLM evaluation frameworks," CI/CD for ML.
**Effort:** 3–4 days.

---

### Phase 12 — Voice streaming (STT → RAG → TTS), with barge-in

**Why:** opens a whole second, well-paid JD track and is a *memorable* demo. Do the free stack first so it costs ₹0; add ElevenLabs as a quality tier.

**Architecture (target: <500ms perceived round-trip — the "conversational" threshold):**
```
Mic ─WebSocket─► STT(stream) ─partial text─► RAG/agent ─token stream─► TTS(stream) ─audio chunks─► Speaker
                    ▲ barge-in: user speech cancels in-flight TTS + LLM
Latency budget:  STT 60–120ms │ LLM first-token 100–250ms │ TTS first-chunk 40–100ms │ net 20–60ms
```

**Build:**
1. **WebSocket** endpoint streaming audio in/out (`app/routers/voice.py`).
2. **STT (free):** `faster-whisper` / `whisper.cpp` streaming (self-hosted, $0). Quality tier: ElevenLabs Scribe / Deepgram (~150ms).
3. **Feed partial transcripts** into your existing streaming generation (`app/generation/streaming.py`) so the LLM starts before the user finishes.
4. **TTS (free):** **Piper** (fast, CPU, $0) or **Coqui XTTS / Kokoro** for nicer prosody. Quality tier: **ElevenLabs** streaming for natural prosody + voice "modulation" (style/emotion params).
5. **Barge-in / interruption:** detect user speech mid-answer → cancel in-flight TTS + LLM (the classic hard part; call it out in the README).
6. **Measure & publish the latency budget** (p50/p95 per layer) — voice JDs explicitly test latency thinking.

**Files:** `app/voice/` (new: `stt.py`, `tts.py`, `session.py`), `app/routers/voice.py`, provider seam mirroring your `app/generation/factory.py` pattern (free vs ElevenLabs) so it's swappable.
**Acceptance:** speak a question, hear a cited spoken answer; interrupt mid-answer and it stops; a published p50/p95 latency table.
**Proves:** entire Voice-AI skill list (streaming, WebSockets, low-latency, interruption handling, STT/TTS).
**Effort:** 7–10 days (barge-in is the long pole).

> **Note:** memory referenced a `docs/VOICE_RAG_STT_TTS_DESIGN.md` design doc — it isn't in the repo. This phase supersedes/becomes that design. Create the file as the detailed spec before coding.

---

### Phase 13 — Scale & system design (make the "10M docs" answer real)

**Why:** you don't need to run at 10M docs — you need a **credible, measured** path and the vocabulary to defend it.

**Build:**
1. **Postgres + pgvector** as an alternative vector/store backend behind your existing store seam (`app/pipeline/vector_store.py`) — swap SQLite→Postgres for metadata too. Keep Chroma as the dev default; make the backend configurable.
2. **Batch + concurrent ingestion:** parallelize the worker (`app/workers/tasks.py`), batch-embed, chunked upserts; measure docs/min.
3. **Load test** with **Locust/k6**: 100 concurrent users, publish p50/p95 latency + throughput, identify the bottleneck (embedding? rerank? LLM?).
4. **Caching tiers** doc: you have a Redis answer cache — document the full hierarchy (embedding cache, retrieval cache, answer cache) and hit-rates.
5. **Write `docs/SYSTEM_DESIGN.md`:** sharding strategy, ANN index choice (HNSW params, recall/latency tradeoff), horizontal scaling of app/worker, vector-DB-at-scale options (pgvector vs Pinecone/Weaviate/Qdrant), multi-tenancy, cost model.

**Files:** `app/pipeline/vector_store.py` (backend seam), `docs/SYSTEM_DESIGN.md`, `loadtest/`.
**Acceptance:** a load-test report with numbers; a working pgvector backend; a system-design doc you can whiteboard from.
**Proves:** "how does this scale," data-engineering, system-design rounds.
**Effort:** 5–7 days.

---

### Phase 14 — One modeling artifact (the ML/DL "training half")

**Why:** kills the "zero training" gap and earns the fine-tuning premium — **without** a months-long detour. Pick **one** of these, all doable free on Colab T4 / Kaggle GPU with **Unsloth/QLoRA**:

**Option A (recommended — highest relevance):** **Fine-tune the cross-encoder reranker** on your own domain pairs. You already have a hardened eval set + a reranking baseline (`eval/phase8_comparison.md`). Generate (query, positive, hard-negative) triples from your fixtures, fine-tune, and show **MRR improvement over the off-the-shelf `ms-marco-MiniLM`** in the *same* comparison table. This is a tight, honest, measurable "before/after training" story that plugs straight into your existing eval narrative.

**Option B:** **QLoRA fine-tune a small open LLM** (Llama-3-8B / Qwen) with **Unsloth** on Colab free tier (fits in ~7–16GB VRAM, ~90 min) for the generation step — e.g., to enforce your citation format or a domain tone. Measure faithfulness (Phase 11) fine-tuned vs base.

**Option C (classical-ML flavor, if targeting "ML Engineer"):** train a **query-intent / scope classifier** (scikit-learn or a small transformer) to replace/augment `app/generation/query_classifier.py`; report precision/recall/F1 with a confusion matrix.

**Build (Option A shape):**
1. Mine training triples from `tests/fixtures/eval_set.json` + your ingested store (hard negatives = near-miss distractors you already designed — perfect for this).
2. Fine-tune with **Unsloth/QLoRA** on Colab; export the adapter.
3. Serve it behind your reranker seam in `app/pipeline/retrieval.py` (config-flag the model).
4. Re-run `python -m app.eval.harness --comparison`; publish `eval/reranker_finetune.md` with the same table format.

**Acceptance:** a training notebook, an adapter/checkpoint, and a metrics table showing your fine-tune beats the stock model on *your* set (or an honest "no gain, here's why" — also a strong signal).
**Proves:** PyTorch/transformers, LoRA/QLoRA, training loop, eval-driven modeling.
**Effort:** 4–6 days. **Optionally spin out as a second repo** for portfolio range.

---

## Part D — System-design decisions cheat sheet (interview ammo)

Be able to defend each *out loud*. These are the questions this project invites.

| Decision | Your answer / tradeoff |
|---|---|
| Chunk size & strategy | Token-based vs sentence vs semantic; overlap tradeoff (recall vs cost/dup); why content-addressable deterministic IDs (idempotent re-index). |
| Dense vs hybrid vs +rerank | Bi-encoder blurs exact identifiers → BM25 recovers lexical; RRF (k=60, Cormack 2009) fuses; cross-encoder does joint scoring for precision. **Cite your own MRR numbers.** |
| RRF vs weighted score fusion | RRF is scale-free, no tuning, robust across score distributions; weighted sum needs per-corpus calibration. |
| Bi-encoder vs cross-encoder | Bi = precompute, fast, scalable (retrieval); cross = per-pair, accurate, expensive (rerank top-k only). |
| Local embeddings + Gemini gen | Cost/privacy for embeddings (no per-call fee, on-box); generation needs frontier quality → API. Clean provider seam either way. |
| ANN index (HNSW) | `M`/`efSearch` tradeoff: recall vs latency vs memory; flat is exact but O(N). |
| Vector DB at scale | Chroma (dev) → pgvector (one DB, transactional) → Pinecone/Qdrant/Weaviate (managed ANN, filtering, sharding) — pick by scale, ops budget, metadata-filter needs. |
| Hallucination control | Groundedness re-check (separate LLM), citation validation, abstention on low confidence — surface, never silently pass. |
| Latency budget (voice) | Where time goes: turn-taking + LLM TTFT dominate; stream everything; barge-in to cancel. |
| Caching hierarchy | Embedding cache → retrieval cache → answer cache (Redis); hit-rate vs staleness. |

---

## Part E — ML/DL principles to be able to explain (own your own code)

You *built* these — make sure you can whiteboard them cold:

- **Embeddings & vector similarity:** what a 384-dim vector is, cosine vs dot vs L2, why normalize.
- **Transformers/attention (conceptual):** Q/K/V, self-attention, why context windows are finite, tokenization (you use tiktoken + NLTK).
- **Bi- vs cross-encoder architectures** (you use both) — the core of your retrieval story.
- **Fine-tuning family:** full vs **LoRA** vs **QLoRA** (low-rank adapters, 4-bit quant, why VRAM drops ~60%); when fine-tune vs RAG vs prompt.
- **Evaluation:** recall@k, MRR, precision@k (retrieval); faithfulness, answer-relevancy, context precision/recall (generation); LLM-as-judge pitfalls.
- **Classical ML basics** (for ML-titled roles): train/val/test split, overfitting, precision/recall/F1, confusion matrix, cross-validation.

---

## Part F — Suggested sequence, effort & priority

**Recommended order (ROI-first):**

1. **Phase 9 — LLMOps** (3–4 d) — instant demo value, instruments everything after.
2. **Phase 11 — Gen eval** (3–4 d) — completes your strongest card; cheap.
3. **Phase 10 — Agentic** (5–7 d) — biggest *named* gap.
4. **Phase 13 — Scale/system design** (5–7 d) — mostly writing + one backend + a load test.
5. **Phase 12 — Voice** (7–10 d) — highest wow-factor; opens a second track.
6. **Phase 14 — Modeling artifact** (4–6 d) — do the reranker fine-tune (Option A) to reuse your eval harness.

**Total:** ~5–6 focused weeks part-time to close all six gaps. **Minimum viable "exclusive" set** if time-boxed: **Phases 9 + 11 + 10** (≈2.5 weeks) already lifts you into confident ₹15+ LPA GenAI territory; add **12** for the voice differentiator.

**Free-tooling summary (₹0 path):**
- Observability: **Langfuse** (self-hosted, open-source).
- Gen eval: **RAGAS / DeepEval** (open-source).
- Agentic: build your own loop; **LangGraph** free.
- Voice STT: **faster-whisper / whisper.cpp**; TTS: **Piper / Coqui XTTS / Kokoro**. ElevenLabs = optional quality tier.
- Fine-tuning: **Unsloth + QLoRA** on **Colab free T4 / Kaggle** (7B fits, ~90 min).
- Scale: **Postgres + pgvector**, **Locust/k6** — all open-source.

**README strategy (ties it together):** restructure the top of `README.md` to lead with a
**metrics table** (retrieval MRR before/after, gen faithfulness, voice p50 latency, load-test throughput) and a **one-line-per-module** map (RAG · Agent · Evals · Voice · LLMOps · Fine-tune). Recruiters skim; numbers + modules = "exclusive."

---

## Sources

- [AI Jobs in India Salary (2026) — buildfastwithai](https://www.buildfastwithai.com/blogs/ai-jobs-india-salary-2026)
- [AI Engineer: Roles, Skills, Salary — taggd.in](https://taggd.in/blogs/ai-engineer/)
- [AI Engineer 2026 Skill Stack — Kalvium](https://kalvium.com/blog/ai-engineer-india-skill-stack/)
- [AI/ML Jobs in India 2026 Hiring Guide — Shifttotech](https://shifttotech.co.in/blog/ai-ml-jobs-india-2025-complete-guide)
- [AI Engineer Skills 2026: GenAI, RAG, Deployment — Technovids](https://technovids.com/ai-engineer-skills)
- [LLM/GenAI/LangChain/LangGraph jobs — Indeed India](https://in.indeed.com/q-llm,gen-ai,-langchain,langgraph-jobs.html)
- [LLMOps Observability: LangSmith vs Arize vs Langfuse vs W&B — Medium/Kanerika](https://medium.com/@kanerika/llmops-observability-langsmith-vs-arize-vs-langfuse-vs-w-b-f1baeabd1bbf)
- [Langfuse — open-source LLM engineering platform (GitHub)](https://github.com/langfuse/langfuse)
- [How Real-Time Voice AI Works (STT→LLM→TTS) — Retell AI](https://www.retellai.com/blog/how-real-time-voice-ai-works-stt-llm-tts)
- [Latency Budgets for Real-Time Voice — The Prompt Bench](https://thepromptbench.com/voice-and-realtime/latency-budgets-for-realtime-voice/)
- [Best STT Providers 2026 — Coval](https://www.coval.ai/blog/best-speech-to-text-providers-in-2026-independent-benchmarks-and-how-to-choose/)
- [Voice AI Engineer Jobs 2026 — Zen van Riel](https://zenvanriel.com/job/voice-ai-engineer-jobs/)
- [Fine-Tuning LLMs 2026: LoRA, QLoRA, Unsloth, MLX — Codersera](https://codersera.com/blog/fine-tuning-llms-complete-guide-2026/)
- [Fine-Tuning with QLoRA & Unsloth — Pockit](https://pockit.tools/blog/fine-tuning-llms-qlora-unsloth-complete-guide/)
- [Ragas RAG Evaluation Metrics Guide 2026 — qaskills](https://qaskills.sh/blog/ragas-rag-evaluation-metrics-complete-guide)
- [RAGAS, TruLens, DeepEval compared — Atlan](https://atlan.com/know/llm-evaluation-frameworks-compared/)
</content>
</invoke>
