# Framework comparison — synthesis over identical retrieval

All three paths call `app/integrations/retriever_core.retrieve`, so the evidence set is **byte-identical**; only synthesis differs. All three are given the same model, because comparing two models and calling it a framework comparison is the easiest way to be confidently wrong.

| path | queries | ok | citation rate | p50 | p95 | avg answer chars |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| **native** | 5 | 5 | 80% | 1801.0 ms | 2835.0 ms | 1051 |
| **llamaindex** | 5 | 5 | 0% | 3075.0 ms | 3888.0 ms | 280 |
| **langchain** | 5 | 5 | 40% | 3390.0 ms | 3747.0 ms | 230 |

> **Sample: 5 queries over a single 62-chunk document.** Large enough to expose a
> categorical difference in citation behaviour, too small to rank latency precisely.
> Treat the citation column as the finding and the millisecond columns as indicative.

### What this measures, and what it does not

- **Citation rate is the headline, not answer quality.** Every downstream guarantee in this system — citation validation, groundedness scoring, confidence, abstention — requires the model to emit `[n]` markers against numbered sources. A framework whose default prompt returns fluent uncited prose has not produced a worse answer; it has produced one this system **cannot verify**.
- **The retriever is the interop claim.** Our hybrid + reranked retrieval satisfies both `BaseRetriever` interfaces, so this pipeline drops into an existing LlamaIndex or LangChain stack rather than requiring one be built around it. That is the portable, reusable part.
- **What the frameworks buy:** composition plumbing and a large integration catalogue. **What they cost:** a dependency tree that has already conflicted in this repo (ragas 0.4.3 pinned `langchain-community<0.4`), and prompt behaviour you must override anyway the moment you need verifiable output.
