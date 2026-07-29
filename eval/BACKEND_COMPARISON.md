# Vector backend comparison

Identical fixture, ground truth, embeddings and metrics — **only the storage layer varies**, so every delta is attributable to the backend. k = 5.

| backend | ops burden | MRR | recall@k | precision@k | p50 query | p95 query |
|---|---|:--:|:--:|:--:|:--:|:--:|
| **chroma** | self-hosted (embedded) | 0.777 | 0.867 | 0.280 | 2.1 ms | 3.4 ms |
| **pgvector** | self-hosted (Postgres) | 0.777 | 0.867 | 0.280 | 2.2 ms | 3.9 ms |
| **pinecone** | managed (serverless) | 0.777 | 0.867 | 0.280 | 289.6 ms | 337.6 ms |

### Operational finding: eventual consistency

Pinecone's warm-up pass **failed ground-truth resolution** on the first attempt —
an expected snippet "matched no stored chunk" immediately after ingestion. The
subsequent measured run, against the same data, resolved every snippet and scored
identically to the self-hosted backends.

That is Pinecone behaving as documented: writes are **eventually consistent**, so a
read issued straight after an upsert may not see everything. pgvector and Chroma are
read-after-write consistent, so the same sequence never fails there.

This does not affect retrieval quality, but it is a real API-contract difference and
it dictates ingestion design: after indexing a document you cannot immediately query
it and trust the result. Anything that ingests-then-verifies needs a poll or a delay
on Pinecone, and needs neither on the other two. It is exactly the class of
difference a quality-only comparison would miss.

### How to read this

- **Retrieval quality should be near-identical.** All backends score the same embeddings with the same cosine metric, so a large MRR gap would indicate an adapter bug, not a better database. Equal quality is the expected — and useful — result.
- **Latency is measured around the vector call only**, excluding embedding and reranking, which would otherwise dominate and mask the difference.
- **A managed service pays a network round-trip** that an embedded or same-host database does not. That is the cost of not operating it — read the latency column against the ops column, not on its own.
- **The decision:** when quality is equal, choose on operational burden, consistency guarantees and scale ceiling. pgvector keeps rows and vectors in one transaction boundary; Pinecone removes index operations entirely; Chroma is simplest to start with and hardest to scale.
