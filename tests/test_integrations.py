"""
Framework adapter tests.

Two layers:

  * `retriever_core` is plain Python and always runs. It is the shared retrieval
    path, so if it drifts, the framework comparison silently stops comparing like
    with like.
  * The LlamaIndex and LangChain adapters skip when the framework is absent. That
    skip IS the isolation guarantee being asserted: the suite must stay green on a
    machine with none of the integration extras installed.
"""
from __future__ import annotations

import pytest

from app.integrations.retriever_core import (
    CANDIDATE_CAP,
    RetrievalDeps,
    chunk_to_payload,
    relevance_score,
    retrieve,
)
from app.pipeline.retrieval import ScoredChunk


def _chunk(cid="c1", text="hello", rr=None, fused=0.5, **kw) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=cid, text=text, doc_id=kw.get("doc_id", "d1"),
        source=kw.get("source", "f.pdf"), page_number=kw.get("page_number", 3),
        fused_score=fused, dense_rank=kw.get("dense_rank", 1),
        bm25_rank=kw.get("bm25_rank", 2), reranker_score=rr,
        metadata=kw.get("metadata", {}),
    )


def _deps() -> RetrievalDeps:
    return RetrievalDeps(user_id="u1", vector_client=object(),
                         bm25_index=object(), embed_fn=lambda t: [0.1] * 384)


# ── shared core ───────────────────────────────────────────────────────────────

def test_retrieve_widens_pool_then_reranks_to_top_k(monkeypatch):
    """The reranker needs more candidates than it returns, or it has nothing to
    reorder and the cross-encoder stage becomes a no-op."""
    seen = {}

    def fake_hybrid(**kw):
        seen.update(kw)
        return [_chunk(f"c{i}") for i in range(20)]

    def fake_rerank(query, candidates, top_k):
        seen["rerank_top_k"] = top_k
        return candidates[:top_k]

    monkeypatch.setattr("app.integrations.retriever_core.hybrid_search", fake_hybrid)
    monkeypatch.setattr("app.integrations.retriever_core.rerank", fake_rerank)

    out = retrieve("q", _deps(), top_k=5)
    assert seen["top_n"] == 30            # 5 * CANDIDATE_MULTIPLIER
    assert seen["rerank_top_k"] == 5
    assert len(out) == 5


def test_candidate_pool_is_capped(monkeypatch):
    seen = {}
    monkeypatch.setattr("app.integrations.retriever_core.hybrid_search",
                        lambda **kw: seen.update(kw) or [])
    retrieve("q", _deps(), top_k=100)
    assert seen["top_n"] == CANDIDATE_CAP


def test_empty_retrieval_skips_rerank(monkeypatch):
    """Reranking an empty list would load the cross-encoder for nothing."""
    called = {"rerank": False}
    monkeypatch.setattr("app.integrations.retriever_core.hybrid_search", lambda **kw: [])

    def _boom(*a, **k):
        called["rerank"] = True
        raise AssertionError("rerank must not run on an empty candidate list")

    monkeypatch.setattr("app.integrations.retriever_core.rerank", _boom)
    assert retrieve("q", _deps()) == []
    assert called["rerank"] is False


def test_doc_id_filter_is_forwarded(monkeypatch):
    seen = {}
    monkeypatch.setattr("app.integrations.retriever_core.hybrid_search",
                        lambda **kw: seen.update(kw) or [])
    deps = _deps()
    deps.doc_id = "doc-42"
    retrieve("q", deps)
    assert seen["doc_id"] == "doc-42"


def test_payload_preserves_per_stage_scores():
    """Frameworks hide retrieval internals; carrying the scores through is what
    keeps a debugging session inside the framework able to see WHY a chunk ranked."""
    text, meta = chunk_to_payload(_chunk(rr=9.5, fused=0.42))
    assert text == "hello"
    assert meta["chunk_id"] == "c1"
    assert meta["reranker_score"] == 9.5
    assert meta["fused_score"] == 0.42
    assert meta["dense_rank"] == 1 and meta["bm25_rank"] == 2
    assert meta["page_number"] == 3


def test_chunk_metadata_is_merged_not_dropped():
    _, meta = chunk_to_payload(_chunk(metadata={"section": "intro"}))
    assert meta["section"] == "intro"
    assert meta["chunk_id"] == "c1"


def test_relevance_prefers_reranker_score():
    assert relevance_score(_chunk(rr=7.25, fused=0.1)) == 7.25


def test_relevance_falls_back_to_fused_score():
    assert relevance_score(_chunk(rr=None, fused=0.33)) == 0.33


# ── isolation guarantee ───────────────────────────────────────────────────────

def test_adapters_import_without_frameworks_installed():
    """Importing an adapter must never require the framework — only CALLING the
    builders does. This is what keeps a missing extra from breaking the app."""
    import app.integrations.langchain_adapter as lc
    import app.integrations.llamaindex_adapter as li
    assert callable(li.build_retriever)
    assert callable(lc.build_retriever)
    assert callable(lc.build_lcel_chain)


def test_context_formatter_numbers_sources():
    """Without numbering the model cannot emit [n], and every downstream citation
    check becomes inapplicable — the benchmark would then compare a citing system
    against a non-citing one and call the gap 'framework quality'."""
    from app.integrations.langchain_adapter import format_context

    class D:
        def __init__(self, t, m):
            self.page_content, self.metadata = t, m

    out = format_context([D("alpha", {"source": "a.pdf", "page_number": 1}),
                          D("beta", {"source": "b.pdf", "page_number": 7})])
    assert "[1] (a.pdf, p.1)" in out
    assert "[2] (b.pdf, p.7)" in out
    assert "alpha" in out and "beta" in out


# ── LlamaIndex (skipped when absent) ──────────────────────────────────────────

def test_llamaindex_retriever_returns_scored_nodes(monkeypatch):
    pytest.importorskip("llama_index.core")
    from app.integrations.llamaindex_adapter import build_retriever

    monkeypatch.setattr("app.integrations.retriever_core.hybrid_search",
                        lambda **kw: [_chunk("c1", "alpha", rr=8.0)])
    monkeypatch.setattr("app.integrations.retriever_core.rerank",
                        lambda q, c, top_k: c[:top_k])

    nodes = build_retriever(_deps(), top_k=3).retrieve("q")
    assert len(nodes) == 1
    assert nodes[0].score == 8.0
    assert nodes[0].node.get_content() == "alpha"
    # Chunk ID must survive so a citation can be traced back through the framework.
    assert nodes[0].node.node_id == "c1"
    assert nodes[0].node.metadata["source"] == "f.pdf"


# ── LangChain (skipped when absent) ───────────────────────────────────────────

def test_langchain_retriever_returns_documents(monkeypatch):
    pytest.importorskip("langchain_core")
    from app.integrations.langchain_adapter import build_retriever

    monkeypatch.setattr("app.integrations.retriever_core.hybrid_search",
                        lambda **kw: [_chunk("c1", "alpha", rr=8.0)])
    monkeypatch.setattr("app.integrations.retriever_core.rerank",
                        lambda q, c, top_k: c[:top_k])

    docs = build_retriever(_deps(), top_k=3).invoke("q")
    assert len(docs) == 1
    assert docs[0].page_content == "alpha"
    assert docs[0].metadata["chunk_id"] == "c1"
    assert docs[0].metadata["reranker_score"] == 8.0


def test_lcel_chain_pipes_retrieval_into_the_model(monkeypatch):
    pytest.importorskip("langchain_core")
    from langchain_core.language_models.fake import FakeListLLM

    from app.integrations.langchain_adapter import build_lcel_chain

    monkeypatch.setattr("app.integrations.retriever_core.hybrid_search",
                        lambda **kw: [_chunk("c1", "The battery is 88-AZ-0097.", rr=8.0)])
    monkeypatch.setattr("app.integrations.retriever_core.rerank",
                        lambda q, c, top_k: c[:top_k])

    chain = build_lcel_chain(FakeListLLM(responses=["The battery is 88-AZ-0097 [1]."]),
                             _deps(), top_k=3)
    assert "88-AZ-0097" in chain.invoke("which battery")
