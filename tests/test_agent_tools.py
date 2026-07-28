"""
Agent tool registry — wraps the EXISTING retrieval functions with correct
user-scoping and normalized output. Retrieval itself is mocked (the wrapping is
what's under test, not hybrid_search).
"""
from app.generation.agent.tools import build_tools
from app.pipeline.retrieval import ScoredChunk


def _chunk(cid, text="text", doc="d1", page=2):
    return ScoredChunk(
        chunk_id=cid, text=text, doc_id=doc, source="s.pdf",
        page_number=page, fused_score=1.0, reranker_score=0.9,
    )


def test_retrieve_docs_is_user_and_doc_scoped(monkeypatch):
    seen = {}

    def fake_hybrid(**kw):
        seen.update(kw)
        return [_chunk("c1"), _chunk("c2")]

    def fake_rerank(query, candidates, top_k=8, reranker=None):
        seen["rerank_top_k"] = top_k
        return candidates[:top_k]

    monkeypatch.setattr("app.generation.agent.tools.hybrid_search", fake_hybrid)
    monkeypatch.setattr("app.generation.agent.tools.rerank", fake_rerank)

    tools = build_tools("user-42", chroma="CH", bm25="BM", embed_fn=lambda t: [0.1],
                        doc_id="docX")
    result = tools["retrieve_docs"].func(query="warranty duration", top_k=4)

    # Scoping: the wrapper pins user_id + doc_id exactly like /generate/answer.
    assert seen["user_id"] == "user-42"
    assert seen["doc_id"] == "docX"
    assert seen["chroma_client"] == "CH"
    assert seen["bm25_index"] == "BM"
    assert seen["rerank_top_k"] == 4

    # Normalized, model-facing payload.
    assert result.status == "success"
    assert result.source == "hybrid_search"
    assert result.data["count"] == 2
    assert [c["chunk_id"] for c in result.data["chunks"]] == ["c1", "c2"]
    # Raw chunks preserved for real citations.
    assert [c.chunk_id for c in result.chunks] == ["c1", "c2"]


def test_retrieve_docs_no_results(monkeypatch):
    monkeypatch.setattr("app.generation.agent.tools.hybrid_search", lambda **kw: [])
    monkeypatch.setattr("app.generation.agent.tools.rerank", lambda *a, **k: [])

    tools = build_tools("u", chroma="CH", bm25="BM", embed_fn=lambda t: [0.1])
    result = tools["retrieve_docs"].func(query="nothing")

    assert result.status == "no_results"
    assert result.chunks == []
    assert "no chunks found" in result.observation


def test_retrieve_docs_caps_top_k(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        "app.generation.agent.tools.hybrid_search",
        lambda **kw: seen.update(kw) or [],
    )
    monkeypatch.setattr("app.generation.agent.tools.rerank", lambda *a, **k: [])

    tools = build_tools("u", chroma="CH", bm25="BM", embed_fn=lambda t: [0.1])
    tools["retrieve_docs"].func(query="q", top_k=9999)
    # top_n is min(k*6, 50); an absurd top_k must not blow the candidate pool.
    assert seen["top_n"] == 50


def test_fetch_document_wraps_whole_document(monkeypatch):
    seen = {}

    def fake_whole(chroma, user_id, doc_id):
        seen.update(chroma=chroma, user_id=user_id, doc_id=doc_id)
        return [_chunk("w1"), _chunk("w2"), _chunk("w3")]

    monkeypatch.setattr("app.generation.agent.tools.fetch_whole_document", fake_whole)

    tools = build_tools("user-7", chroma="CH", bm25="BM", embed_fn=lambda t: [0.1])
    result = tools["fetch_document"].func(doc_id="docZ")

    assert seen == {"chroma": "CH", "user_id": "user-7", "doc_id": "docZ"}
    assert result.source == "whole_document"
    assert result.data["count"] == 3
    assert [c.chunk_id for c in result.chunks] == ["w1", "w2", "w3"]


def test_tool_schema_shape():
    tools = build_tools("u", chroma="CH", bm25="BM", embed_fn=lambda t: [0.1])
    schema = tools["retrieve_docs"].to_schema()
    assert schema["name"] == "retrieve_docs"
    assert schema["parameters"]["required"] == ["query"]
    assert "query" in schema["parameters"]["properties"]
    # doc_id is no longer required — an omitted id resolves to the doc in context.
    assert tools["fetch_document"].to_schema()["parameters"]["required"] == []


# ── fetch_document id resolution (the planner-invents-a-fake-id fix) ─────────────

def _fake_whole(monkeypatch):
    seen = {}
    def fake(chroma, user_id, doc_id):
        seen["doc_id"] = doc_id
        return [_chunk("w1", doc=doc_id)] if doc_id == "real-1" else []
    monkeypatch.setattr("app.generation.agent.tools.fetch_whole_document", fake)
    return seen


def test_fetch_document_resolves_invented_id_when_single_doc(monkeypatch):
    seen = _fake_whole(monkeypatch)
    tools = build_tools("u", chroma="CH", bm25="BM", embed_fn=lambda t: [0.1],
                        available_docs=[{"doc_id": "real-1", "name": "only.pdf"}])
    # planner invents "document_0" — must resolve to the sole real document.
    result = tools["fetch_document"].func(doc_id="document_0")
    assert seen["doc_id"] == "real-1"
    assert result.status == "success"
    assert result.data["doc_id"] == "real-1"
    assert result.data["requested_doc_id"] == "document_0"


def test_fetch_document_omitted_id_uses_single_doc(monkeypatch):
    seen = _fake_whole(monkeypatch)
    tools = build_tools("u", chroma="CH", bm25="BM", embed_fn=lambda t: [0.1],
                        available_docs=[{"doc_id": "real-1", "name": "only.pdf"}])
    result = tools["fetch_document"].func()          # no doc_id at all
    assert seen["doc_id"] == "real-1" and result.status == "success"


def test_fetch_document_known_id_used_as_is(monkeypatch):
    seen = _fake_whole(monkeypatch)
    tools = build_tools("u", chroma="CH", bm25="BM", embed_fn=lambda t: [0.1],
                        available_docs=[{"doc_id": "real-1", "name": "a"},
                                        {"doc_id": "real-2", "name": "b"}])
    result = tools["fetch_document"].func(doc_id="real-1")
    assert seen["doc_id"] == "real-1" and result.status == "success"


def test_fetch_document_unknown_id_multi_doc_hints_valid_ids(monkeypatch):
    _fake_whole(monkeypatch)
    tools = build_tools("u", chroma="CH", bm25="BM", embed_fn=lambda t: [0.1],
                        available_docs=[{"doc_id": "real-1", "name": "a"},
                                        {"doc_id": "real-2", "name": "b"}])
    # ambiguous: an invented id with several docs → no guess, but a useful hint.
    result = tools["fetch_document"].func(doc_id="document_0")
    assert result.status == "no_results"
    assert "valid ids" in result.observation


def test_fetch_document_scoped_doc_id_overrides_unknown(monkeypatch):
    seen = _fake_whole(monkeypatch)
    tools = build_tools("u", chroma="CH", bm25="BM", embed_fn=lambda t: [0.1],
                        doc_id="real-1",  # run scoped to one doc
                        available_docs=[{"doc_id": "real-1", "name": "a"},
                                        {"doc_id": "real-2", "name": "b"}])
    result = tools["fetch_document"].func(doc_id="document_0")
    assert seen["doc_id"] == "real-1" and result.status == "success"


def test_planner_prompt_lists_available_documents():
    from app.generation.agent.react import build_planner_prompt
    from app.generation.agent.state import AgentState
    tools = build_tools("u", chroma="CH", bm25="BM", embed_fn=lambda t: [0.1])
    state = AgentState(question="summarize", user_id="u",
                       available_docs=[{"doc_id": "real-1", "name": "report.pdf"}])
    prompt = build_planner_prompt(state, tools, remaining=5)
    assert "AVAILABLE DOCUMENTS" in prompt
    assert 'doc_id="real-1"' in prompt and "report.pdf" in prompt
