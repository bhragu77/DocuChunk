"""
LlamaIndex interop — expose DocuChunk's retriever as a `BaseRetriever`.

The deliberate choice here is to plug our retrieval INTO LlamaIndex rather than
rebuild the pipeline in it. Rebuilding would test LlamaIndex's chunker, embedder and
vector store — none of which are ours — and would tell us nothing about our system.
Wrapping the retriever isolates the one thing worth comparing: what LlamaIndex's
`ResponseSynthesizer` does with the same evidence our own prompt path receives.

It also proves something an interviewer will care about more than either result: our
retriever satisfies a standard interface, so this system can drop into an existing
LlamaIndex stack instead of demanding one be built around it.

Every llama_index import is lazy. Importing this module with LlamaIndex absent is
safe; only calling the builders requires it.

    pip install -r requirements-integrations.txt
"""
from __future__ import annotations

from typing import Any

from app.integrations.retriever_core import (
    RetrievalDeps,
    chunk_to_payload,
    relevance_score,
    retrieve,
)


def build_retriever(deps: RetrievalDeps, top_k: int = 5) -> Any:
    """Return a LlamaIndex `BaseRetriever` backed by hybrid search + reranking.

    Constructed inside the function because `BaseRetriever` must be subclassed at
    definition time, and defining the class at module scope would make importing
    this module fail without LlamaIndex installed.
    """
    from llama_index.core.retrievers import BaseRetriever
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

    class DocuChunkRetriever(BaseRetriever):
        """Hybrid dense+BM25 retrieval with cross-encoder reranking, as a LlamaIndex
        retriever. Node IDs are our chunk IDs so citations remain traceable back
        through the framework to the exact source chunk."""

        def __init__(self, deps: RetrievalDeps, top_k: int):
            self._deps = deps
            self._top_k = top_k
            super().__init__()

        def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
            query = getattr(query_bundle, "query_str", str(query_bundle))
            out: list[NodeWithScore] = []
            for c in retrieve(query, self._deps, top_k=self._top_k):
                text, meta = chunk_to_payload(c)
                node = TextNode(text=text, id_=c.chunk_id, metadata=meta)
                out.append(NodeWithScore(node=node, score=relevance_score(c)))
            return out

    return DocuChunkRetriever(deps, top_k)


def build_query_engine(deps: RetrievalDeps, llm: Any = None, top_k: int = 5) -> Any:
    """A LlamaIndex `RetrieverQueryEngine` over our retriever.

    `llm` accepts any LlamaIndex-compatible LLM. Passing None uses whatever is in
    `Settings.llm`, which the benchmark sets explicitly so the framework comparison
    runs against the SAME model our native path uses — otherwise the measurement
    would compare two models rather than two synthesis strategies.
    """
    from llama_index.core.query_engine import RetrieverQueryEngine
    from llama_index.core.response_synthesizers import get_response_synthesizer

    retriever = build_retriever(deps, top_k=top_k)
    synth = get_response_synthesizer(llm=llm) if llm is not None else get_response_synthesizer()
    return RetrieverQueryEngine(retriever=retriever, response_synthesizer=synth)
