"""
LangChain interop — a `BaseRetriever` and an LCEL generation chain.

Two separate pieces, deliberately:

  * `build_retriever()` makes our hybrid+reranked retrieval usable inside any
    LangChain chain — the interop claim.
  * `build_lcel_chain()` expresses generation as `prompt | llm | parser`, LangChain's
    core composition primitive, so it can be benchmarked against `generate_answer`.

The comparison is only meaningful because BOTH paths retrieve through
`retriever_core.retrieve` and are given the SAME model. What differs is prompt
construction and response assembly — which is exactly the part a framework claims to
do for you.

Note what LangChain's default prompting does NOT do: our native path numbers every
source `[n]` and requires the model to cite them, which is what makes citation
validation and groundedness scoring possible downstream. A generic
"answer from the context" chain produces prose with no citation anchors, so the
verification layer has nothing to check. That trade-off is the finding, not a
shortcoming to hide.

Every langchain import is lazy; importing this module without LangChain is safe.

    pip install -r requirements-integrations.txt
"""
from __future__ import annotations

from typing import Any

from app.integrations.retriever_core import (
    RetrievalDeps,
    chunk_to_payload,
    retrieve,
)

# Mirrors the intent of app/generation/prompt.py: numbered sources, explicit
# grounding instruction, explicit refusal path. Kept close to the native prompt so
# the benchmark compares FRAMEWORK MECHANICS rather than prompt-writing skill.
LCEL_PROMPT = """You are answering strictly from the numbered sources below.

Rules:
- Use ONLY the sources. Do not add outside knowledge.
- Cite every claim with its source number in square brackets, e.g. [1].
- If the sources do not contain the answer, say exactly: NO_ANSWER

Sources:
{context}

Question: {question}

Answer:"""


def build_retriever(deps: RetrievalDeps, top_k: int = 5) -> Any:
    """Return a LangChain `BaseRetriever` backed by hybrid search + reranking.

    Defined inside the function so this module imports cleanly without LangChain.
    """
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever

    class DocuChunkRetriever(BaseRetriever):
        deps: RetrievalDeps
        top_k: int = 5

        # LangChain retrievers are pydantic models; RetrievalDeps is a plain
        # dataclass, so arbitrary types must be permitted.
        model_config = {"arbitrary_types_allowed": True}

        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None
        ) -> list[Document]:
            docs = []
            for c in retrieve(query, self.deps, top_k=self.top_k):
                text, meta = chunk_to_payload(c)
                docs.append(Document(page_content=text, metadata=meta))
            return docs

    return DocuChunkRetriever(deps=deps, top_k=top_k)


def format_context(docs: list) -> str:
    """Number the sources so the model can cite them.

    LangChain's stock document formatters join page contents with newlines and no
    identifiers. Without numbering the model cannot emit `[n]`, and every downstream
    citation check becomes inapplicable — so the benchmark would be comparing a
    citing system against a non-citing one and calling the difference "framework
    quality".
    """
    parts = []
    for i, d in enumerate(docs, start=1):
        meta = getattr(d, "metadata", {}) or {}
        src = meta.get("source", "?")
        page = meta.get("page_number", "?")
        content = getattr(d, "page_content", str(d))
        parts.append(f"[{i}] ({src}, p.{page})\n{content}")
    return "\n\n".join(parts)


def build_lcel_chain(llm: Any, deps: RetrievalDeps, top_k: int = 5) -> Any:
    """`retrieve | prompt | llm | parse` as a LangChain Runnable.

    `llm` must be a LangChain chat model. The benchmark passes one wrapping the same
    provider our native path uses, so the delta measured is synthesis, not model.
    """
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough

    retriever = build_retriever(deps, top_k=top_k)
    prompt = ChatPromptTemplate.from_template(LCEL_PROMPT)

    def _fetch(question: str) -> str:
        return format_context(retriever.invoke(question))

    return (
        {"context": RunnableLambda(_fetch), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
