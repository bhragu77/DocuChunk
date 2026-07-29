"""
Framework interoperability adapters (LlamaIndex, LangChain).

These exist to answer a question the rest of the codebase cannot: **what do the
mainstream RAG frameworks actually buy over a hand-rolled pipeline, measured on the
same fixture?**

Nothing in the running application imports this package. Every framework import is
lazy, inside the function that needs it, so a missing dependency can never break the
app — the same isolation policy `requirements-eval.txt` uses for RAGAS, which was
adopted after ragas 0.4.3 forced `langchain-community<0.4` and broke the
environment it was installed beside.

    pip install -r requirements-integrations.txt
"""
