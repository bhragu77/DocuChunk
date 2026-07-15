"""
Generation providers — the pluggable LLM backends that turn a prompt into text.

Mirrors app/pipeline/embedding_providers.py exactly: a Protocol (base.py), concrete
providers (gemini / openai_compat / stub), and a config-driven factory
(build_gen_provider). The web process wires the built provider's generate() onto
app.state.llm_fn in lifespan; the /generate/answer endpoint passes that llm_fn into
generate_answer() / groundedness_check(). Generation is a WEB-process concern (see
the note in app/worker.py) — the arq worker never builds one.
"""
