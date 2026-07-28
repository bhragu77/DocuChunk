"""
POST /generate/agent — the agentic (ReAct) counterpart to /generate/answer.

Same contract, one difference: instead of a single retrieval, the model PLANS
multi-hop retrieval (app/generation/agent) and the gathered chunks are then run
through the EXACT SAME grounded-generation + verification path as /generate/answer.
So the response schema is /generate/answer's, plus `agent_steps` for transparency,
and every trust signal (citations, groundedness, confidence_signals) is unchanged.

This router deliberately REUSES search.py's building blocks rather than copying
them: the response model, citation shaping, _build_answer_response (validation +
groundedness + confidence), and the bm25/embed_fn state dependencies. The only
new logic is the loop that decides WHAT to retrieve.

Streaming (SSE) is three-phase:
    event: step          — one per executed agent step, LIVE as it happens
    event: token         — final grounded answer tokens (reused stream seam)
    event: verification  — the full payload (identical shape to stream=false)

The answer cache is intentionally NOT used here: agent trajectories vary per
request, so a (user, doc, query) key would be unsound. Generation still records
token cost via the provider seam exactly as the RAG path does.
"""
from __future__ import annotations

import logging
import time
from typing import Callable

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.dependencies import get_current_user, get_chroma
from app.database import get_db
from app.generation.agent import AgentState, agent_events, build_tools
from app.generation.base import GenerationError
from app.generation.prompt_builder import build_grounded_prompt
from app.generation.query_classifier import (
    AnswerTask,
    QueryType,
    classify_query,
    classify_scope,
    classify_task,
)
from app.generation.streaming import sse_event, stream_generation
from app.models.user import User
from app.observability import (
    adopt,
    capture_text_enabled,
    current_trace_id,
    hash_user_id,
    span,
)
from app.pipeline.retrieval import ScoredChunk, generate_answer
from app.routers.search import (
    Citation,
    GenerateAnswerRequest,
    GenerateAnswerResponse,
    _attach_verify_scores,
    _build_answer_response,
    _generation_error_response,
    _get_bm25,
    _get_embed_fn,
)

logger = logging.getLogger(__name__)

agent_router = APIRouter(prefix="/generate", tags=["agent"])

# retrieval_scope value reported by the agent path — the model, not the heuristic
# router, chose what to fetch, so neither "local" nor "global" is accurate.
_AGENT_SCOPE = "agent"


class AgentAnswerResponse(GenerateAnswerResponse):
    """/generate/answer's payload + the executed agent trajectory (transparency)."""
    agent_steps: list[dict] = []


def _citations(chunks: list[ScoredChunk]) -> list[Citation]:
    return [
        Citation(
            chunk_id=c.chunk_id,
            source=c.source,
            page_number=c.page_number,
            text_excerpt=c.text[:200],
        )
        for c in chunks
    ]


def _not_configured(query: str, query_type: str) -> AgentAnswerResponse:
    """Honest shape when no generation backend is wired (the agent needs llm_fn to
    plan AND to answer). Mirrors /generate/answer's generation_not_configured."""
    return AgentAnswerResponse(
        query=query,
        answer=None,
        citations=[],
        cited_sources=[],
        all_sources=[],
        abstained=False,
        grounded=False,
        confidence=0.0,
        unsupported_claims=[],
        verified=False,
        error="generation_not_configured",
        query_type=query_type,
        retrieval_scope=_AGENT_SCOPE,
        agent_steps=[],
    )


@agent_router.post("/agent", response_model=AgentAnswerResponse)
async def generate_agent_endpoint(
    payload: GenerateAnswerRequest,
    request: Request,
    stream: bool = True,
    current_user: User = Depends(get_current_user),
    chroma=Depends(get_chroma),
    bm25=Depends(_get_bm25),
    embed_fn: Callable = Depends(_get_embed_fn),
    db: Session = Depends(get_db),
):
    """Agentic RAG: plan multi-hop retrieval → grounded generation → verification.

    ?stream=true (default) streams `step` events live, then answer `token`s, then
    a final `verification` event. ?stream=false returns the whole payload as JSON.
    The global STREAM_ENABLED kill switch forces JSON for everyone when false.
    """
    settings = get_settings()
    use_stream = stream and settings.stream_enabled

    # Same heuristic classifiers as /generate/answer — they shape the FINAL
    # grounded prompt (response style + task instructions), not the planning loop.
    query_type = classify_query(payload.query)
    answer_task = classify_task(payload.query)

    llm_fn: Callable[[str], str] | None = getattr(request.app.state, "llm_fn", None)
    verify_fn: Callable[[str], str] | None = (
        getattr(request.app.state, "verify_fn", None) or llm_fn
    )

    if llm_fn is None:
        return _not_configured(payload.query, query_type.value)

    allowed = set(settings.agent_allowed_tools_list) or None
    # Give the planner the user's REAL documents so fetch_document gets a valid id
    # instead of an invented placeholder (which silently returns 0 chunks).
    from app.models.document import Document, DocumentStatus
    _dq = db.query(Document).filter(
        Document.user_id == current_user.id, Document.status == DocumentStatus.ready
    )
    if payload.doc_id:
        _dq = _dq.filter(Document.id == payload.doc_id)
    available_docs = [
        {"doc_id": d.id, "name": d.original_filename}
        for d in _dq.order_by(Document.created_at.desc()).limit(25).all()
    ]
    tools = build_tools(
        current_user.id, chroma, bm25, embed_fn, doc_id=payload.doc_id,
        available_docs=available_docs,
    )

    # ── Observability root (driven manually so the streaming generator, which runs
    # after this coroutine returns, can close it — mirrors search.py) ─────────────
    root_ctx = span(
        "agent.request",
        user=hash_user_id(current_user.id),
        doc_id=payload.doc_id,
        top_k=payload.top_k,
        stream=use_stream,
        query_type=query_type.value,
        answer_task=answer_task.value,
        max_steps=settings.agent_max_steps,
        **({"query": payload.query} if capture_text_enabled() else {}),
    )
    root = root_ctx.__enter__()
    root_trace_id = current_trace_id()

    gen_model_name = getattr(request.app.state, "gen_model_name", None)

    def _new_state() -> AgentState:
        return AgentState(
            question=payload.query,
            user_id=current_user.id,
            doc_id=payload.doc_id,
            top_k=payload.top_k,
            available_docs=available_docs,
        )

    def _finalize(state: AgentState, answer_draft: str) -> AgentAnswerResponse:
        """Reuse /generate/answer's verification + build the agent response."""
        citations = _citations(state.collected)
        with span("verify") as vsp:
            base = _build_answer_response(
                payload.query, state.collected, citations, answer_draft, verify_fn,
                query_type.value, _AGENT_SCOPE, answer_task,
            )
            _attach_verify_scores(vsp, base)
        root.update(abstained=bool(base.abstained), steps=state.steps)
        return AgentAnswerResponse(
            **base.model_dump(),
            agent_steps=[r.to_event() for r in state.history],
        )

    # ── Non-streaming path ─────────────────────────────────────────────────────
    if not use_stream:
        state = _new_state()
        with span("agent", question_len=len(payload.query)) as asp:
            for _ in agent_events(
                state, tools, llm_fn,
                max_steps=settings.agent_max_steps, allowed_tools=allowed,
            ):
                pass
            asp.update(steps=state.steps, collected=len(state.collected))

        if not state.collected:
            root_ctx.__exit__(None, None, None)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No relevant content found. Upload documents and ensure they are processed.",
            )

        thin_context = len(state.collected) < 2
        try:
            answer_draft = generate_answer(
                payload.query, state.collected, llm_fn=llm_fn,
                doc_chunks_fetcher=None,  # agent already gathered the evidence set
                query_type=query_type, thin_context=thin_context,
                answer_task=answer_task, model_name=gen_model_name,
            )
        except GenerationError as exc:
            logger.warning("Agent generation failed: %s", exc)
            root.update(error="generation_failed")
            root_ctx.__exit__(None, None, None)
            base = _generation_error_response(
                payload.query, _citations(state.collected),
                "generation_failed", query_type.value,
            )
            return AgentAnswerResponse(
                **base.model_dump(),
                agent_steps=[r.to_event() for r in state.history],
            )

        response = _finalize(state, answer_draft)
        root_ctx.__exit__(None, None, None)
        return response

    # ── Streaming path (SSE) ───────────────────────────────────────────────────
    stream_fn: Callable[[str], object] | None = getattr(
        request.app.state, "llm_stream_fn", None
    )
    # Close the root's contextvar scope here; the generator re-adopts it as parent.
    root_ctx.__exit__(None, None, None)

    async def event_generator():
        with adopt(root, root_trace_id):
            state = _new_state()

            # PHASE 1 — plan + retrieve, emitting each step LIVE.
            with span("agent", question_len=len(payload.query)) as asp:
                for rec in agent_events(
                    state, tools, llm_fn,
                    max_steps=settings.agent_max_steps, allowed_tools=allowed,
                ):
                    yield sse_event("step", rec.to_event())
                asp.update(steps=state.steps, collected=len(state.collected))

            if not state.collected:
                # No evidence even after fallback — close honestly, don't 500.
                yield sse_event("token", {"text": "", "done": True, "full_answer": ""})
                err = AgentAnswerResponse(
                    **_generation_error_response(
                        payload.query, [], "no_relevant_content", query_type.value
                    ).model_dump(),
                    agent_steps=[r.to_event() for r in state.history],
                )
                yield sse_event("verification", err.model_dump())
                return

            # PHASE 2 — stream the grounded answer tokens.
            thin_context = len(state.collected) < 2
            with span("prompt_build") as psp:
                prompt = build_grounded_prompt(
                    payload.query, state.collected, doc_chunks_fetcher=None,
                    query_type=query_type, thin_context=thin_context,
                    answer_task=answer_task,
                )
                psp.update(source_count=len(state.collected))
                if capture_text_enabled():
                    psp.update(prompt=prompt)

            full_answer = ""
            try:
                started = time.perf_counter()
                first = True
                with span("generate", model=gen_model_name) as gsp:
                    for token in stream_generation(prompt, stream_fn, llm_fn):
                        if first:
                            first = False
                            gsp.update(
                                ttft_ms=round((time.perf_counter() - started) * 1000, 1)
                            )
                        full_answer += token
                        yield sse_event("token", {"text": token, "done": False})
            except GenerationError as exc:
                logger.warning("Agent streaming generation failed: %s", exc)
                root.update(error="generation_failed")
                yield sse_event("token", {"text": "", "done": True, "full_answer": ""})
                err = AgentAnswerResponse(
                    **_generation_error_response(
                        payload.query, _citations(state.collected),
                        "generation_failed", query_type.value,
                    ).model_dump(),
                    agent_steps=[r.to_event() for r in state.history],
                )
                yield sse_event("verification", err.model_dump())
                return

            full_answer = full_answer.strip()
            yield sse_event("token", {"text": "", "done": True, "full_answer": full_answer})

            # PHASE 3 — verify the whole answer and emit the final payload.
            response = _finalize(state, full_answer)
            yield sse_event("verification", response.model_dump())

    return StreamingResponse(event_generator(), media_type="text/event-stream")
