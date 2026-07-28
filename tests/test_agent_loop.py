"""
Agent loop control + guardrails — pure, hermetic unit tests.

No Chroma, no BM25, no real LLM: the planner is a scripted string→string
callable and the tools are fakes returning canned ScoredChunks. This isolates
the LOOP LOGIC (multi-hop, max-steps, loop detection, hallucinated tools, empty
args, allowed-tools, no-progress, fallback-to-RAG) from retrieval.
"""
import json

from app.generation.agent import (
    AgentState,
    Tool,
    ToolResult,
    agent_events,
    parse_action,
    run_agent,
)
from app.generation.base import GenerationError
from app.pipeline.retrieval import ScoredChunk


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk(cid: str, text: str = "text") -> ScoredChunk:
    return ScoredChunk(
        chunk_id=cid, text=text, doc_id="d1", source="s.pdf",
        page_number=1, fused_score=1.0,
    )


def _retrieve_tool(returns) -> Tool:
    """A fake retrieve_docs. `returns(query, top_k) -> list[ScoredChunk]`."""
    def func(query: str, top_k: int = 5) -> ToolResult:
        chunks = returns(query, top_k)
        return ToolResult(
            status="success" if chunks else "no_results",
            source="hybrid_search",
            summary=f'retrieve_docs("{query}") → {len(chunks)}',
            observation="obs",
            chunks=chunks,
            data={"count": len(chunks)},
        )
    return Tool(
        name="retrieve_docs", description="search",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
            "required": ["query"],
        },
        func=func,
    )


def _fetch_tool() -> Tool:
    def func(doc_id: str) -> ToolResult:
        return ToolResult(
            status="success", source="whole_document",
            summary=f"fetch_document({doc_id})", observation="obs",
            chunks=[_chunk(f"{doc_id}-c1")], data={},
        )
    return Tool(
        name="fetch_document", description="load doc",
        parameters={
            "type": "object",
            "properties": {"doc_id": {"type": "string"}},
            "required": ["doc_id"],
        },
        func=func,
    )


class Script:
    """A scripted planner: returns queued responses, then finish forever."""
    def __init__(self, *responses):
        self.responses = list(responses)
        self.i = 0
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        r = self.responses[self.i] if self.i < len(self.responses) else _finish()
        self.i += 1
        return r


def _act(tool: str, **args) -> str:
    return json.dumps({"tool": tool, "args": args})


def _finish() -> str:
    return json.dumps({"tool": "finish", "args": {}})


def _state(question="q", top_k=8) -> AgentState:
    return AgentState(question=question, user_id="u1", top_k=top_k)


# ── Multi-hop: distinct sub-queries accumulate a deduped evidence set ──────────

def test_multi_hop_accumulates_distinct_chunks():
    by_query = {"a": [_chunk("c1"), _chunk("c2")], "b": [_chunk("c3")]}
    tools = {"retrieve_docs": _retrieve_tool(lambda q, k: by_query.get(q, []))}
    llm = Script(_act("retrieve_docs", query="a"), _act("retrieve_docs", query="b"), _finish())

    state = run_agent(_state(), tools, llm, max_steps=5)

    assert state.steps == 2
    assert [c.chunk_id for c in state.collected] == ["c1", "c2", "c3"]
    assert [r.status for r in state.history] == ["ok", "ok"]


# ── MAX_STEPS caps an otherwise non-terminating planner ───────────────────────

def test_max_steps_caps_iteration():
    # A new unique chunk every call so the no-progress guard never fires, and a
    # new unique query every call so loop-detection never fires — only the cap stops it.
    n = {"i": 0}

    def returns(q, k):
        n["i"] += 1
        return [_chunk(f"c{n['i']}")]

    def planner(prompt):  # always asks to retrieve something new
        return _act("retrieve_docs", query=f"q{n['i']}")

    tools = {"retrieve_docs": _retrieve_tool(returns)}
    state = run_agent(_state(), tools, planner, max_steps=3)

    assert state.steps == 3
    assert len(state.history) == 3


# ── Loop detection: an identical repeated call stops the loop ─────────────────

def test_loop_detection_breaks_on_repeat():
    tools = {"retrieve_docs": _retrieve_tool(lambda q, k: [_chunk("c1")])}
    llm = Script(_act("retrieve_docs", query="same"), _act("retrieve_docs", query="same"))

    state = run_agent(_state(), tools, llm, max_steps=5)

    assert state.steps == 2
    assert state.history[0].status == "ok"
    assert state.history[1].status == "error"
    assert "repeated" in state.history[1].error


# ── No-progress guard: a step that adds no new chunks stops the loop ──────────

def test_no_progress_guard_breaks():
    # Same chunk regardless of query → 2nd distinct query adds nothing new.
    tools = {"retrieve_docs": _retrieve_tool(lambda q, k: [_chunk("dup")])}
    llm = Script(_act("retrieve_docs", query="a"), _act("retrieve_docs", query="b"), _finish())

    state = run_agent(_state(), tools, llm, max_steps=5)

    assert state.steps == 2
    assert [c.chunk_id for c in state.collected] == ["dup"]


# ── Hallucinated tool → recorded error step, loop continues ───────────────────

def test_hallucinated_tool_is_error_not_crash():
    tools = {"retrieve_docs": _retrieve_tool(lambda q, k: [_chunk("c1")])}
    llm = Script(_act("get_warranty_info", product="x"),
                 _act("retrieve_docs", query="a"), _finish())

    state = run_agent(_state(), tools, llm, max_steps=5)

    assert state.history[0].status == "error"
    assert "unknown tool" in state.history[0].error
    assert state.history[1].status == "ok"
    assert [c.chunk_id for c in state.collected] == ["c1"]


# ── Empty/missing required arg → recorded error step ──────────────────────────

def test_empty_required_arg_rejected():
    tools = {"retrieve_docs": _retrieve_tool(lambda q, k: [_chunk("c1")])}
    llm = Script(_act("retrieve_docs", query="   "),
                 _act("retrieve_docs", query="real"), _finish())

    state = run_agent(_state(), tools, llm, max_steps=5)

    assert state.history[0].status == "error"
    assert "missing required arg 'query'" in state.history[0].error
    assert state.history[1].status == "ok"


# ── allowed_tools restricts the registry ──────────────────────────────────────

def test_allowed_tools_blocks_disallowed():
    tools = {
        "retrieve_docs": _retrieve_tool(lambda q, k: [_chunk("c1")]),
        "fetch_document": _fetch_tool(),
    }
    llm = Script(_act("fetch_document", doc_id="d1"),
                 _act("retrieve_docs", query="a"), _finish())

    state = run_agent(_state(), tools, llm, max_steps=5, allowed_tools={"retrieve_docs"})

    assert state.history[0].status == "error"
    assert "not permitted" in state.history[0].error
    assert state.history[1].status == "ok"


# ── Fallback-to-RAG: finish with no evidence still retrieves once ─────────────

def test_fallback_retrieves_when_nothing_collected():
    tools = {"retrieve_docs": _retrieve_tool(lambda q, k: [_chunk("fb1")])}
    llm = Script(_finish())  # planner gives up immediately

    state = run_agent(_state(question="the real question"), tools, llm, max_steps=5)

    assert len(state.history) == 1
    assert state.history[0].summary.startswith("[fallback]")
    assert [c.chunk_id for c in state.collected] == ["fb1"]


def test_no_fallback_when_evidence_already_gathered():
    tools = {"retrieve_docs": _retrieve_tool(lambda q, k: [_chunk("c1")])}
    llm = Script(_act("retrieve_docs", query="a"), _finish())

    state = run_agent(_state(), tools, llm, max_steps=5)

    # Exactly one executed step, no synthetic fallback appended.
    assert len(state.history) == 1
    assert not state.history[0].summary.startswith("[fallback]")


# ── Parse error on the very first step → terminate, then fallback ─────────────

def test_unparseable_first_step_falls_back():
    tools = {"retrieve_docs": _retrieve_tool(lambda q, k: [_chunk("fb1")])}
    llm = Script("I refuse to output JSON")

    state = run_agent(_state(), tools, llm, max_steps=5)

    assert state.history[0].status == "error"
    assert state.history[-1].summary.startswith("[fallback]")
    assert [c.chunk_id for c in state.collected] == ["fb1"]


# ── Planner LLM failure → recorded error step, then fallback (never zero evidence) ─

def test_planner_timeout_still_falls_back_to_rag():
    """A planner call that RAISES (Ollama timeout, cloud 504) must not escape the
    generator. Before the fix it propagated out of agent_events, skipping the
    fallback entirely, so the whole request returned no evidence at all — strictly
    worse than single-shot RAG."""
    tools = {"retrieve_docs": _retrieve_tool(lambda q, k: [_chunk("fb1")])}

    def boom(prompt: str) -> str:
        raise GenerationError("OpenAI-compatible generation failed: timed out")

    state = run_agent(_state(question="the real question"), tools, boom, max_steps=5)

    assert state.history[0].status == "error"
    assert "planner call failed" in state.history[0].error
    # The fallback ran and gathered evidence despite the planner being dead.
    assert state.history[-1].summary.startswith("[fallback]")
    assert [c.chunk_id for c in state.collected] == ["fb1"]


def test_planner_failure_midway_keeps_evidence_already_gathered():
    """A planner failure on step 2 stops planning but preserves step 1's evidence,
    and adds no fallback (we already have chunks)."""
    tools = {"retrieve_docs": _retrieve_tool(lambda q, k: [_chunk("c1")])}
    calls = {"n": 0}

    def flaky(prompt: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return _act("retrieve_docs", query="a")
        raise GenerationError("504 DEADLINE_EXCEEDED")

    state = run_agent(_state(), tools, flaky, max_steps=5)

    assert [c.chunk_id for c in state.collected] == ["c1"]
    assert state.history[-1].status == "error"
    assert not any(h.summary.startswith("[fallback]") for h in state.history)


# ── parse_action contract (tolerant extraction) ───────────────────────────────

def test_parse_action_forms():
    assert parse_action(_act("retrieve_docs", query="x")) == ("call", "retrieve_docs", {"query": "x"})
    assert parse_action('```json\n{"tool":"finish","args":{}}\n```')[0] == "finish"
    # Flattened args (no wrapper) still parse.
    assert parse_action('{"tool":"retrieve_docs","query":"y"}') == ("call", "retrieve_docs", {"query": "y"})
    assert parse_action("not json")[0] == "error"
    assert parse_action('{"args":{"query":"z"}}')[0] == "error"  # no tool name
