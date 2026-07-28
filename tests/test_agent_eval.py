"""
Phase D — agent-vs-RAG comparison harness tests.

Fast unit tests over the offline planner + decomposer + report formatter, plus
ONE full offline integration run that ingests the fixture and asserts the core
invariant: the agent NEVER loses context-recall vs single-shot RAG.
"""
import json

from app.eval import agent_harness as ah
from app.eval.gen_harness import _METRIC_KEYS


# ── Deterministic decomposer ──────────────────────────────────────────────────

def test_decompose_splits_compound_questions():
    subs = ah.decompose("Who is the CEO and what is the revenue?")
    assert subs == ["Who is the CEO", "what is the revenue"]


def test_decompose_keeps_simple_question_single():
    assert ah.decompose("What replacement battery does the GX-4200 use?") == \
        ["What replacement battery does the GX-4200 use"]


def test_decompose_caps_sub_queries():
    subs = ah.decompose("alpha and beta and gamma and delta and epsilon", max_sub=3)
    assert subs == ["alpha", "beta", "gamma"]


def test_decompose_dedups():
    subs = ah.decompose("battery and battery")
    assert subs == ["battery"]


# ── Offline planner emits actions then finish ─────────────────────────────────

def test_offline_planner_emits_retrieve_then_finish():
    planner = ah.make_offline_planner("Who is CEO and the revenue?", k=7)
    a1 = json.loads(planner("prompt"))
    a2 = json.loads(planner("prompt"))
    a3 = json.loads(planner("prompt"))

    assert a1 == {"tool": "retrieve_docs", "args": {"query": "Who is CEO", "top_k": 7}}
    assert a2 == {"tool": "retrieve_docs", "args": {"query": "the revenue", "top_k": 7}}
    assert a3 == {"tool": "finish", "args": {}}
    # Exhausted → keeps returning finish (loop terminator).
    assert json.loads(planner("prompt")) == {"tool": "finish", "args": {}}


def test_offline_planner_single_hop_for_simple_question():
    planner = ah.make_offline_planner("plain question about batteries", k=5)
    first = json.loads(planner("p"))
    assert first["tool"] == "retrieve_docs"
    assert json.loads(planner("p"))["tool"] == "finish"


# ── Multi-hop accumulation through the real loop (fake tool, no Chroma) ────────

def test_offline_planner_drives_multi_hop_accumulation():
    from app.generation.agent import AgentState, Tool, ToolResult, run_agent
    from app.pipeline.retrieval import ScoredChunk

    def _chunk(cid):
        return ScoredChunk(chunk_id=cid, text="t", doc_id="d", source="s",
                           page_number=1, fused_score=1.0)

    # distinct chunk per sub-query → the two hops must accumulate to 2 chunks.
    by_query = {"who is the ceo": [_chunk("c1")], "the revenue": [_chunk("c2")]}

    def func(query, top_k=5):
        return ToolResult(status="success", source="s", summary="s", observation="o",
                          chunks=by_query.get(query.lower().strip(), []))

    tool = Tool(name="retrieve_docs", description="d",
                parameters={"type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"]},
                func=func)

    planner = ah.make_offline_planner("who is the ceo and the revenue?", k=5)
    state = AgentState(question="who is the ceo and the revenue?", user_id="u")
    run_agent(state, {"retrieve_docs": tool}, planner, max_steps=5)

    assert state.steps == 2
    assert sorted(c.chunk_id for c in state.collected) == ["c1", "c2"]


# ── Report formatter ──────────────────────────────────────────────────────────

def _mini_comparison() -> dict:
    metrics = {"faithfulness": 1.0, "answer_relevancy": 0.8,
               "context_precision": 0.9, "context_recall": 1.0,
               "answer_correctness": 0.5}
    fb = {c: 0 for c in ("retrieval_miss", "over_refusal", "hallucination",
                         "off_topic", "partial_answer", "ok")}
    return {
        "harness": "agent_vs_rag", "k": 5, "max_steps": 5, "profile": "surrogate",
        "generator": "extractive_surrogate", "judge": "lexical_surrogate",
        "planner": "offline_decomposer",
        "embedding_provider": {"name": "local", "model": "m", "semantic": True},
        "note": "surrogate note", "num_queries": 33,
        "rag": {"aggregate": dict(metrics), "num_abstained": 0,
                "failure_breakdown": dict(fb), "failure_rates": {}},
        "agent": {"aggregate": {**metrics, "context_recall": 1.0}, "num_abstained": 0,
                  "failure_breakdown": dict(fb), "failure_rates": {}},
        "deltas": {m: 0.0 for m in _METRIC_KEYS},
        "agent_trajectory": {"avg_steps": 1.4, "avg_tool_calls": 1.4,
                             "avg_chunks_gathered": 6.2, "multi_hop_queries": 5,
                             "fallback_queries": 0, "hit_max_steps": 0},
    }


def test_format_comparison_report_sections():
    md = ah.format_comparison_report(_mini_comparison())
    assert "Agent vs RAG comparison" in md
    assert "Metric comparison" in md
    assert "Agent trajectory" in md
    assert "context-recall" in md
    assert "Multi-hop queries" in md


# ── Full offline integration run (ingests the fixture) ────────────────────────

def test_run_agent_comparison_offline():
    report = ah.run_agent_comparison(k=5)  # offline → deterministic

    # Derived from the fixture, not hardcoded — growing the eval set is not a failure.
    n_queries = len(ah.load_eval_set(ah.DEFAULT_EVAL_SET)["queries"])

    assert report["harness"] == "agent_vs_rag"
    assert report["profile"] == "surrogate"
    assert report["num_queries"] == n_queries
    assert set(report["deltas"]) == set(_METRIC_KEYS)

    # The multi_hop slice is the reason the agent exists — it must be scored, and the
    # planner must actually take >1 step on those compound queries.
    assert "multi_hop" in report["agent"]["by_category"]
    assert report["agent_trajectory"]["multi_hop_queries"] > 0

    # CORE INVARIANT: the agent gathers a superset, so it never loses recall.
    assert report["deltas"]["context_recall"] >= 0.0
    assert report["agent"]["aggregate"]["context_recall"] >= 0.9

    # Trajectory stats are present and sane.
    traj = report["agent_trajectory"]
    assert traj["avg_steps"] >= 1.0
    assert traj["avg_chunks_gathered"] >= 1.0

    # Extractive generation never hallucinates on either arm.
    assert report["agent"]["failure_breakdown"]["hallucination"] == 0
    assert report["rag"]["failure_breakdown"]["hallucination"] == 0
