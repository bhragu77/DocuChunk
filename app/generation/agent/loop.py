"""
The agent loop — an EXPLICIT while-loop (LangGraph's router/tool/llm nodes made
plain), driven as a GENERATOR so both request paths consume it identically:

  * non-streaming: `for _ in agent_events(...): pass`, then read state.collected.
  * streaming:     `for rec in agent_events(...): yield sse_event("step", ...)`.

Each yielded StepRecord is one executed step. The generator MUTATES `state` in
place (collected chunks, history, step count); the caller reads it afterwards.

GUARDRAILS (all four from the findings, plus what this system needs):
  * MAX_STEPS               — hard cap on iterations.
  * allowed-tool check      — a hallucinated tool is a recorded error step, never
                              a crash, and the loop continues (bounded by the cap).
  * arg validation          — missing/empty required args → recorded error step.
  * loop detection          — the same (tool, args) twice, or a step that adds no
                              new chunks after progress, breaks out.
  * fallback-to-RAG         — if the loop ends with zero collected chunks (model
                              said finish immediately / only errored), run one
                              plain retrieve_docs(question) so the agent is NEVER
                              worse than the existing single-shot RAG retrieval.

Tracing: each step is a `span("agent.step")`; the planner call a `span`; tools
open their own spans (see tools.py). The caller wraps consumption in
`span("agent")`, mirroring how search.py wraps its streaming generate loop.
"""
from __future__ import annotations

import logging
from typing import Callable, Iterator

from app.generation.agent.react import FINISH, build_planner_prompt, parse_action
from app.generation.agent.state import AgentState, StepRecord
from app.generation.agent.tools import Tool
from app.generation.base import GenerationError
from app.observability import span

logger = logging.getLogger(__name__)


def _canonical_args(args: dict) -> tuple:
    """A hashable, order-independent signature of an args dict for loop detection."""
    return tuple(sorted((str(k), str(v)) for k, v in args.items()))


def _required_missing(tool: Tool, args: dict) -> str | None:
    """Return the name of the first required arg that is missing/empty, else None."""
    for name in (tool.parameters or {}).get("required", []):
        val = args.get(name)
        if val is None or (isinstance(val, str) and not val.strip()):
            return name
    return None


def agent_events(
    state: AgentState,
    tools: dict[str, Tool],
    llm_fn: Callable[[str], str],
    *,
    max_steps: int = 5,
    allowed_tools: set[str] | None = None,
) -> Iterator[StepRecord]:
    """Run the ReAct planning loop, yielding one StepRecord per executed step.

    `allowed_tools` (optional) further restricts which registry tools may run
    (AGENT_ALLOWED_TOOLS); None means "every tool in `tools`". `finish` is always
    allowed. The generator ends by yielding a synthetic fallback step if nothing
    was collected.
    """
    allowed = allowed_tools if allowed_tools is not None else set(tools)
    seen_calls: set[tuple[str, tuple]] = set()

    while state.steps < max_steps:
        n = state.steps + 1
        remaining = max_steps - state.steps
        terminate = False

        with span("agent.step", n=n) as ssp:
            plan_error: str | None = None
            with span("agent.plan"):
                prompt = build_planner_prompt(state, tools, remaining)
                try:
                    raw = llm_fn(prompt)
                except GenerationError as exc:
                    # The planner MODEL call failed (timeout on a slow local LLM, a
                    # cloud 504, a rate limit). This must not escape the generator:
                    # if it does, the fallback-to-RAG below never runs and the whole
                    # request returns zero evidence — strictly WORSE than single-shot
                    # RAG, which is exactly what this loop promises never to be.
                    # Record it as an error step and stop planning instead.
                    plan_error = str(exc)
                    raw = ""

            if plan_error is not None:
                rec = _error_step(
                    state, n, "planner", {}, f"planner call failed: {plan_error}"
                )
                ssp.update(kind="error", tool="planner")
                terminate = True
            else:
                kind, name, args = parse_action(raw)
                ssp.update(kind=kind, tool=name or "")

                if kind == "finish":
                    logger.info("agent: planner finished at step %d", n)
                    break

                if kind == "error":
                    rec = _error_step(state, n, "?", {}, args.get("reason", "parse error"))
                    # A parse error on the very first step usually means the model
                    # ignored the format — stop planning and let the fallback retrieve.
                    terminate = not state.history[:-1]
                elif name not in tools:
                    rec = _error_step(state, n, name, args, f"unknown tool '{name}'")
                elif name not in allowed:
                    rec = _error_step(state, n, name, args, f"tool '{name}' not permitted")
                else:
                    tool = tools[name]
                    missing = _required_missing(tool, args)
                    if missing:
                        rec = _error_step(
                            state, n, name, args, f"missing required arg '{missing}'"
                        )
                    else:
                        sig = (name, _canonical_args(args))
                        if sig in seen_calls:
                            rec = _error_step(
                                state, n, name, args, "repeated identical call — stopping"
                            )
                            terminate = True
                        else:
                            seen_calls.add(sig)
                            rec = _run_tool(state, n, tool, args)
                            # No-progress guard: a successful call that adds nothing new
                            # (after we already have evidence) means we're circling.
                            if (
                                rec.status == "ok"
                                and rec.data.get("_new_chunks") == 0
                                and len(state.collected) > 0
                            ):
                                terminate = True

            ssp.update(status=rec.status)

        state.history.append(rec)
        state.steps = n
        yield rec
        if terminate:
            break

    # ── Fallback-to-RAG — never worse than single-shot retrieval ───────────────
    if not state.collected and "retrieve_docs" in tools:
        n = state.steps + 1
        with span("agent.step", n=n, fallback=True):
            rec = _run_tool(
                state, n, tools["retrieve_docs"],
                {"query": state.question, "top_k": state.top_k},
                fallback=True,
            )
        state.history.append(rec)
        state.steps = n
        yield rec


def run_agent(
    state: AgentState,
    tools: dict[str, Tool],
    llm_fn: Callable[[str], str],
    *,
    max_steps: int = 5,
    allowed_tools: set[str] | None = None,
) -> AgentState:
    """Drive the loop to completion (non-streaming). Returns the mutated state."""
    with span("agent", question_len=len(state.question or "")) as asp:
        for _ in agent_events(
            state, tools, llm_fn, max_steps=max_steps, allowed_tools=allowed_tools
        ):
            pass
        asp.update(steps=state.steps, collected=len(state.collected))
    return state


# ── Step builders ─────────────────────────────────────────────────────────────

def _run_tool(
    state: AgentState, n: int, tool: Tool, args: dict, *, fallback: bool = False
) -> StepRecord:
    """Execute a tool, accumulate its chunks, and build the StepRecord."""
    result = tool.func(**args)
    added = state.add_chunks(result.chunks)
    data = dict(result.data)
    data["_new_chunks"] = added
    summary = result.summary + (f" (+{added} new)" if result.chunks else "")
    if fallback:
        summary = "[fallback] " + summary
    return StepRecord(
        n=n,
        tool=tool.name,
        args=args,
        status="ok",
        summary=summary,
        observation=result.observation,
        data=data,
    )


def _error_step(
    state: AgentState, n: int, tool: str, args: dict, reason: str
) -> StepRecord:
    logger.info("agent: step %d error (%s): %s", n, tool, reason)
    return StepRecord(
        n=n, tool=tool, args=args, status="error",
        summary=f"{tool}: {reason}", error=reason,
    )
