"""
ReAct driver — the provider-agnostic half of the loop (Option A from the plan).

No provider protocol change is needed: the planner is just a PROMPT the existing
str→str `llm_fn` answers with a single JSON action. This module owns

  * build_planner_prompt(state, tools, remaining) — renders the tool menu, the
    question, and the evidence gathered so far, and asks for ONE JSON action.
  * parse_action(text) — a tolerant parser that pulls the JSON action back out of
    whatever the model returned (fenced block, prose-wrapped, or bare object).

When a native function-calling provider is added later (Phase E), it slots in
behind parse_action's return contract without touching loop.py.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from app.generation.agent.state import AgentState
from app.generation.agent.tools import Tool

logger = logging.getLogger(__name__)

# The pseudo-tool the planner emits when the gathered evidence is sufficient. It
# is NOT in the registry; the loop handles it as a terminal signal.
FINISH = "finish"


def _render_tool_menu(tools: dict[str, Tool]) -> str:
    lines = []
    for t in tools.values():
        props = (t.parameters or {}).get("properties", {})
        req = set((t.parameters or {}).get("required", []))
        sig = ", ".join(
            f"{name}: {spec.get('type', 'string')}"
            + ("" if name in req else "?")
            for name, spec in props.items()
        )
        lines.append(f'- {t.name}({sig})\n    {t.description}')
    lines.append(
        f"- {FINISH}()\n    Stop retrieving. Call this as soon as the evidence "
        "gathered is enough to answer the question."
    )
    return "\n".join(lines)


def _render_evidence(state: AgentState) -> str:
    if not state.history:
        return "(nothing retrieved yet)"
    blocks = []
    for rec in state.history:
        if rec.status == "error":
            blocks.append(f"Step {rec.n}: {rec.summary} — ERROR: {rec.error}")
        else:
            blocks.append(f"Step {rec.n}: {rec.observation}")
    return "\n".join(blocks)


def _render_documents(state: AgentState) -> str:
    """List the user's real documents so the planner passes a VALID doc_id to
    fetch_document instead of inventing one (the cause of empty whole-doc fetches)."""
    docs = state.available_docs or []
    if not docs:
        return ""
    lines = [f'  - doc_id="{d.get("doc_id")}" — {d.get("name") or "document"}' for d in docs[:25]]
    note = (
        " There is exactly ONE document — use its doc_id."
        if len(docs) == 1 else
        " Use one of these EXACT doc_id values — never invent one."
    )
    return (
        "AVAILABLE DOCUMENTS (for fetch_document):" + note + "\n"
        + "\n".join(lines) + "\n\n"
    )


def build_planner_prompt(
    state: AgentState, tools: dict[str, Tool], remaining: int
) -> str:
    """Render the planner prompt for the next step. Pure string assembly."""
    return (
        "You are a RETRIEVAL PLANNING agent for a document question-answering "
        "system. Your job is NOT to answer the question yourself — a separate "
        "grounded-answer step does that. Your ONLY job is to decide what to "
        "RETRIEVE so that step has all the evidence it needs.\n\n"
        f"QUESTION:\n{state.question}\n\n"
        f"{_render_documents(state)}"
        "TOOLS (choose exactly one):\n"
        f"{_render_tool_menu(tools)}\n\n"
        "EVIDENCE GATHERED SO FAR:\n"
        f"{_render_evidence(state)}\n\n"
        "RULES:\n"
        "- Respond with EXACTLY ONE JSON object and nothing else.\n"
        '- Format: {"tool": "<name>", "args": { ... }}\n'
        "- Break a multi-part question into focused sub-queries across steps.\n"
        "- Do NOT repeat a query you already tried.\n"
        f'- Call {FINISH} (e.g. {{"tool": "{FINISH}", "args": {{}}}}) as soon as '
        "the evidence covers the question. Prefer fewer steps.\n"
        f"- You have at most {remaining} step(s) left.\n\n"
        "Your next action (JSON only):"
    )


# ── Tolerant action parsing ───────────────────────────────────────────────────

def _extract_json_object(text: str) -> dict | None:
    """Pull the first balanced {...} JSON object out of arbitrary model text.

    Handles: a bare object, a ```json fenced block, or an object embedded in
    prose. Returns the parsed dict, or None if no valid object is found.
    """
    if not text:
        return None
    s = text.strip()
    # Fast path: the whole thing is JSON.
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except (ValueError, TypeError):
        pass
    # Scan for the first balanced top-level object, respecting strings/escapes.
    start = s.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except (ValueError, TypeError):
                        break  # malformed; try the next '{'
        start = s.find("{", start + 1)
    return None


def parse_action(text: str) -> tuple[str, str | None, dict]:
    """Parse a planner response into a normalized action.

    Returns one of:
      ("call",   tool_name, args_dict)   — execute a registry tool
      ("finish", "finish",  args_dict)   — terminal: evidence is sufficient
      ("error",  None,      {"reason"})  — unparseable / malformed action

    Accepts both the nested form {"tool": t, "args": {...}} and a flattened form
    {"tool": t, "query": "..."} (small models often drop the args wrapper).
    """
    obj = _extract_json_object(text)
    if obj is None:
        return ("error", None, {"reason": "no JSON action found in model output"})

    name = obj.get("tool") or obj.get("action") or obj.get("name")
    if not name or not isinstance(name, str):
        return ("error", None, {"reason": "action missing a 'tool' name"})
    name = name.strip()

    args: Any = obj.get("args")
    if args is None:
        # Flattened form: everything except the tool key is treated as args.
        args = {k: v for k, v in obj.items() if k not in ("tool", "action", "name")}
    if not isinstance(args, dict):
        return ("error", None, {"reason": "'args' must be a JSON object"})

    if name.lower() == FINISH:
        return ("finish", FINISH, args)
    return ("call", name, args)
