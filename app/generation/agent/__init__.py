"""
Agentic retrieval layer — a ReAct planning loop that lets the model decide WHEN
and HOW MANY TIMES to retrieve before the existing grounded-generation path
answers. See loop.py for the control flow and tools.py for the (reused) tools.

Public surface:
    from app.generation.agent import AgentState, build_tools, agent_events, run_agent
"""
from app.generation.agent.loop import agent_events, run_agent
from app.generation.agent.react import FINISH, build_planner_prompt, parse_action
from app.generation.agent.state import AgentState, StepRecord
from app.generation.agent.tools import Tool, ToolResult, build_tools

__all__ = [
    "AgentState",
    "StepRecord",
    "Tool",
    "ToolResult",
    "build_tools",
    "agent_events",
    "run_agent",
    "build_planner_prompt",
    "parse_action",
    "FINISH",
]
