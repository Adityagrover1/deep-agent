"""Agent factory and built-in sub-agent presets."""

from deep_agent.agents.factory import create_deep_agent
from deep_agent.agents.subagents import (
    CODER_SUBAGENT,
    DEFAULT_SUBAGENTS,
    PLANNER_SUBAGENT,
    RESEARCH_SUBAGENT,
    WRITER_SUBAGENT,
)

__all__ = [
    "create_deep_agent",
    "DEFAULT_SUBAGENTS",
    "PLANNER_SUBAGENT",
    "RESEARCH_SUBAGENT",
    "WRITER_SUBAGENT",
    "CODER_SUBAGENT",
]
