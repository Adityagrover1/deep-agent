"""Generalised Deep Agent.

A configurable, production-oriented deep agent built on LangGraph: an orchestrator that
plans, delegates to specialized sub-agents, manages memory, recovers from errors, and can
pause for human input.

Public API:
    create_deep_agent(config) -> compiled LangGraph agent
    AgentConfig                -> top-level configuration
    SubAgentConfig             -> describe a custom sub-agent
    DeepAgentState             -> the shared state schema
    DEFAULT_SUBAGENTS          -> the four built-in presets
"""

from deep_agent.agents.factory import create_deep_agent
from deep_agent.agents.subagents import (
    CODER_SUBAGENT,
    DEFAULT_SUBAGENTS,
    PLANNER_SUBAGENT,
    RESEARCH_SUBAGENT,
    WRITER_SUBAGENT,
)
from deep_agent.config import AgentConfig, SubAgentConfig
from deep_agent.state import DeepAgentState, Todo

__all__ = [
    "create_deep_agent",
    "AgentConfig",
    "SubAgentConfig",
    "DeepAgentState",
    "Todo",
    "DEFAULT_SUBAGENTS",
    "PLANNER_SUBAGENT",
    "RESEARCH_SUBAGENT",
    "WRITER_SUBAGENT",
    "CODER_SUBAGENT",
]
