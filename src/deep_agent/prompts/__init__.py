"""System prompts and tool descriptions."""

from deep_agent.prompts.orchestrator import ORCHESTRATOR_SYSTEM_PROMPT
from deep_agent.prompts.subagents import (
    CODER_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
    WRITER_SYSTEM_PROMPT,
)

__all__ = [
    "ORCHESTRATOR_SYSTEM_PROMPT",
    "PLANNER_SYSTEM_PROMPT",
    "RESEARCHER_SYSTEM_PROMPT",
    "WRITER_SYSTEM_PROMPT",
    "CODER_SYSTEM_PROMPT",
]
