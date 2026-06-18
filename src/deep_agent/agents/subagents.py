"""Built-in sub-agent presets.

Each preset is a `SubAgentConfig` referencing tools by name. The names are resolved against
the orchestrator's master tool registry in `create_task_tool`.
"""

from deep_agent.config import SubAgentConfig
from deep_agent.prompts.subagents import (
    CODER_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    RESEARCHER_SYSTEM_PROMPT,
    WRITER_SYSTEM_PROMPT,
)

PLANNER_SUBAGENT: SubAgentConfig = {
    "name": "planner",
    "description": "Decomposes a task into a structured, ordered list of executable steps.",
    "prompt": PLANNER_SYSTEM_PROMPT,
    "tools": ["think_tool"],
}

RESEARCH_SUBAGENT: SubAgentConfig = {
    "name": "researcher",
    "description": "Searches the web and synthesizes information into findings files.",
    "prompt": RESEARCHER_SYSTEM_PROMPT,
    "tools": ["tavily_search", "think_tool", "read_file", "write_file", "ls"],
}

WRITER_SUBAGENT: SubAgentConfig = {
    "name": "writer",
    "description": "Writes structured documents, reports, and summaries from source files.",
    "prompt": WRITER_SYSTEM_PROMPT,
    "tools": ["read_file", "write_file", "ls", "think_tool"],
}

CODER_SUBAGENT: SubAgentConfig = {
    "name": "coder",
    "description": "Writes, edits, and analyzes code in the virtual filesystem. Also executes Python for math and verification.",
    "tools": ["read_file", "write_file", "edit_file", "ls", "think_tool", "python_exec"],
    "prompt": CODER_SYSTEM_PROMPT,
}

DEFAULT_SUBAGENTS: list[SubAgentConfig] = [
    PLANNER_SUBAGENT,
    RESEARCH_SUBAGENT,
    WRITER_SUBAGENT,
    CODER_SUBAGENT,
]
