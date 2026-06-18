"""Configuration objects for the deep agent.

`SubAgentConfig` describes one specialized sub-agent (name, description, prompt, and the
string names of the tools it may use). `AgentConfig` is the single knob bag passed to
`create_deep_agent()` — models, limits, and feature flags.
"""

from dataclasses import dataclass, field
from typing import Literal

from typing_extensions import NotRequired, TypedDict


class SubAgentConfig(TypedDict):
    """Configuration for a specialized sub-agent.

    Attributes:
        name: Unique identifier used as `subagent_type` when delegating.
        description: One-line summary shown to the orchestrator so it can choose this agent.
        prompt: System prompt for the sub-agent.
        tools: Optional list of tool *names* (resolved against the master tool registry).
               If omitted, the sub-agent receives the full tool set.
    """

    name: str
    description: str
    prompt: str
    tools: NotRequired[list[str]]


@dataclass
class AgentConfig:
    """Top-level configuration for `create_deep_agent()`.

    Attributes:
        model: Model id for the orchestrator and sub-agents.
        summarizer_model: Cheaper model used to compress search results.
        subagents: Built-in + custom sub-agents available for delegation.
        extra_tools: Additional tool objects exposed to the orchestrator.
        recursion_limit: Max graph steps before LangGraph aborts a run.
        max_parallel: Cap on concurrent sub-agents in a single `task_batch` call.
        temperature: Sampling temperature for all agents.
        checkpointer: "memory" (in-process) or "sqlite" (file-backed persistence).
        db_path: SQLite file used when `checkpointer == "sqlite"`.
        enable_human_in_loop: Expose the `ask_human` tool (requires a checkpointer).
        enable_vector_memory: Expose `remember`/`save_to_memory` (requires ChromaDB).
        vector_db_path: On-disk location for the ChromaDB collection.
    """

    model: str = "openai:gpt-4o-mini"
    summarizer_model: str = "openai:gpt-4o-mini"
    subagents: list[SubAgentConfig] = field(default_factory=list)
    extra_tools: list = field(default_factory=list)
    recursion_limit: int = 50
    max_parallel: int = 4
    temperature: float = 0.0

    checkpointer: Literal["memory", "sqlite"] = "sqlite"
    db_path: str = "agent_memory.db"

    enable_human_in_loop: bool = True

    enable_vector_memory: bool = False
    vector_db_path: str = "./chroma_db"

    workspace_dir: str | None = None
