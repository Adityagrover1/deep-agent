"""Agent factory — the main entry point.

`create_deep_agent(config)` wires together the orchestrator: its tools, the sub-agent
delegation tools (`task` / `task_batch`), persistent memory (checkpointer + optional vector
store), and human-in-the-loop. It returns a compiled LangGraph agent ready to `.invoke()`.

Correctness note: `create_agent` (LangChain 1.0) returns an already-compiled
`CompiledStateGraph`. The checkpointer is passed INTO `create_agent(...)` — we never call
`.compile()` on the result.
"""

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model

from deep_agent.agents.subagents import DEFAULT_SUBAGENTS
from deep_agent.config import AgentConfig
from deep_agent.memory.checkpointer import get_checkpointer
from deep_agent.memory.vector_store import get_vector_store
from deep_agent.prompts.orchestrator import ORCHESTRATOR_SYSTEM_PROMPT
from deep_agent.state import DeepAgentState
from deep_agent.tools.file_tools import (
    create_real_file_tools,
    edit_file,
    ls,
    read_file,
    write_file,
)
from deep_agent.tools.human_tools import ask_human
from deep_agent.tools.memory_tools import create_memory_tools
from deep_agent.tools.code_exec_tools import python_exec
from deep_agent.tools.research_tools import tavily_search, think_tool
from deep_agent.tools.task_tool import create_task_tool
from deep_agent.tools.todo_tools import read_todos, write_todos


def create_deep_agent(config: AgentConfig | None = None):
    """Build a generalised deep agent from a configuration.

    Args:
        config: An `AgentConfig`. If omitted, defaults are used (gpt-4o-mini orchestrator, the four
            built-in sub-agents, SQLite checkpointing, human-in-the-loop on, vector memory off).

    Returns:
        A compiled LangGraph agent. Invoke with a `thread_id` in the run config to enable
        persistence/resume, e.g.:
            agent.invoke(inputs, config={"configurable": {"thread_id": "task-001"}})
    """
    config = config or AgentConfig()
    subagents = config.subagents or DEFAULT_SUBAGENTS

    model = init_chat_model(config.model, temperature=config.temperature)
    checkpointer = get_checkpointer(config)

    # Optional long-term memory tools.
    memory_tools = []
    if config.enable_vector_memory:
        collection = get_vector_store(config)
        memory_tools = create_memory_tools(collection)

    # File tools — real disk I/O when workspace_dir is set, virtual otherwise.
    if config.workspace_dir:
        file_tools = create_real_file_tools(config.workspace_dir)
    else:
        file_tools = [ls, read_file, write_file, edit_file]

    # Master tool list — also used to resolve sub-agent string tool names.
    base_tools = [
        write_todos,
        read_todos,
        *file_tools,
        tavily_search,
        think_tool,
        python_exec,
    ]
    if config.enable_human_in_loop:
        base_tools.append(ask_human)
    base_tools.extend(memory_tools)

    task, task_batch = create_task_tool(
        subagents,
        all_tools=base_tools,
        model=model,
        max_parallel=config.max_parallel,
    )

    orchestrator_tools = [
        write_todos,
        read_todos,
        *file_tools,
        task,
        task_batch,
        *memory_tools,
        *config.extra_tools,
    ]
    if config.enable_human_in_loop:
        orchestrator_tools.append(ask_human)

    agent = create_agent(
        model,
        orchestrator_tools,
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        state_schema=DeepAgentState,
        checkpointer=checkpointer,
    ).with_config({"recursion_limit": config.recursion_limit})

    return agent
