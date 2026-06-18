# deep-agent

A configurable, production-oriented deep agent built on LangGraph: an **orchestrator** that
plans, delegates to specialized **sub-agents** (Planner, Researcher, Writer, Coder), manages
memory, recovers from errors, and can pause for human input.

## Capabilities

- **Plan** — Planner sub-agent decomposes the task; orchestrator tracks it as a TODO list.
- **Act** — tools for web search, a virtual file system, and code editing.
- **Remember** — SQLite checkpointing (resume by `thread_id`) + optional ChromaDB vector recall.
- **Delegate** — `task` (sequential) and `task_batch` (parallel fan-out) to sub-agents in isolated contexts.
- **Recover** — retry with backoff; failed tools return messages so the orchestrator reroutes.
- **Ask** — `ask_human` pauses via LangGraph `interrupt()` and resumes on `Command(resume=...)`.

## Install

```bash
uv sync          # or: pip install -e .
cp .env.example .env   # fill in OPENAI_API_KEY, TAVILY_API_KEY
```

## Use

```python
from deep_agent import create_deep_agent, AgentConfig

agent = create_deep_agent(AgentConfig())
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Research MCP and write a report"}]},
    config={"configurable": {"thread_id": "task-001"}},
)
```

See `examples/` for research, custom sub-agents, human-in-the-loop, and parallel fan-out.

## Architecture & Workflow

Full architecture, workflow, and diagrams are documented in the workspace root:
[`../README.md`](../README.md) and [`../system_architecture.md`](../system_architecture.md).
