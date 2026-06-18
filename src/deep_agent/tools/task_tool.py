"""Sub-agent delegation tools (context isolation + parallel fan-out).

`create_task_tool` builds a registry of sub-agents (one compiled graph each) and returns two
delegation tools:

- `task`        — sequential delegation for steps that depend on prior output.
- `task_batch`  — parallel delegation (asyncio.gather over `.ainvoke`) for independent steps.

Each delegation runs the sub-agent in an isolated context: its message history is reset to
just the task description, so it is never confused by the orchestrator's conversation. Files
produced by sub-agents are merged back into parent state via the `file_reducer`.
"""

import asyncio
import copy
from typing import Annotated

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.config import get_stream_writer
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from typing_extensions import TypedDict

from deep_agent.config import SubAgentConfig
from deep_agent.prompts.tool_descriptions import (
    TASK_BATCH_DESCRIPTION_PREFIX,
    TASK_DESCRIPTION_PREFIX,
)
from deep_agent.state import DeepAgentState


class TaskSpec(TypedDict):
    """One unit of work in a parallel `task_batch` call."""

    description: str
    subagent_type: str


def create_task_tool(
    subagents: list[SubAgentConfig],
    all_tools: list,
    model,
    state_schema=DeepAgentState,
    max_parallel: int = 4,
) -> tuple[BaseTool, BaseTool]:
    """Create the `task` and `task_batch` delegation tools.

    Args:
        subagents: Sub-agent configurations to register.
        all_tools: Master tool list; sub-agent string tool names are resolved against it.
        model: Model used for every sub-agent.
        state_schema: State schema for the sub-agent graphs.
        max_parallel: Concurrency cap for `task_batch`.

    Returns:
        (task, task_batch) tools.
    """
    # Resolve string tool names against the master registry (course pattern).
    tools_by_name = {}
    for tool_ in all_tools:
        if not isinstance(tool_, BaseTool):
            tool_ = tool(tool_)
        tools_by_name[tool_.name] = tool_

    agents = {}
    for cfg in subagents:
        if "tools" in cfg:
            selected = [tools_by_name[t] for t in cfg["tools"]]
        else:
            selected = list(tools_by_name.values())
        agents[cfg["name"]] = create_agent(
            model, system_prompt=cfg["prompt"], tools=selected, state_schema=state_schema
        )

    agent_directory = "\n".join(f"- {cfg['name']}: {cfg['description']}" for cfg in subagents)

    def _isolated_state(base_state: DeepAgentState, description: str) -> dict:
        """Copy parent state but reset messages to only the task description."""
        new_state = copy.copy(base_state)
        new_state["messages"] = [{"role": "user", "content": description}]
        return new_state

    @tool(description=TASK_DESCRIPTION_PREFIX.format(other_agents=agent_directory))
    def task(
        description: str,
        subagent_type: str,
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Delegate one task to a sub-agent with isolated context (sequential)."""
        if subagent_type not in agents:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            f"Error: unknown subagent_type '{subagent_type}'. "
                            f"Allowed: {list(agents)}",
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )

        try:
            write = get_stream_writer()
            collected_messages: list = []
            collected_files: dict = {}

            for event in agents[subagent_type].stream(
                _isolated_state(state, description), stream_mode="updates"
            ):
                for _node, update in event.items():
                    if "files" in update:
                        collected_files.update(update["files"])
                    for msg in update.get("messages", []):
                        collected_messages.append(msg)
                        if isinstance(msg, AIMessage):
                            for tc in getattr(msg, "tool_calls", None) or []:
                                write({"agent": subagent_type, "tool": tc["name"], "args": tc.get("args", {})})

            final = next(
                (m for m in reversed(collected_messages) if isinstance(m, AIMessage) and not getattr(m, "tool_calls", None)),
                collected_messages[-1] if collected_messages else None,
            )
            content = final.content if final else "(no response)"

        except Exception as e:
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            f"Sub-agent '{subagent_type}' failed: {e}. "
                            "Consider a simpler task, a different agent, or rerouting.",
                            tool_call_id=tool_call_id,
                        )
                    ]
                }
            )

        return Command(
            update={
                "files": collected_files,
                "messages": [ToolMessage(content, tool_call_id=tool_call_id)],
            }
        )

    @tool(description=TASK_BATCH_DESCRIPTION_PREFIX.format(other_agents=agent_directory))
    def task_batch(
        tasks: list[TaskSpec],
        state: Annotated[DeepAgentState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
    ) -> Command:
        """Delegate several INDEPENDENT tasks to sub-agents concurrently (parallel)."""
        async def _main():
            semaphore = asyncio.Semaphore(max_parallel)

            async def _run(spec: TaskSpec):
                sub = spec.get("subagent_type")
                desc = spec.get("description", "")
                if sub not in agents:
                    return sub, desc, None, f"unknown subagent_type '{sub}'"
                async with semaphore:
                    try:
                        result = await agents[sub].ainvoke(_isolated_state(state, desc))
                        return sub, desc, result, None
                    except Exception as e:  # noqa: BLE001 - surfaced to orchestrator
                        return sub, desc, None, str(e)

            return await asyncio.gather(*[_run(spec) for spec in tasks])

        outcomes = asyncio.run(_main())

        merged_files: dict[str, str] = {}
        report_lines = []
        for sub, desc, result, error in outcomes:
            if error is not None:
                report_lines.append(f"❌ [{sub}] {desc[:60]} → FAILED: {error}")
                continue
            merged_files.update(result.get("files", {}))
            last = result["messages"][-1].content
            report_lines.append(f"✅ [{sub}] {desc[:60]} → {last}")

        return Command(
            update={
                "files": merged_files,
                "messages": [
                    ToolMessage(
                        "Parallel delegation complete:\n" + "\n".join(report_lines),
                        tool_call_id=tool_call_id,
                    )
                ],
            }
        )

    return task, task_batch
