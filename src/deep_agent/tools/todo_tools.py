"""TODO management tools for task planning and progress tracking."""

from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from deep_agent.prompts.tool_descriptions import WRITE_TODOS_DESCRIPTION
from deep_agent.state import DeepAgentState, Todo


@tool(description=WRITE_TODOS_DESCRIPTION, parse_docstring=True)
def write_todos(
    todos: list[Todo], tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Create or replace the agent's TODO list.

    Args:
        todos: Full list of Todo items (content + status). Always send the complete list.
        tool_call_id: Injected tool call identifier.

    Returns:
        Command updating `todos` in state.
    """
    return Command(
        update={
            "todos": todos,
            "messages": [
                ToolMessage(f"Updated todo list to {todos}", tool_call_id=tool_call_id)
            ],
        }
    )


@tool(parse_docstring=True)
def read_todos(
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> str:
    """Read the current TODO list to stay oriented on remaining work.

    Args:
        state: Injected agent state holding the current TODO list.
        tool_call_id: Injected tool call identifier.

    Returns:
        Human-readable rendering of the current TODO list.
    """
    todos = state.get("todos", [])
    if not todos:
        return "No todos currently in the list."

    status_emoji = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}
    result = "Current TODO List:\n"
    for i, todo in enumerate(todos, 1):
        emoji = status_emoji.get(todo["status"], "❓")
        result += f"{i}. {emoji} {todo['content']} ({todo['status']})\n"
    return result.strip()
