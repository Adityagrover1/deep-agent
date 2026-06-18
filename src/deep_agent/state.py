"""Shared agent state: TODO tracking + virtual file system.

`DeepAgentState` extends LangChain's `AgentState` (which carries `messages`) with:
- `todos`: a task list the orchestrator recites and updates as work progresses
- `files`: a virtual file system used to offload heavy content out of the message stream

The `file_reducer` lets concurrent/sequential updates merge safely (right side wins),
which is what makes parallel sub-agent fan-out (`task_batch`) work without clobbering files.
"""

from typing import Annotated, Literal, NotRequired

from langchain.agents import AgentState
from typing_extensions import TypedDict


class Todo(TypedDict):
    """A single tracked task.

    Attributes:
        content: Short, specific description of the task.
        status: One of "pending", "in_progress", "completed".
    """

    content: str
    status: Literal["pending", "in_progress", "completed"]


def file_reducer(left, right):
    """Merge two virtual-filesystem dicts, with the right side taking precedence.

    Args:
        left: Existing files (or None).
        right: New/updated files (or None).

    Returns:
        Merged dict where right-hand keys override left-hand keys.
    """
    if left is None:
        return right
    if right is None:
        return left
    return {**left, **right}


class DeepAgentState(AgentState):
    """Extended agent state with task planning and a virtual file system.

    Inherits `messages` from `AgentState` and adds:
    - todos: task list for planning and progress tracking
    - files: virtual filesystem mapping filename -> content
    """

    todos: NotRequired[list[Todo]]
    files: Annotated[NotRequired[dict[str, str]], file_reducer]
