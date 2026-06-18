"""Human-in-the-loop tool.

`ask_human` wraps LangGraph's `interrupt()`: it pauses the graph, surfaces a question to the
caller, and resumes from the exact same point once the caller supplies an answer via
`Command(resume=...)`. Requires the agent to be compiled with a checkpointer so the paused
state can be persisted.
"""

from langchain_core.tools import tool
from langgraph.types import interrupt


@tool(parse_docstring=True)
def ask_human(question: str) -> str:
    """Pause execution and ask the human a question, then resume with their answer.

    Use before any irreversible/destructive action, when the user's intent is genuinely
    ambiguous and guessing wrong would waste significant effort, or when you need private
    information (credentials, preferences) you do not have.

    Args:
        question: The question to put to the human.

    Returns:
        The human's answer (supplied on resume).
    """
    return interrupt({"question": question})
