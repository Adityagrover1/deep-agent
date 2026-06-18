"""Long-term semantic memory tools backed by a vector store.

These are factory-built so the ChromaDB collection is injected at creation time (in the agent
factory) rather than imported as a global. `remember` retrieves relevant past task summaries;
`save_to_memory` persists new ones for cross-session recall.
"""

from langchain_core.tools import BaseTool, tool


def create_memory_tools(collection) -> list[BaseTool]:
    """Build `remember` and `save_to_memory` tools bound to a ChromaDB collection.

    Args:
        collection: A ChromaDB collection (see `deep_agent.memory.vector_store`).

    Returns:
        A list of two tools: [remember, save_to_memory].
    """

    @tool(parse_docstring=True)
    def remember(query: str, n_results: int = 3) -> str:
        """Search long-term memory for context relevant to the current task.

        Args:
            query: What to look for in past task summaries.
            n_results: How many matches to return (default 3).

        Returns:
            Formatted matches, or a notice if memory is empty.
        """
        try:
            results = collection.query(query_texts=[query], n_results=n_results)
        except Exception as e:
            return f"Memory query failed: {e}"

        docs = (results.get("documents") or [[]])[0]
        ids = (results.get("ids") or [[]])[0]
        if not docs:
            return "No relevant long-term memories found."
        return "\n\n".join(f"[{mid}] {doc}" for mid, doc in zip(ids, docs))

    @tool(parse_docstring=True)
    def save_to_memory(key: str, content: str) -> str:
        """Persist a summary to long-term memory for future sessions.

        Args:
            key: Unique identifier for this memory (used to upsert).
            content: The text to store.

        Returns:
            Confirmation message.
        """
        try:
            collection.upsert(documents=[content], ids=[key])
        except Exception as e:
            return f"Failed to save memory '{key}': {e}"
        return f"Saved '{key}' to long-term memory."

    return [remember, save_to_memory]
