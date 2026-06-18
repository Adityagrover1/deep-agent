"""Checkpointer factory.

Returns the LangGraph checkpointer that persists `DeepAgentState` after every graph step.
This is what powers session resume (same `thread_id`) and is required for human-in-the-loop
(`interrupt()` needs somewhere to persist the paused state).
"""

from deep_agent.config import AgentConfig


def get_checkpointer(config: AgentConfig):
    """Build a checkpointer based on `config.checkpointer`.

    Args:
        config: Agent configuration.

    Returns:
        A LangGraph checkpointer instance:
        - "memory": in-process `MemorySaver` (lost on exit; fine for dev).
        - "sqlite": file-backed `SqliteSaver` (survives restarts; default).
    """
    if config.checkpointer == "memory":
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver()

    # File-backed persistence. We open a raw sqlite3 connection so the saver outlives
    # the `from_conn_string` context manager (which would otherwise close on exit).
    import sqlite3

    from langgraph.checkpoint.sqlite import SqliteSaver

    conn = sqlite3.connect(config.db_path, check_same_thread=False)
    return SqliteSaver(conn)
