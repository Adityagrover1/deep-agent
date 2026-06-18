"""Vector store factory for long-term semantic memory.

Returns a persistent ChromaDB collection used by the `remember`/`save_to_memory` tools.
Only constructed when `config.enable_vector_memory` is true, so ChromaDB is an optional
dependency at runtime.
"""

from deep_agent.config import AgentConfig

_COLLECTION_NAME = "deep_agent_memory"


def get_vector_store(config: AgentConfig):
    """Create (or open) a persistent ChromaDB collection for agent memory.

    Args:
        config: Agent configuration (uses `vector_db_path`).

    Returns:
        A ChromaDB collection handle.
    """
    import chromadb

    client = chromadb.PersistentClient(path=config.vector_db_path)
    return client.get_or_create_collection(name=_COLLECTION_NAME)
