"""Example: register a custom sub-agent, add an extra tool, enable vector memory.

Run:
    python examples/custom_agent.py
"""

from langchain_core.tools import tool
from dotenv import load_dotenv

from deep_agent import AgentConfig, SubAgentConfig, create_deep_agent
from deep_agent.agents.subagents import (
    CODER_SUBAGENT,
    PLANNER_SUBAGENT,
    RESEARCH_SUBAGENT,
    WRITER_SUBAGENT,
)

load_dotenv()


# A simple extra tool exposed to the orchestrator.
@tool(parse_docstring=True)
def word_count(text: str) -> int:
    """Count the words in a piece of text.

    Args:
        text: The text to count.

    Returns:
        The number of whitespace-separated words.
    """
    return len(text.split())


# A custom sub-agent. Its `tools` are resolved by name against the orchestrator's registry.
ANALYST_SUBAGENT: SubAgentConfig = {
    "name": "analyst",
    "description": "Analyzes data/notes already saved in the virtual filesystem.",
    "prompt": (
        "You are a data analyst. Read the relevant files, reason carefully with think_tool, "
        "and write a structured analysis to a uniquely named file."
    ),
    "tools": ["read_file", "write_file", "ls", "think_tool"],
}


def main() -> None:
    config = AgentConfig(
        subagents=[
            PLANNER_SUBAGENT,
            RESEARCH_SUBAGENT,
            WRITER_SUBAGENT,
            CODER_SUBAGENT,
            ANALYST_SUBAGENT,
        ],
        extra_tools=[word_count],
        enable_vector_memory=True,
    )
    agent = create_deep_agent(config)

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Summarize the key trade-offs of serverless vs containers, then analyze which suits a small startup.",
                }
            ]
        },
        config={"configurable": {"thread_id": "custom-001"}},
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
