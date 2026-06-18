"""Example: research a topic end-to-end with checkpointing + human-in-the-loop.

Run:
    python examples/research_agent.py

Requires OPENAI_API_KEY and TAVILY_API_KEY in the environment (.env).
"""

from dotenv import load_dotenv

from deep_agent import AgentConfig, create_deep_agent

load_dotenv()


def main() -> None:
    agent = create_deep_agent(
        AgentConfig(checkpointer="sqlite", enable_human_in_loop=True)
    )

    run_config = {"configurable": {"thread_id": "mcp-research-001"}}
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Research the latest advances in MCP (Model Context Protocol) "
                        "and write a concise report."
                    ),
                }
            ]
        },
        config=run_config,
    )

    print("\n=== FINAL MESSAGE ===")
    print(result["messages"][-1].content)

    print("\n=== FILES PRODUCED ===")
    for name in result.get("files", {}):
        print(f"- {name}")

    print("\n=== TODOS ===")
    for todo in result.get("todos", []):
        print(f"- [{todo['status']}] {todo['content']}")


if __name__ == "__main__":
    main()
