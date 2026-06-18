"""Example: the interrupt/resume cycle.

The agent may call `ask_human` mid-task, which pauses the graph. We detect the interrupt,
collect input at the console, and resume with `Command(resume=...)`.

Run:
    python examples/human_in_loop.py
"""

from dotenv import load_dotenv
from langgraph.types import Command

from deep_agent import AgentConfig, create_deep_agent

load_dotenv()


def main() -> None:
    agent = create_deep_agent(
        AgentConfig(checkpointer="sqlite", enable_human_in_loop=True)
    )
    run_config = {"configurable": {"thread_id": "hitl-001"}}

    inputs = {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Plan a 3-day trip. If anything about my preferences is unclear, "
                    "ask me before committing to a plan."
                ),
            }
        ]
    }

    result = agent.invoke(inputs, config=run_config)

    # Resume loop: keep answering interrupts until the agent finishes.
    while "__interrupt__" in result:
        interrupt = result["__interrupt__"][0]
        question = interrupt.value.get("question", "(no question)")
        answer = input(f"\n🤖 Agent asks: {question}\n> ")
        result = agent.invoke(Command(resume=answer), config=run_config)

    print("\n=== FINAL ===")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
