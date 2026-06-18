"""Example: parallel sub-agent fan-out via task_batch.

A multi-subtopic prompt encourages the orchestrator to dispatch several researcher
sub-agents concurrently with `task_batch`. Compare wall-clock time against a sequential run
to see the speedup; inspect a LangSmith trace to confirm concurrency.

Run:
    python examples/parallel_research.py
"""

import time

from dotenv import load_dotenv

from deep_agent import AgentConfig, create_deep_agent

load_dotenv()


def main() -> None:
    agent = create_deep_agent(AgentConfig(max_parallel=4))

    prompt = (
        "Research three independent subtopics and write a combined brief: "
        "(1) the cost trade-offs of solar vs wind energy, "
        "(2) the environmental impact of each, and "
        "(3) recent adoption trends. "
        "These are independent — research them in parallel."
    )

    start = time.time()
    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config={"configurable": {"thread_id": "parallel-001"}},
    )
    elapsed = time.time() - start

    print(f"\n=== DONE in {elapsed:.1f}s ===")
    print(result["messages"][-1].content)
    print("\nFiles:", list(result.get("files", {}).keys()))


if __name__ == "__main__":
    main()
