"""Interactive demo — type your own prompts at the console.

Builds one deep agent and runs a chat loop on a single thread, so the conversation
(and its TODOs / files) persists across turns. If the agent calls `ask_human`, the
graph pauses and you answer inline, then it continues.

Run:
    python examples/demo.py

Requires OPENAI_API_KEY (and TAVILY_API_KEY for web-search prompts) in .env.

Commands while running:
    /files   list files the agent has written
    /todos   show the current task list
    /quit    exit

Output format:
    Thinking: ...       orchestrator reasoning before acting
    Delegate -> agent   delegation to a sub-agent
    toolname  arg       direct tool call
      Result: ...       tool / sub-agent result (truncated)
    Todos: [x]/[ ]      todo list as checkboxes
    Answer: ...         final response
"""

import sys
import textwrap
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from deep_agent import AgentConfig, create_deep_agent

load_dotenv()

_B = "\033[1m"   # bold
_R = "\033[0m"   # reset

_RESULT_PREVIEW = 220
_WRAP = 100

_STATUS = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}


def _truncate(text: str, limit: int = _RESULT_PREVIEW) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + " ..."


def _fmt_key_arg(args: dict) -> str:
    """Return the single most-relevant argument value for a tool call."""
    for key in ("file_path", "query", "path", "key", "reflection"):
        if key in args and isinstance(args[key], str):
            return args[key][:80]
    for v in args.values():
        if isinstance(v, str):
            return v[:80]
    return ""


def _fmt_todos(todos: list) -> None:
    for todo in todos:
        mark = _STATUS.get(todo.get("status", "pending"), "[ ]")
        print(f"    {mark} {todo['content']}")


def _is_final_ai(msg) -> bool:
    return isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None)


def _handle_message(msg, skip_ids: set) -> None:
    if isinstance(msg, AIMessage):
        # Orchestrator reasoning/decision text shown before tool calls.
        if msg.content and getattr(msg, "tool_calls", None):
            print(f"  {_B}Orchestrator:{_R}")
            for line in textwrap.wrap(str(msg.content).strip(), _WRAP - 4):
                print(f"    {line}")
            print()

        for tc in getattr(msg, "tool_calls", []) or []:
            name = tc["name"]
            args = tc.get("args", {})
            tc_id = tc.get("id", "")

            if name == "task":
                agent = args.get("subagent_type", "?")
                desc = args.get("description", "")
                print(f"  {_B}Delegate -> {agent}{_R}")
                for line in textwrap.wrap(desc, _WRAP - 4) or [""]:
                    print(f"    {line}")

            elif name == "task_batch":
                tasks = args.get("tasks", [])
                print(f"  {_B}Parallel{_R} ({len(tasks)} tasks):")
                for t in tasks:
                    agent = t.get("subagent_type", "?")
                    desc = t.get("description", "")
                    first, *rest = textwrap.wrap(desc, _WRAP - 12) or [""]
                    print(f"    [{_B}{agent}{_R}] {first}")
                    for extra in rest:
                        print(f"           {extra}")

            elif name == "write_todos":
                todos = args.get("todos", [])
                skip_ids.add(tc_id)
                print(f"  {_B}Todos:{_R}")
                _fmt_todos(todos)

            elif name in ("read_todos", "think_tool"):
                skip_ids.add(tc_id)

            else:
                key_arg = _fmt_key_arg(args)
                display = f"  {_B}{name}{_R}"
                if key_arg:
                    display += f"  {key_arg}"
                print(display)

        # Final answer — no tool calls on this message.
        if _is_final_ai(msg):
            print(f"\n{_B}Answer:{_R}")
            for line in textwrap.wrap(str(msg.content).strip(), _WRAP):
                print(f"  {line}")
            print()

    elif isinstance(msg, ToolMessage):
        if getattr(msg, "tool_call_id", "") in skip_ids:
            return
        preview = _truncate(str(msg.content))
        print(f"    Result: {preview}")
        print()


def _stream_turn(agent, inputs, config) -> tuple[dict | None, None]:
    interrupt = None
    skip_ids: set = set()

    for event_type, event_data in agent.stream(inputs, config=config, stream_mode=["updates", "custom"]):
        if event_type == "custom":
            agent_name = event_data.get("agent", "?")
            tool_name = event_data.get("tool", "?")
            key_arg = _fmt_key_arg(event_data.get("args", {}))
            line = f"    [{_B}{agent_name}{_R}] {tool_name}"
            if key_arg:
                line += f"  {key_arg[:60]}"
            print(line)
        elif event_type == "updates":
            for node, update in event_data.items():
                if node == "__interrupt__":
                    interrupt = update[0].value if update else {}
                    break
                for msg in update.get("messages", []):
                    _handle_message(msg, skip_ids)

    return interrupt, None


_DEFAULT_WORKSPACE = Path(__file__).parent.parent / "filesystem"


def main() -> None:
    workspace_dir = sys.argv[1] if len(sys.argv) > 1 else str(_DEFAULT_WORKSPACE)

    agent = create_deep_agent(
        AgentConfig(checkpointer="memory", enable_human_in_loop=True, workspace_dir=workspace_dir)
    )
    run_config = {"configurable": {"thread_id": "demo-session"}}

    print(f"Workspace: {Path(workspace_dir).resolve()}")
    print("Deep Agent demo. Type a prompt (or /files, /todos, /quit).\n")

    while True:
        try:
            user_input = input("\nYou > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return

        if not user_input:
            continue
        if user_input == "/quit":
            print("bye")
            return
        if user_input == "/files":
            if workspace_dir:
                ws = Path(workspace_dir).resolve()
                found = sorted(
                    str(p.relative_to(ws)).replace("\\", "/")
                    for p in ws.rglob("*") if p.is_file()
                )
                print("Files:", found or "(none)")
            else:
                state = agent.get_state(run_config).values
                files = state.get("files", {})
                print("Files:", list(files.keys()) or "(none)")
            continue
        if user_input == "/todos":
            state = agent.get_state(run_config).values
            todos = state.get("todos", []) or []
            if not todos:
                print("  (no todos)")
            else:
                for todo in todos:
                    mark = _STATUS.get(todo.get("status", "pending"), "[ ]")
                    print(f"  {mark} {todo['content']}")
            continue

        print()
        interrupt, _ = _stream_turn(
            agent,
            {"messages": [{"role": "user", "content": user_input}]},
            run_config,
        )

        while interrupt:
            question = interrupt.get("question", "(no question)")
            answer = input(f"Agent asks: {question}\nYour answer > ")
            print()
            interrupt, _ = _stream_turn(agent, Command(resume=answer), run_config)


if __name__ == "__main__":
    main()
