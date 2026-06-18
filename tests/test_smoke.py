"""Smoke tests for the generalised deep agent.

These verify the package wires together correctly WITHOUT making any LLM/API calls:
- pure logic (state reducer)
- tool metadata and tool behaviour (called via each tool's underlying function)
- config defaults
- the task / task_batch delegation factory
- full orchestrator graph construction

A dummy OPENAI_API_KEY is set so models can be constructed; no request is ever sent.
"""

import os

import pytest
from langchain_core.tools import BaseTool

os.environ.setdefault("OPENAI_API_KEY", "test-key-not-used")
os.environ.setdefault("TAVILY_API_KEY", "test-key-not-used")


# --------------------------------------------------------------------------- #
# Package import / public API
# --------------------------------------------------------------------------- #
def test_public_api_imports():
    import deep_agent

    for name in (
        "create_deep_agent",
        "AgentConfig",
        "SubAgentConfig",
        "DeepAgentState",
        "Todo",
    ):
        assert hasattr(deep_agent, name), f"missing public export: {name}"


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
def test_file_reducer_merges_right_wins():
    from deep_agent.state import file_reducer

    assert file_reducer(None, {"a": "1"}) == {"a": "1"}
    assert file_reducer({"a": "1"}, None) == {"a": "1"}
    merged = file_reducer({"a": "1", "b": "2"}, {"b": "9", "c": "3"})
    assert merged == {"a": "1", "b": "9", "c": "3"}


def test_state_has_todos_and_files_fields():
    from deep_agent.state import DeepAgentState

    ann = DeepAgentState.__annotations__
    assert "todos" in ann
    assert "files" in ann


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_config_defaults():
    from deep_agent.config import AgentConfig

    cfg = AgentConfig()
    assert cfg.model.startswith("openai:")
    assert cfg.max_parallel >= 1
    assert cfg.checkpointer in ("memory", "sqlite")
    assert cfg.enable_human_in_loop is True
    assert cfg.enable_vector_memory is False
    assert cfg.subagents == []  # default empty -> factory falls back to DEFAULT_SUBAGENTS


def test_default_subagents_present():
    from deep_agent.agents.subagents import DEFAULT_SUBAGENTS

    names = {s["name"] for s in DEFAULT_SUBAGENTS}
    assert {"planner", "researcher", "writer", "coder"} <= names


# --------------------------------------------------------------------------- #
# Tool metadata
# --------------------------------------------------------------------------- #
def test_tools_are_named_basetools():
    from deep_agent.tools.file_tools import edit_file, ls, read_file, write_file
    from deep_agent.tools.research_tools import think_tool
    from deep_agent.tools.todo_tools import read_todos, write_todos

    for t in (write_todos, read_todos, ls, read_file, write_file, edit_file, think_tool):
        assert isinstance(t, BaseTool)
        assert t.name and t.description


# --------------------------------------------------------------------------- #
# Tool behaviour (no LLM) — call the raw function behind each @tool
# --------------------------------------------------------------------------- #
def test_write_then_read_file_roundtrip():
    from deep_agent.tools.file_tools import read_file, write_file

    state = {"files": {}}
    cmd = write_file.func(
        file_path="notes.md", content="line1\nline2", state=state, tool_call_id="tc1"
    )
    files = cmd.update["files"]
    assert files["notes.md"] == "line1\nline2"

    rendered = read_file.func(file_path="notes.md", state={"files": files})
    assert "line1" in rendered and "line2" in rendered


def test_read_missing_file_returns_error():
    from deep_agent.tools.file_tools import read_file

    out = read_file.func(file_path="ghost.md", state={"files": {}})
    assert "not found" in out.lower()


def test_edit_file_unique_match_and_ambiguity():
    from deep_agent.tools.file_tools import edit_file

    state = {"files": {"a.txt": "foo bar foo"}}

    # Ambiguous without replace_all -> error message, no change.
    amb = edit_file.func(
        file_path="a.txt", old_string="foo", new_string="X",
        state=state, tool_call_id="tc", replace_all=False,
    )
    assert "appears 2 times" in amb.update["messages"][0].content

    # replace_all succeeds.
    ok = edit_file.func(
        file_path="a.txt", old_string="foo", new_string="X",
        state={"files": {"a.txt": "foo bar foo"}}, tool_call_id="tc", replace_all=True,
    )
    assert ok.update["files"]["a.txt"] == "X bar X"


def test_write_and_read_todos():
    from deep_agent.tools.todo_tools import read_todos, write_todos

    todos = [
        {"content": "Plan", "status": "completed"},
        {"content": "Execute", "status": "in_progress"},
    ]
    cmd = write_todos.func(todos=todos, tool_call_id="tc")
    assert cmd.update["todos"] == todos

    rendered = read_todos.func(state={"todos": todos}, tool_call_id="tc")
    assert "Plan" in rendered and "Execute" in rendered


# --------------------------------------------------------------------------- #
# Delegation factory
# --------------------------------------------------------------------------- #
def test_create_task_tool_returns_task_and_task_batch():
    from langchain.chat_models import init_chat_model

    from deep_agent.agents.subagents import DEFAULT_SUBAGENTS
    from deep_agent.tools.file_tools import edit_file, ls, read_file, write_file
    from deep_agent.tools.research_tools import tavily_search, think_tool
    from deep_agent.tools.task_tool import create_task_tool
    from deep_agent.tools.todo_tools import read_todos, write_todos

    model = init_chat_model("openai:gpt-4o-mini", temperature=0.0)
    base = [write_todos, read_todos, ls, read_file, write_file, edit_file,
            tavily_search, think_tool]
    task, task_batch = create_task_tool(DEFAULT_SUBAGENTS, all_tools=base, model=model)

    assert task.name == "task"
    assert task_batch.name == "task_batch"
    assert isinstance(task, BaseTool) and isinstance(task_batch, BaseTool)


# --------------------------------------------------------------------------- #
# Full graph construction (memory checkpointer, no vector store, no HITL keys needed)
# --------------------------------------------------------------------------- #
def test_create_deep_agent_builds_compiled_graph():
    from langgraph.graph.state import CompiledStateGraph

    from deep_agent import AgentConfig, create_deep_agent

    agent = create_deep_agent(AgentConfig(checkpointer="memory"))
    assert isinstance(agent, CompiledStateGraph)
    # Graph should expose the standard agent/tools nodes.
    nodes = set(agent.get_graph().nodes)
    assert "tools" in nodes


def test_create_deep_agent_default_config():
    from deep_agent import create_deep_agent

    # Default config uses sqlite; ensure it constructs without error.
    agent = create_deep_agent()
    assert agent is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
