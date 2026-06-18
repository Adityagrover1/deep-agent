"""File system tools.

Two modes:
- Virtual (default): files live in agent state (`state["files"]`), no real disk I/O.
- Real (`create_real_file_tools(workspace_dir)`): files are read/written on disk inside
  a scoped workspace directory. Paths that escape the workspace are rejected.
"""

from pathlib import Path
from typing import Annotated

from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

from deep_agent.prompts.tool_descriptions import (
    EDIT_FILE_DESCRIPTION,
    LS_DESCRIPTION,
    READ_FILE_DESCRIPTION,
    WRITE_FILE_DESCRIPTION,
)
from deep_agent.state import DeepAgentState


@tool(description=LS_DESCRIPTION)
def ls(state: Annotated[DeepAgentState, InjectedState]) -> list[str]:
    """List all files in the virtual filesystem."""
    return list(state.get("files", {}).keys())


@tool(description=READ_FILE_DESCRIPTION, parse_docstring=True)
def read_file(
    file_path: str,
    state: Annotated[DeepAgentState, InjectedState],
    offset: int = 0,
    limit: int = 2000,
) -> str:
    """Read file content with optional line offset and limit.

    Args:
        file_path: Path of the file to read.
        state: Injected agent state holding the virtual filesystem.
        offset: Line number to start from (default 0).
        limit: Maximum number of lines to read (default 2000).

    Returns:
        File content with line numbers, or an error string if not found.
    """
    files = state.get("files", {})
    if file_path not in files:
        return f"Error: File '{file_path}' not found"

    content = files[file_path]
    if not content:
        return "System reminder: File exists but has empty contents"

    lines = content.splitlines()
    start_idx = offset
    end_idx = min(start_idx + limit, len(lines))

    if start_idx >= len(lines):
        return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"

    result_lines = [f"{i + 1:6d}\t{lines[i][:2000]}" for i in range(start_idx, end_idx)]
    return "\n".join(result_lines)


@tool(description=WRITE_FILE_DESCRIPTION, parse_docstring=True)
def write_file(
    file_path: str,
    content: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Create or completely overwrite a file in the virtual filesystem.

    Args:
        file_path: Destination path.
        content: Full content to write.
        state: Injected agent state holding the virtual filesystem.
        tool_call_id: Injected tool call identifier.

    Returns:
        Command updating `files` in state.
    """
    files = state.get("files", {})
    files[file_path] = content
    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(f"Updated file {file_path}", tool_call_id=tool_call_id)
            ],
        }
    )


@tool(description=EDIT_FILE_DESCRIPTION, parse_docstring=True)
def edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    replace_all: bool = False,
) -> Command:
    """Replace an exact substring within an existing file.

    Args:
        file_path: Path of the file to edit.
        old_string: Exact text to find. Must be unique unless `replace_all` is true.
        new_string: Text to substitute in.
        state: Injected agent state holding the virtual filesystem.
        tool_call_id: Injected tool call identifier.
        replace_all: Replace every occurrence instead of requiring a unique match.

    Returns:
        Command updating `files` in state, or an error ToolMessage on a bad match.
    """
    files = state.get("files", {})

    def _err(msg: str) -> Command:
        return Command(
            update={"messages": [ToolMessage(msg, tool_call_id=tool_call_id)]}
        )

    if file_path not in files:
        return _err(f"Error: File '{file_path}' not found")

    content = files[file_path]
    occurrences = content.count(old_string)
    if occurrences == 0:
        return _err(f"Error: old_string not found in '{file_path}'")
    if occurrences > 1 and not replace_all:
        return _err(
            f"Error: old_string appears {occurrences} times in '{file_path}'. "
            "Provide more context to make it unique, or set replace_all=true."
        )

    files[file_path] = content.replace(old_string, new_string)
    edited = "all occurrences" if replace_all else "1 occurrence"
    return Command(
        update={
            "files": files,
            "messages": [
                ToolMessage(f"Edited {file_path} ({edited})", tool_call_id=tool_call_id)
            ],
        }
    )


def create_real_file_tools(workspace_dir: str) -> list:
    """Create file tools that operate on the real filesystem inside *workspace_dir*.

    All paths are resolved relative to the workspace root and validated to prevent
    directory-traversal escapes (``../`` etc.).  The workspace is created on first use
    if it does not already exist.
    """
    _workspace = Path(workspace_dir).resolve()
    _workspace.mkdir(parents=True, exist_ok=True)

    def _safe(file_path: str) -> Path:
        resolved = (_workspace / file_path).resolve()
        if not resolved.is_relative_to(_workspace):
            raise ValueError(f"Path escapes workspace: {file_path!r}")
        return resolved

    @tool(description=LS_DESCRIPTION)
    def ls() -> list[str]:  # type: ignore[no-redef]
        """List all files in the workspace directory."""
        return sorted(
            str(p.relative_to(_workspace)).replace("\\", "/")
            for p in _workspace.rglob("*")
            if p.is_file()
        )

    @tool(description=READ_FILE_DESCRIPTION, parse_docstring=True)
    def read_file(file_path: str, offset: int = 0, limit: int = 2000) -> str:  # type: ignore[no-redef]
        """Read file content with optional line offset and limit.

        Args:
            file_path: Path of the file to read, relative to workspace.
            offset: Line number to start from (default 0).
            limit: Maximum number of lines to read (default 2000).

        Returns:
            File content with line numbers, or an error string if not found.
        """
        path = _safe(file_path)
        if not path.exists():
            return f"Error: File '{file_path}' not found"
        content = path.read_text(encoding="utf-8")
        if not content:
            return "System reminder: File exists but has empty contents"
        lines = content.splitlines()
        end_idx = min(offset + limit, len(lines))
        if offset >= len(lines):
            return f"Error: Line offset {offset} exceeds file length ({len(lines)} lines)"
        return "\n".join(f"{i + 1:6d}\t{lines[i][:2000]}" for i in range(offset, end_idx))

    @tool(description=WRITE_FILE_DESCRIPTION, parse_docstring=True)
    def write_file(file_path: str, content: str) -> str:  # type: ignore[no-redef]
        """Create or completely overwrite a file in the workspace.

        Args:
            file_path: Destination path, relative to workspace.
            content: Full content to write.

        Returns:
            Confirmation string.
        """
        path = _safe(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Wrote {file_path}"

    @tool(description=EDIT_FILE_DESCRIPTION, parse_docstring=True)
    def edit_file(  # type: ignore[no-redef]
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> str:
        """Replace an exact substring within an existing file.

        Args:
            file_path: Path of the file to edit, relative to workspace.
            old_string: Exact text to find. Must be unique unless replace_all is true.
            new_string: Text to substitute in.
            replace_all: Replace every occurrence instead of requiring a unique match.

        Returns:
            Confirmation string, or an error string on a bad match.
        """
        path = _safe(file_path)
        if not path.exists():
            return f"Error: File '{file_path}' not found"
        content = path.read_text(encoding="utf-8")
        occurrences = content.count(old_string)
        if occurrences == 0:
            return f"Error: old_string not found in '{file_path}'"
        if occurrences > 1 and not replace_all:
            return (
                f"Error: old_string appears {occurrences} times in '{file_path}'. "
                "Provide more context to make it unique, or set replace_all=true."
            )
        edited = "all occurrences" if replace_all else "1 occurrence"
        path.write_text(content.replace(old_string, new_string), encoding="utf-8")
        return f"Edited {file_path} ({edited})"

    return [ls, read_file, write_file, edit_file]
