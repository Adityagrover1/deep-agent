"""Tool descriptions surfaced to the LLM (kept separate from system prompts)."""

WRITE_TODOS_DESCRIPTION = """Create and manage a structured task list for tracking progress through complex work.

## When to Use
- Multi-step or non-trivial tasks requiring coordination
- When the user provides multiple tasks or asks for a plan
- Avoid for single, trivial actions

## Rules
- Maintain ONE list of todo objects (content, status)
- Status must be: pending, in_progress, or completed
- Only ONE task in_progress at a time
- Mark completed immediately when a task is fully done
- Always send the FULL updated list when changing anything

## Parameters
- todos: full list of TODO items with content and status

## Returns
Updates agent state with the new todo list."""

LS_DESCRIPTION = """List all files in the virtual filesystem stored in agent state.

Use this to orient yourself before other file operations. No parameters required."""

READ_FILE_DESCRIPTION = """Read content from a file in the virtual filesystem with optional pagination.

Returns content with line numbers (like `cat -n`) and supports chunked reads of large files.

Parameters:
- file_path (required): path to read
- offset (optional, default 0): starting line
- limit (optional, default 2000): max lines

Always read a file before editing it."""

WRITE_FILE_DESCRIPTION = """Create a new file or completely overwrite an existing one in the virtual filesystem.

Use for initial creation or full rewrites. Replaces the entire file content.

Parameters:
- file_path (required): destination path
- content (required): full content to write

Prefer unique, descriptive filenames so parallel sub-agents never collide."""

EDIT_FILE_DESCRIPTION = """Replace an exact substring within an existing file (find-and-replace).

Use for targeted edits instead of rewriting the whole file. The match must be exact.

Parameters:
- file_path (required): file to edit
- old_string (required): exact text to find (must be unique unless replace_all=true)
- new_string (required): replacement text
- replace_all (optional, default false): replace every occurrence

Read the file first so old_string matches exactly."""

THINK_TOOL_DESCRIPTION = """Strategic reflection tool. Use it to pause and reason deliberately.

When to use:
- After receiving results: what did I learn?
- Before deciding next steps: do I have enough to proceed?
- When assessing gaps: what is still missing?
- Before concluding: can I deliver a complete answer now?

Reflection should cover: current findings, gaps, quality, and the next decision."""

SUMMARIZE_WEB_SEARCH = """You are creating a minimal summary for research steering — your goal is to help an agent know what information it has collected, NOT to preserve all details.

<webpage_content>
{webpage_content}
</webpage_content>

Create a VERY CONCISE summary focusing on:
1. Main topic/subject in 1-2 sentences
2. Key information type (facts, tutorial, news, analysis, etc.)
3. The most significant 1-2 findings or points

Keep the summary under 150 words. Also generate a descriptive filename indicating the
content type and topic (e.g. "mcp_protocol_overview.md").

Output format:
```json
{{
   "filename": "descriptive_filename.md",
   "summary": "Brief summary under 150 words"
}}
```

Today's date: {date}
"""

TASK_DESCRIPTION_PREFIX = """Delegate ONE task to a specialized sub-agent with an isolated context (runs sequentially).

Use this when a step depends on the output of a previous step. The sub-agent sees ONLY your
task description — provide complete, standalone instructions (no acronyms, no references to
prior conversation).

Available sub-agents:
{other_agents}

Parameters:
- description: a clear, self-contained task
- subagent_type: which sub-agent to use (one of the names above)"""

TASK_BATCH_DESCRIPTION_PREFIX = """Delegate SEVERAL INDEPENDENT tasks to sub-agents that run CONCURRENTLY.

Use this to fan out work that does not depend on each other — e.g. researching multiple
subtopics at once. Each task runs in its own isolated context. Provide complete, standalone
instructions for each. Tell sub-agents to use unique, descriptive filenames so their outputs
do not collide.

Available sub-agents:
{other_agents}

Parameters:
- tasks: a list of objects, each with:
    - description: a clear, self-contained task
    - subagent_type: which sub-agent to use (one of the names above)"""
