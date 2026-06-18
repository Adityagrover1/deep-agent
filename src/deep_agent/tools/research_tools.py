"""Research tools: web search with context offloading, plus a reflection tool.

`tavily_search` searches the web, summarizes each result with a cheap model, writes the full
content to the virtual filesystem, and returns only a short summary to the agent. Transient
API failures are retried with exponential backoff; permanent failures return a descriptive
ToolMessage instead of raising, so the orchestrator can route around them.
"""

import base64
import os
import uuid
from datetime import datetime
from typing import Annotated, Literal

import httpx
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import InjectedToolArg, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from markdownify import markdownify
from pydantic import BaseModel, Field
from tavily import TavilyClient
from tenacity import retry, stop_after_attempt, wait_exponential

from deep_agent.prompts.tool_descriptions import SUMMARIZE_WEB_SEARCH, THINK_TOOL_DESCRIPTION
from deep_agent.state import DeepAgentState

# Summarizer is created lazily so importing the package doesn't require an OpenAI key.
_summarization_model = None
_tavily_client = None


def _get_summarizer(model: str = "openai:gpt-4o-mini"):
    global _summarization_model
    if _summarization_model is None:
        _summarization_model = init_chat_model(model=model)
    return _summarization_model


def _get_tavily() -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        _tavily_client = TavilyClient()
    return _tavily_client


class Summary(BaseModel):
    """Schema for webpage content summarization."""

    filename: str = Field(description="Descriptive filename to store the content under.")
    summary: str = Field(description="Concise key learnings from the webpage.")


def get_today_str() -> str:
    """Current date as a human-readable string."""
    return datetime.now().strftime("%a %b %d, %Y")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def run_tavily_search(
    search_query: str,
    max_results: int = 1,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = True,
) -> dict:
    """Execute a single Tavily search, retrying transient failures with backoff."""
    return _get_tavily().search(
        search_query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )


def summarize_webpage_content(webpage_content: str) -> Summary:
    """Summarize raw webpage content; fall back to a truncated copy on failure."""
    try:
        structured_model = _get_summarizer().with_structured_output(Summary)
        return structured_model.invoke(
            [
                HumanMessage(
                    content=SUMMARIZE_WEB_SEARCH.format(
                        webpage_content=webpage_content, date=get_today_str()
                    )
                )
            ]
        )
    except Exception:
        snippet = webpage_content[:1000] + "..." if len(webpage_content) > 1000 else webpage_content
        return Summary(filename="search_result.md", summary=snippet)


def process_search_results(results: dict) -> list[dict]:
    """Fetch each result URL, convert to markdown, and summarize."""
    processed_results = []
    http_client = httpx.Client(timeout=30.0)

    for result in results.get("results", []):
        url = result["url"]
        try:
            response = http_client.get(url)
            if response.status_code == 200:
                raw_content = markdownify(response.text)
                summary_obj = summarize_webpage_content(raw_content)
            else:
                raw_content = result.get("raw_content", "")
                summary_obj = Summary(
                    filename="URL_error.md",
                    summary=result.get("content", "Error reading URL; try another search."),
                )
        except (httpx.TimeoutException, httpx.RequestError):
            raw_content = result.get("raw_content", "")
            summary_obj = Summary(
                filename="connection_error.md",
                summary=result.get(
                    "content", "Could not fetch URL (timeout/connection error). Try another search."
                ),
            )

        # Uniquify filenames so parallel sub-agents never collide.
        uid = base64.urlsafe_b64encode(uuid.uuid4().bytes).rstrip(b"=").decode("ascii")[:8]
        name, ext = os.path.splitext(summary_obj.filename)
        summary_obj.filename = f"{name}_{uid}{ext or '.md'}"

        processed_results.append(
            {
                "url": result["url"],
                "title": result["title"],
                "summary": summary_obj.summary,
                "filename": summary_obj.filename,
                "raw_content": raw_content,
            }
        )

    return processed_results


@tool(parse_docstring=True)
def tavily_search(
    query: str,
    state: Annotated[DeepAgentState, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId],
    max_results: Annotated[int, InjectedToolArg] = 1,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> Command:
    """Search the web, offload full results to files, and return a short summary.

    Args:
        query: Search query to execute.
        state: Injected agent state for file storage.
        tool_call_id: Injected tool call identifier.
        max_results: Maximum number of results (default 1).
        topic: 'general', 'news', or 'finance' (default 'general').

    Returns:
        Command saving full results to files plus a minimal summary message. On failure,
        a descriptive error ToolMessage so the agent can reroute.
    """
    try:
        search_results = run_tavily_search(
            query, max_results=max_results, topic=topic, include_raw_content=True
        )
        processed_results = process_search_results(search_results)
    except Exception as e:  # retries exhausted or unexpected error
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        f"Search failed for '{query}': {e}. "
                        "Try rephrasing the query or a different approach.",
                        tool_call_id=tool_call_id,
                    )
                ]
            }
        )

    files = state.get("files", {})
    saved_files = []
    summaries = []
    for result in processed_results:
        filename = result["filename"]
        files[filename] = (
            f"# Search Result: {result['title']}\n\n"
            f"**URL:** {result['url']}\n"
            f"**Query:** {query}\n"
            f"**Date:** {get_today_str()}\n\n"
            f"## Summary\n{result['summary']}\n\n"
            f"## Raw Content\n{result['raw_content'] or 'No raw content available'}\n"
        )
        saved_files.append(filename)
        summaries.append(f"- {filename}: {result['summary']}")

    summary_text = (
        f"🔍 Found {len(processed_results)} result(s) for '{query}':\n\n"
        f"{chr(10).join(summaries)}\n\n"
        f"Files: {', '.join(saved_files)}\n"
        f"💡 Use read_file() to access full details when needed."
    )
    return Command(
        update={
            "files": files,
            "messages": [ToolMessage(summary_text, tool_call_id=tool_call_id)],
        }
    )


@tool(description=THINK_TOOL_DESCRIPTION, parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Record a strategic reflection to create a deliberate planning pause.

    Args:
        reflection: Your analysis of progress, gaps, and next steps.

    Returns:
        Confirmation that the reflection was recorded.
    """
    return f"Reflection recorded: {reflection}"
