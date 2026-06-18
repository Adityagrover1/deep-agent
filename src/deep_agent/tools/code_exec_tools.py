"""Code execution tool — runs Python in a subprocess for safe, isolated evaluation."""

import ast
import subprocess
import sys

from langchain_core.tools import tool


def _ensure_output(code: str) -> str:
    """If the last statement is a bare expression, wrap it with print() so stdout is non-empty."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body[-1]
        expr_src = ast.get_source_segment(code, last)
        if expr_src:
            lines = code.splitlines()
            before = "\n".join(lines[: last.lineno - 1])
            return (before + "\n" if before else "") + f"print({expr_src})"
    return code


@tool
def python_exec(code: str, timeout: int = 10) -> str:
    """Execute Python code in a subprocess and return stdout.

    Use this for any mathematical computation, data transformation, or logic
    that needs a precise, verified result. Never compute math in your head —
    write the code and run it here. Always use print() to output results.

    Args:
        code: Valid Python source code to execute.
        timeout: Max seconds before the process is killed (default 10).

    Returns:
        stdout on success, or the stderr message on non-zero exit.
    """
    runnable = _ensure_output(code)
    try:
        result = subprocess.run(
            [sys.executable, "-c", runnable],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Error: execution timed out after {timeout}s"

    if result.returncode != 0:
        return f"Error:\n{result.stderr.strip()}"
    return result.stdout.strip() or "(no output)"
