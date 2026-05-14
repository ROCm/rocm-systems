"""CI guardrail: only framework.py may import the OpenAI Agents SDK (review N5).

Scans every file in perfxpert/agents/ and perfxpert/runtime/ and asserts
that none of them imports `openai_agents`, `openai.agents`, or similar —
except framework.py itself.
"""

import ast
from pathlib import Path


AGENT_PKG = Path(__file__).parent.parent.parent / "perfxpert" / "agents"
RUNTIME_PKG = Path(__file__).parent.parent.parent / "perfxpert" / "runtime"

ALLOWED = {"framework.py"}


def _scan_tree(root: Path) -> list:
    violators = []
    for py in root.rglob("*.py"):
        if py.name in ALLOWED:
            continue
        tree = ast.parse(py.read_text(), filename=str(py))
        if any(_is_sdk_import(node) for node in ast.walk(tree)):
            violators.append(str(py))
    return violators


def _is_sdk_import(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(alias.name in {"openai_agents", "openai.agents"} for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        return node.module in {"openai_agents", "openai.agents"}
    return False


def test_no_sdk_import_in_agents_package():
    violators = _scan_tree(AGENT_PKG)
    assert not violators, (
        f"These files import openai_agents directly — must go via framework.py: "
        f"{violators}"
    )


def test_no_sdk_import_in_runtime_package():
    violators = _scan_tree(RUNTIME_PKG)
    assert not violators, f"Runtime files importing openai_agents: {violators}"
