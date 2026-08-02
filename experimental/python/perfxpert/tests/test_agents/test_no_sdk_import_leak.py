"""CI guardrail: only framework.py may import the OpenAI Agents SDK (review N5).

Scans every file in perfxpert/agents/ and perfxpert/runtime/ and asserts
that none of them imports the SDK — except framework.py itself.

The SDK installs as the top-level ``agents`` package and that is how
framework.py imports it, so a check for ``openai_agents`` alone would miss
every realistic leak. ``perfxpert.agents`` is our own package and is fine;
only the bare ``agents`` root is the SDK.
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


SDK_ROOTS = ("openai_agents", "openai.agents", "agents")


def _is_sdk_module(name: str | None) -> bool:
    if not name:
        return False
    return any(name == root or name.startswith(root + ".") for root in SDK_ROOTS)


def _is_sdk_import(node: ast.AST) -> bool:
    if isinstance(node, ast.Import):
        return any(_is_sdk_module(alias.name) for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        # level > 0 is a relative import (``from . import x``) — never the SDK.
        if node.level:
            return False
        return _is_sdk_module(node.module)
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


def test_detector_recognizes_the_canonical_sdk_module():
    """framework.py imports `from agents import ...`; the guard must see that."""
    for src in (
        "import agents",
        "from agents import Agent",
        "from agents.models.multi_provider import MultiProvider",
        "import openai_agents",
    ):
        tree = ast.parse(src)
        assert any(_is_sdk_import(n) for n in ast.walk(tree)), src


def test_detector_ignores_our_own_agents_package():
    for src in (
        "from perfxpert.agents import schemas",
        "import perfxpert.agents.framework",
        "from . import framework",
    ):
        tree = ast.parse(src)
        assert not any(_is_sdk_import(n) for n in ast.walk(tree)), src


def test_framework_is_the_only_allowed_importer():
    """Guard against the allowlist quietly growing."""
    assert ALLOWED == {"framework.py"}
