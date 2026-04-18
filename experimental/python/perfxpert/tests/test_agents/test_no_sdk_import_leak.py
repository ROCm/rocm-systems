"""CI guardrail: only framework.py may import the OpenAI Agents SDK (review N5).

Scans every file in perfxpert/agents/ and perfxpert/runtime/ and asserts
that none of them imports `openai_agents`, `openai.agents`, or similar —
except framework.py itself.
"""

import re
from pathlib import Path

import pytest


AGENT_PKG = Path(__file__).parent.parent.parent / "perfxpert" / "agents"
RUNTIME_PKG = Path(__file__).parent.parent.parent / "perfxpert" / "runtime"

ALLOWED = {"framework.py"}

SDK_IMPORT_RE = re.compile(
    r"^\s*(from\s+openai_agents|import\s+openai_agents|"
    r"from\s+openai\.agents|import\s+openai\.agents)",
    re.MULTILINE,
)


def _scan_tree(root: Path) -> list:
    violators = []
    for py in root.rglob("*.py"):
        if py.name in ALLOWED:
            continue
        text = py.read_text()
        if SDK_IMPORT_RE.search(text):
            violators.append(str(py))
    return violators


def test_no_sdk_import_in_agents_package():
    violators = _scan_tree(AGENT_PKG)
    assert not violators, (
        f"These files import openai_agents directly — must go via framework.py: "
        f"{violators}"
    )


def test_no_sdk_import_in_runtime_package():
    violators = _scan_tree(RUNTIME_PKG)
    assert not violators, f"Runtime files importing openai_agents: {violators}"
