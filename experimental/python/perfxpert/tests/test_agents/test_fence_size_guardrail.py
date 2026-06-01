"""CI guardrail: every agent fence ≤ 400 lines (spec §2)."""

from pathlib import Path

import pytest


FENCE_DIR = Path(__file__).parent.parent.parent / "perfxpert" / "agents" / "fence"

REQUIRED_FENCE_FILES = [
    "always.md",
    "root.md",
    "analysis.md",
    "recommendation.md",
    "correctness.md",
    "compute_specialist.md",
    "memory_specialist.md",
    "latency_specialist.md",
]


@pytest.mark.parametrize("name", REQUIRED_FENCE_FILES)
def test_fence_file_within_400_lines(name):
    """Each agent ships a fence slice under agents/fence/. If missing,
    test is skipped (a split fence is required for the agent runtime)."""
    path = FENCE_DIR / name
    if not path.exists():
        pytest.skip(f"{name} not yet present")
    n = path.read_text().count("\n") + 1
    assert n <= 400, f"{name} has {n} lines (cap 400)"
