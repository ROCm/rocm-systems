"""CI guardrail: every agent fence ≤ 400 lines (spec §2).

Fences are read off the agent definitions, so the set under test is whatever
the agents actually load — not a hand-written filename list that can drift.
A missing fence is a failure, not a skip: an agent running without its fence
has no role constraints at all.
"""

from pathlib import Path

import pytest

from perfxpert.agents import AGENT_BUILDERS


FENCE_DIR = Path(__file__).parent.parent.parent / "perfxpert" / "agents" / "fence"

FENCE_LINE_CAP = 400

# Loaded by composition rather than owned by one agent.
SHARED_FENCE = "always.md"


def _agent_id(builder):
    return builder.__name__


@pytest.mark.parametrize("builder", AGENT_BUILDERS, ids=_agent_id)
def test_agent_declares_a_fence(builder):
    agent = builder()
    assert agent.fence_path is not None, (
        f"{agent.name} has no fence file; it would run without role constraints"
    )
    assert Path(agent.fence_path).exists(), (
        f"{agent.name} points at a missing fence: {agent.fence_path}"
    )


@pytest.mark.parametrize("builder", AGENT_BUILDERS, ids=_agent_id)
def test_agent_fence_within_cap(builder):
    agent = builder()
    assert agent.fence_line_count <= FENCE_LINE_CAP, (
        f"{agent.name} fence has {agent.fence_line_count} lines "
        f"(cap {FENCE_LINE_CAP})"
    )


def test_shared_fence_within_cap():
    path = FENCE_DIR / SHARED_FENCE
    assert path.exists(), f"{SHARED_FENCE} is missing from {FENCE_DIR}"
    n = path.read_text().count("\n") + 1
    assert n <= FENCE_LINE_CAP, f"{SHARED_FENCE} has {n} lines (cap {FENCE_LINE_CAP})"


def test_every_fence_file_is_claimed_by_an_agent():
    """No orphaned or duplicated slices sitting unused in the fence directory."""
    claimed = {Path(b().fence_path).name for b in AGENT_BUILDERS}
    claimed.add(SHARED_FENCE)
    on_disk = {p.name for p in FENCE_DIR.glob("*.md")}
    orphaned = on_disk - claimed
    assert not orphaned, (
        f"fence files not loaded by any agent: {sorted(orphaned)}. "
        f"Remove them or wire them up — unused fences drift from the live prompt."
    )
