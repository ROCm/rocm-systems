"""CI guardrail: no agent has any execution-class tool in its allowlist (spec §5.8).

Gates run in runtime.gate_cascade.py middleware — not inside agents.
This test enumerates every agent builder and asserts the allowlist is clean.
"""

import pytest

from perfxpert.agents import AGENT_BUILDERS


EXECUTION_TOOLS = frozenset({
    # spec §5.8 execution class
    "patch.apply", "patch.revert", "patch.verify_output",
    "compile.build", "profile.run", "anchors.check",
})


def _agent_id(builder):
    return builder.__name__


@pytest.mark.parametrize("builder", AGENT_BUILDERS, ids=_agent_id)
def test_no_execution_tool_in_allowlist(builder):
    agent = builder()
    declared = {t.name for t in agent.tools}
    intersection = declared & EXECUTION_TOOLS
    assert not intersection, (
        f"{agent.name} declares execution tool(s): {intersection}. "
        f"Execution tools belong in runtime.gate_cascade.py middleware."
    )


@pytest.mark.parametrize("builder", AGENT_BUILDERS, ids=_agent_id)
def test_allowlist_within_cap(builder):
    agent = builder()
    assert len(agent.tools) <= 5, f"{agent.name} has {len(agent.tools)} tools (cap 5)"
