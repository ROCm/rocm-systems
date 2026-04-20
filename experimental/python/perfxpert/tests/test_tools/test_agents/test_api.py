"""``perfxpert.api`` is a 1:1 mirror of the agent MCP tools.

The contract: every callable re-exported from :mod:`perfxpert.api`
is the SAME function object the MCP server would wrap — zero second
implementation. This test is a regression guard against accidental
re-implementation or shim insertion.
"""

from __future__ import annotations

import pytest


_AGENT_NAMES = (
    "agent_root",
    "agent_analysis",
    "agent_recommendation",
    "agent_correctness",
    "agent_compute_specialist",
    "agent_memory_specialist",
    "agent_latency_specialist",
)


def test_api_exports_exactly_the_seven_agent_tools() -> None:
    """``api.__all__`` lists exactly the 7 agent callables."""
    from perfxpert import api

    assert tuple(sorted(api.__all__)) == tuple(sorted(_AGENT_NAMES))


def test_api_module_docstring_references_mirror() -> None:
    """The module docstring must be explicit that this is a mirror,
    not a second implementation. Caller-facing invariant."""
    from perfxpert import api

    doc = (api.__doc__ or "").lower()
    assert "mirror" in doc, api.__doc__


@pytest.mark.parametrize("name", _AGENT_NAMES)
def test_api_callable_is_identical_to_tool_callable(name: str) -> None:
    """``perfxpert.api.agent_*`` IS the same function object as
    ``perfxpert.tools.agents.*.agent_*`` — no wrapper, no shim."""
    from perfxpert import api
    from perfxpert.tools import agents as tools_agents

    api_fn = getattr(api, name)
    tool_fn = getattr(tools_agents, name)
    assert api_fn is tool_fn, (
        f"{name!r}: perfxpert.api points at a different callable than "
        f"perfxpert.tools.agents — they must be identical"
    )


def test_api_airgap_root_smoke() -> None:
    """End-to-end sanity: ``api.agent_root(airgap=True)`` returns a dict
    with the documented RootOutput keys. No network. No DB."""
    from perfxpert import api

    result = api.agent_root(airgap=True, user_query="test")
    for key in (
        "narrative",
        "recommendations",
        "primary_bottleneck",
        "warnings",
        "metadata",
    ):
        assert key in result, f"api.agent_root missing {key!r}: {sorted(result)}"
