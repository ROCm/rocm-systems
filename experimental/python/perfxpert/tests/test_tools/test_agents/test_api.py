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
    "agent_diff_specialist",
)

# Non-agent callables that live alongside the agent mirrors in
# ``perfxpert.api``. Confluence row #7 added ``trace_diff_diff_runs``
# so the Python API has first-class access to the diff engine.
_NON_AGENT_API_NAMES = (
    "trace_diff_diff_runs",
)


def test_api_exports_exactly_the_agent_tools_plus_trace_diff() -> None:
    """``api.__all__`` lists the 8 agent callables + the trace-diff mirror."""
    from perfxpert import api

    expected = tuple(sorted(_AGENT_NAMES + _NON_AGENT_API_NAMES))
    assert tuple(sorted(api.__all__)) == expected


def test_api_trace_diff_diff_runs_is_the_tool_function():
    """``perfxpert.api.trace_diff_diff_runs`` IS the same function object
    as ``perfxpert.tools.trace_diff.diff_runs`` — no wrapper, no shim."""
    from perfxpert import api
    from perfxpert.tools.trace_diff import diff_runs

    assert api.trace_diff_diff_runs is diff_runs


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
