"""Public Python API for PerfXpert's agent hierarchy.

1:1 mirror of the MCP tools under :mod:`perfxpert.tools.agents`. Each
callable exported here IS the same function the MCP server wraps — the
Python API and the MCP surface share a single implementation. Use this
module to embed PerfXpert's analysis brain in your own tooling without
running the MCP server.

Examples:
    >>> from perfxpert import api
    >>> verdict = api.agent_root(airgap=True, user_query="why slow?")
    >>> verdict["primary_bottleneck"]
    'mixed'

    >>> # Direct call into a Layer-2 specialist when the bottleneck is known:
    >>> techniques = api.agent_compute_specialist(
    ...     input={"gfx_id": "gfx942", "hot_kernels": hot},
    ...     airgap=True,
    ... )

Every callable honors ``PERFXPERT_AIRGAP=1`` plus the shared provider
selection semantics from :func:`perfxpert.agents.runtime.build_session`.
"""

from perfxpert.tools.agents.analysis import agent_analysis
from perfxpert.tools.agents.compute import agent_compute_specialist
from perfxpert.tools.agents.correctness import agent_correctness
from perfxpert.tools.agents.diff import agent_diff_specialist
from perfxpert.tools.agents.latency import agent_latency_specialist
from perfxpert.tools.agents.memory import agent_memory_specialist
from perfxpert.tools.agents.recommendation import agent_recommendation
from perfxpert.tools.agents.root import agent_root
from perfxpert.tools.trace_diff import diff_runs as trace_diff_diff_runs

__all__ = [
    "agent_root",
    "agent_analysis",
    "agent_recommendation",
    "agent_correctness",
    "agent_compute_specialist",
    "agent_memory_specialist",
    "agent_latency_specialist",
    "agent_diff_specialist",
    "trace_diff_diff_runs",
]
