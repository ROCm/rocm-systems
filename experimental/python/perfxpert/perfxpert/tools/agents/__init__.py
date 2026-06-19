# XXX(docs): tool count is now 43, agent tools available (8 after diff_specialist)
"""perfxpert.tools.agents — each agent in the hierarchy as an MCP tool.

Every agent the perfxpert runtime ships (Root, Analysis, Recommendation,
Correctness, Compute-Specialist, Memory-Specialist, Latency-Specialist,
Trace-Diff-Specialist) is exposed here as a READ_ONLY MCP tool. The MCP
auto-discovery registry walks this sub-package and registers the 8
callables so backend TUIs (opencode / claude / codex / gemini) can
invoke any agent directly — the backend LLM picks which agent to call
based on the user's intent.

The same 8 functions are mirrored 1:1 at :mod:`perfxpert.api` so Python
callers embedding perfxpert can reach the same entry points without
standing up an MCP server.
"""

from perfxpert.tools.agents.analysis import agent_analysis
from perfxpert.tools.agents.compute import agent_compute_specialist
from perfxpert.tools.agents.correctness import agent_correctness
from perfxpert.tools.agents.diff import agent_diff_specialist
from perfxpert.tools.agents.latency import agent_latency_specialist
from perfxpert.tools.agents.memory import agent_memory_specialist
from perfxpert.tools.agents.recommendation import agent_recommendation
from perfxpert.tools.agents.root import agent_root

__all__ = [
    "agent_root",
    "agent_analysis",
    "agent_recommendation",
    "agent_correctness",
    "agent_compute_specialist",
    "agent_memory_specialist",
    "agent_latency_specialist",
    "agent_diff_specialist",
]
