"""Framework facade over OpenAI Agents SDK.

Thin abstraction layer so agent definitions depend on this facade,
not the SDK directly. Enables swapping backends (e.g. Pydantic AI)
if the SDK changes API.

Phase 1: stub with placeholder Agent and Handoff types.
Phase 3: full implementation.
"""

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class AgentSpec:
    """Placeholder — full definition in Phase 3."""
    name: str


def register_agent(spec: AgentSpec) -> Any:
    """Placeholder — full registration in Phase 3."""
    raise NotImplementedError("Agent framework facade populated in Phase 3")
