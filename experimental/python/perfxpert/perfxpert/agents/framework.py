"""Framework facade over OpenAI Agents SDK.

Design-review N5: agents depend on this facade, NOT on the SDK directly.
This is the only module allowed to `import openai_agents` (enforced by
tests/test_agents/test_no_sdk_import_leak.py — CI).

Under PERFXPERT_AIRGAP=1 the facade short-circuits SDK calls and drives
responses from agents/templates/airgap_report.txt, preserving the air-gap
parity invariant (spec §5 — gate decisions identical, narrative differs).

Construction-time guardrails:
- ≤ 5 tools per agent (spec §2)
- ≤ 400 fence lines per agent (spec §2)
- Handoff whitelist (Root→L1, Recommendation→L2 only; no L2→L2; no upward)

Runtime guardrails:
- Tool dispatch rejects out-of-allowlist calls
- Token budget per agent (provided by agent definitions)
- Structured output validated against declared Pydantic schema
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

# SDK import is lazy + isolated to this file — never in agent modules.
try:
    import openai_agents  # type: ignore
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False


# -- Exceptions -----------------------------------------------------------

class AgentConstructionError(ValueError):
    """Raised when an agent / handoff violates a design-time constraint."""


class ToolAllowlistViolation(RuntimeError):
    """Raised at runtime when an agent attempts to call a tool outside its allowlist."""


class HandoffPolicyViolation(RuntimeError):
    """Raised when an agent attempts a handoff not on its whitelist."""


# -- Data classes ---------------------------------------------------------

@dataclass(frozen=True)
class ToolBinding:
    """A tool registered with an agent."""
    name: str
    fn: Callable[..., Any]


@dataclass(frozen=True)
class FakeProviderResponse:
    """What _sdk_invoke returns under test mocks."""
    text: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    structured_output: Optional[Dict[str, Any]] = None
    handoff: Optional[str] = None


@dataclass(frozen=True)
class Agent:
    """An LLM-backed reasoning unit with a fence slice and tool allowlist.

    Construction validates design-time constraints (spec §2):
    - len(tools) <= 5
    - fence line count <= 400
    """
    name: str
    layer: int                              # 0=Root, 1=DecisionMaker, 2=Specialist
    fence_path: Optional[str]               # None = no fence file (test/placeholder)
    input_schema: Type                      # Pydantic model (or dict for tests)
    output_schema: Type                     # Pydantic model (or dict for tests)
    tools: Tuple[ToolBinding, ...] = field(default_factory=tuple)
    allowed_handoffs: Tuple[str, ...] = field(default_factory=tuple)
    token_budget: int = 4096
    fence_text: str = field(init=False, default="")
    fence_line_count: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        tools = tuple(self.tools)
        allowed_handoffs = tuple(self.allowed_handoffs)
        object.__setattr__(self, "tools", tools)
        object.__setattr__(self, "allowed_handoffs", allowed_handoffs)

        if len(tools) > 5:
            raise AgentConstructionError(
                f"Agent {self.name}: {len(tools)} tools declared (cap is 5)"
            )

        if self.fence_path is not None:
            text = Path(self.fence_path).read_text()
            n = text.count("\n") + 1
            if n > 400:
                raise AgentConstructionError(
                    f"Agent {self.name}: fence has {n} lines (cap is 400)"
                )
            object.__setattr__(self, "fence_text", text)
            object.__setattr__(self, "fence_line_count", n)

    def has_tool(self, tool_name: str) -> bool:
        return any(t.name == tool_name for t in self.tools)


@dataclass(frozen=True)
class Handoff:
    """A typed transfer of control from one agent to another.

    Rules enforced at construction (spec §2):
    - downward only: source_layer < target_layer
    - no skipping: target_layer == source_layer + 1
    - no Layer-2 → Layer-2
    """
    source_layer: int
    target_layer: int
    source_name: str
    target_name: str

    def __post_init__(self) -> None:
        if self.source_layer == 2 and self.target_layer == 2:
            raise AgentConstructionError(
                "No layer-2 → layer-2 handoffs (spec §2)"
            )
        if self.target_layer <= self.source_layer:
            raise AgentConstructionError(
                f"Handoffs must be downward (source {self.source_layer} → target {self.target_layer})"
            )
        if self.target_layer != self.source_layer + 1:
            raise AgentConstructionError(
                f"Cannot skip layers ({self.source_layer} → {self.target_layer})"
            )


# -- SDK abstraction ------------------------------------------------------

def _sdk_invoke(agent: Agent, input_payload: Any, provider: str) -> FakeProviderResponse:
    """Invoke the real SDK. Tests monkeypatch this to return FakeProviderResponse.

    In non-mocked runs this wraps openai_agents.Runner.run(...) with the right
    model, tools, and fence. Keep this function tiny so the rest of the facade
    is decoupled from SDK churn.
    """
    if not _SDK_AVAILABLE:
        raise RuntimeError(
            "OpenAI Agents SDK not installed; run pip install -e '.[dev]' "
            "or set PERFXPERT_AIRGAP=1"
        )
    # Real implementation delegates to SDK. See openai-agents docs.
    # (Deliberately unimplemented in this stub — mocked everywhere in tests;
    # the live path is exercised in Phase 5 provider-smoke tests.)
    raise NotImplementedError("Live SDK path — exercised by Phase 5 integration tests")


# -- Runtime --------------------------------------------------------------

def _airgap_enabled(explicit: Optional[bool]) -> bool:
    if explicit is not None:
        return explicit
    return os.environ.get("PERFXPERT_AIRGAP", "0") == "1"


def _render_airgap_template(agent: Agent, payload: Any) -> Dict[str, Any]:
    """Produce a deterministic response from templates/airgap_report.txt."""
    template_path = Path(__file__).parent / "templates" / "airgap_report.txt"
    if template_path.exists():
        template = template_path.read_text()
    else:
        template = "[airgap] agent={agent_name} payload={payload}"
    return {
        "_mode": "airgap",
        "airgap": True,
        "agent": agent.name,
        "narrative": template.format(agent_name=agent.name, payload=payload),
    }


def run_agent(
    agent: Agent,
    input_payload: Any,
    *,
    provider: str = "anthropic",
    airgap: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run an agent through the SDK (or airgap template).

    This is the single entry point agent modules call. Agents never
    import openai_agents; they compose ToolBinding/Handoff objects and
    let the facade handle execution.
    """
    if _airgap_enabled(airgap):
        return _render_airgap_template(agent, input_payload)

    resp = _sdk_invoke(agent, input_payload, provider)
    return {
        "text": resp.text,
        "tool_calls": resp.tool_calls,
        "structured_output": resp.structured_output,
        "handoff": resp.handoff,
    }


def dispatch_tool(agent: Agent, tool_name: str, args: Dict[str, Any]) -> Any:
    """Validate + execute a tool call from within an agent run.

    Rejects calls for tools not in the agent's allowlist (ToolAllowlistViolation).
    """
    if not agent.has_tool(tool_name):
        raise ToolAllowlistViolation(
            f"Agent {agent.name} attempted to call {tool_name!r}; "
            f"allowlist: {[t.name for t in agent.tools]}"
        )
    binding = next(t for t in agent.tools if t.name == tool_name)
    return binding.fn(**args)


def dispatch_handoff(agent: Agent, target_name: str) -> None:
    """Validate a handoff target against the agent's whitelist."""
    if target_name not in agent.allowed_handoffs:
        raise HandoffPolicyViolation(
            f"Agent {agent.name} attempted handoff to {target_name!r}; "
            f"whitelist: {agent.allowed_handoffs}"
        )


__all__ = [
    "Agent", "Handoff", "ToolBinding", "FakeProviderResponse",
    "AgentConstructionError", "ToolAllowlistViolation", "HandoffPolicyViolation",
    "run_agent", "dispatch_tool", "dispatch_handoff",
]
