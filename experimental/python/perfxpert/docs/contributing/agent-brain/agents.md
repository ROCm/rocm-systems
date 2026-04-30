# Contributing: new agent

## What you're adding

A new node in the 8-agent hierarchy that orchestrates tools to solve a
narrowly-scoped problem. Agents route via typed handoffs and run within the
5-gate correctness cascade.

## File locations

- Agent implementation: `perfxpert/agents/<name>.py`
- Fence instructions: `perfxpert/agents/fence/<name>.md` (≤ 400 lines)
- Handoff schemas: `perfxpert/agents/schemas.py` (typed dataclasses)
- Registration: agent added to Root or Recommendation `AGENT_SPEC.tools` allowlist
- Tests: `tests/test_agents/test_<name>.py` (≥ 5 isolation tests per agent)

## Key constraints

- Fence file ≤ 400 lines (CI-enforced)
- ≤ 5 tools in allowlist (CI-enforced)
- ≤ 10 input fields / ≤ 5 output fields per handoff (CI-enforced)
- Handoff schemas must be immutable dataclasses or Pydantic models

## Template

### Agent skeleton

```python
# SKIP-SAMPLE — template: <name>/<Name> placeholders, pedagogical scaffolding
"""<name> — <one-sentence purpose>."""

from typing import Any, Dict

from perfxpert.agents.framework import Agent, AgentSpec
from perfxpert.agents.schemas import HandoffSchema
from perfxpert.tools import tool1, tool2


class NameHandoff(HandoffSchema):
    """Handoff schema: what this agent outputs."""
    result: str
    confidence: float


AGENT_SPEC = AgentSpec(
    name="<name>",
    fence_path="perfxpert/agents/fence/<name>.md",
    tools=[tool1, tool2],
    handoff_schema=NameHandoff,
    description="<purpose>"
)


async def run(context: Dict[str, Any]) -> NameHandoff:
    """Main entry point for this agent."""
    # Call tools, process results
    return NameHandoff(result=..., confidence=...)
```

### Fence skeleton

Use `perfxpert/agents/fence/diff_specialist.md` as the canonical
template — it has the definitive section layout (Purpose, Tools you
can use, Instructions, Output contract, plus any agent-specific
sub-sections). The Root (`root.md`) and Recommendation
(`recommendation.md`) fences have been aligned to the same
diff_specialist template during the final doc pass, so new agents
should mirror that structure to stay consistent.

```markdown
# <Name> Agent

## Purpose
<What this agent does and when it runs in the pipeline.>

## Tools you can use
1. tool1() — <description>
2. tool2() — <description>

## Instructions
<Narrative instructions for the LLM on how to solve the problem using the tools.>

## Output contract
<Fields the agent MUST populate, matching the handoff schema.>
```

## Schema constraints (CI-enforced)

- Agent fence ≤ 400 lines
- Tools allowlist ≤ 5 items
- Handoff schema ≤ 10 input / ≤ 5 output fields
- Framework adapter in `agents/framework.py` handles provider routing

## Tests you must add

Minimum 5 isolation tests in `tests/test_agents/test_<name>.py`:

- `test_<name>_loads_spec()` — metadata intact
- `test_<name>_calls_tool_1()` — tool invocation
- `test_<name>_calls_tool_2()` — tool invocation
- `test_<name>_returns_valid_handoff()` — output shape
- `test_<name>_integrates_in_full_flow()` — end-to-end

Plus one integration test in the full-scenario suite that exercises this agent.

## Review requirements

- 2 core reviewers (architectural change)
- Fence reviewed for clarity + correctness
- CI green (isolation + integration)

## Common pitfalls

- Don't add tools not used in the fence narrative
- Handoff fields are load-bearing — agents downstream depend on them; breaking changes require an RFC
- Fence is narrative LLM instructions, not a spec — be clear but not verbose
- If adding to Root or Recommendation, update their `AGENT_SPEC.tools` allowlist + tests

## Related docs

- Design spec: §2 (agent tree + responsibilities)
- Existing agents under `perfxpert/agents/` as references
- Framework adapter: `perfxpert/agents/framework.py`
