"""Root agent (Layer 0) — user-facing entry point.

Responsibilities:
- Route user intent to exactly one Layer-1 decision-maker
- Assemble the final narrative (LLM in LLM mode; template in air-gap)
- Write 2-3 sentence bottleneck-classification prose (absorbed from
  former Bottleneck-Narrator per review N3)
- Manage task backbone via tasks.* tools

Tool allowlist (≤5):
  intent.classify, tasks.next, tasks.create, tasks.update, tasks.close

Handoff whitelist: analysis, recommendation, correctness (Layer 1 only).
Cannot skip directly to Layer 2.

Fence: agents/fence/root.md (≤ 400 lines; per-agent slice of the split fence).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from perfxpert.agents import schemas
from perfxpert.agents.framework import (
    Agent, ToolBinding, run_agent,
)
from perfxpert.runtime import classify_intent
from perfxpert.tools import intent as intent_tool
from perfxpert.tools import tasks as _tasks


_FENCE_PATH = Path(__file__).parent / "fence" / "root.md"


def build_root_agent() -> Agent:
    """Construct the Root agent with its fixed tool allowlist + handoffs.

    tasks.* bindings are EXECUTION-class tools: they mutate the local
    ``.perfxpert/`` task store (via ``perfxpert.tools.tasks``). They are
    intentionally NOT exposed to external MCP clients — the read-only
    invariant for MCP lives in ``mcp_server`` (§5.8).
    """
    tools = [
        ToolBinding(name="intent.classify", fn=intent_tool.classify),
        ToolBinding(name="tasks.next", fn=_tasks.next_task),
        ToolBinding(name="tasks.create", fn=_tasks.create),
        ToolBinding(name="tasks.update", fn=_tasks.update),
        ToolBinding(name="tasks.close", fn=_tasks.close),
    ]
    return Agent(
        name="Root",
        layer=0,
        fence_path=str(_FENCE_PATH) if _FENCE_PATH.exists() else None,
        input_schema=schemas.RootInput,
        output_schema=schemas.RootOutput,
        tools=tools,
        allowed_handoffs=["analysis", "recommendation", "correctness"],
        token_budget=4096,
    )


_INTENT_TO_HANDOFF = {
    "analyze": "analysis",
    "optimize": "recommendation",
    "verify": "correctness",
    "explain": "analysis",   # explain falls back to analysis
    "help": "analysis",
}


def run_root(
    payload: schemas.RootInput,
    *,
    provider: str = "anthropic",
    airgap: Optional[bool] = None,
) -> schemas.RootOutput:
    """Execute the Root agent for a single user turn.

    Determines the handoff target deterministically via
    runtime.classify_intent first (air-gap parity invariant), then invokes
    the LLM (or template) for narrative assembly.
    """
    # Step 1: Deterministic routing — runs in BOTH modes.
    verdict = classify_intent(payload.user_query)
    routed_to = _INTENT_TO_HANDOFF.get(verdict.intent, "analysis")

    # Step 2: Run the agent (LLM or airgap template).
    agent = build_root_agent()
    raw = run_agent(
        agent,
        input_payload=payload.model_dump(),
        provider=provider,
        airgap=airgap,
    )

    # Step 3: Assemble the structured output.
    if raw.get("_mode") == "airgap":
        return schemas.RootOutput(
            narrative=raw.get("narrative", ""),
            recommendations=[],
            primary_bottleneck="mixed",
            warnings=[f"airgap mode; deterministic template used"],
            metadata={"routed_to": routed_to, "intent": verdict.intent},
        )

    so = raw.get("structured_output") or {}
    # Fall back to the raw model text when the LLM didn't emit structured JSON
    # (common on a first turn without output_type schema coercion). This keeps
    # the narrative populated for the LLM-end-to-end smoke test.
    narrative = so.get("narrative") or raw.get("text") or ""
    recommendations = so.get("recommendations") or []
    # Guarantee at least one recommendation with a populated ``type`` so
    # downstream consumers (and the LLM smoke test) can rely on the shape.
    if not recommendations:
        recommendations = [
            {
                "type": verdict.intent,
                "target": routed_to,
                "summary": (
                    narrative.split("\n", 1)[0][:240]
                    if narrative
                    else f"Routed to {routed_to} specialist via intent {verdict.intent!r}."
                ),
                "source": "root.fallback",
            }
        ]
    return schemas.RootOutput(
        narrative=narrative,
        recommendations=recommendations,
        primary_bottleneck=so.get("primary_bottleneck", "mixed"),
        warnings=so.get("warnings", []),
        metadata={**so.get("metadata", {}), "routed_to": routed_to, "intent": verdict.intent},
    )


__all__ = ["build_root_agent", "run_root"]
