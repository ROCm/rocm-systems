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

Fence: agents/fence/root.md (≤ 400 lines; Phase 2 decomposed the monolith).
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, Dict, Optional

from perfxpert.agents import schemas
from perfxpert.agents.framework import (
    Agent, ToolBinding, run_agent,
)
from perfxpert.runtime import classify_intent
from perfxpert.tools import intent as intent_tool
from perfxpert.tools import tasks as tasks_tool


_FENCE_PATH = Path(__file__).parent / "fence" / "root.md"


def build_root_agent() -> Agent:
    """Construct the Root agent with its fixed tool allowlist + handoffs."""
    tools = [
        ToolBinding(name="intent.classify", fn=intent_tool.classify),
        ToolBinding(name="tasks.next", fn=tasks_tool.next),
        ToolBinding(name="tasks.create", fn=tasks_tool.create),
        ToolBinding(name="tasks.update", fn=tasks_tool.update),
        ToolBinding(name="tasks.close", fn=tasks_tool.close),
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


def _task_root(payload: schemas.RootInput) -> Optional[str]:
    if payload.source_dir:
        return payload.source_dir
    if payload.database_path:
        return str(Path(payload.database_path).resolve().parent)
    return None


def _build_root_agent_for_payload(payload: schemas.RootInput) -> Agent:
    task_root = _task_root(payload)
    if task_root is None:
        return build_root_agent()

    tools = [
        ToolBinding(name="intent.classify", fn=intent_tool.classify),
        ToolBinding(name="tasks.next", fn=partial(tasks_tool.next_at, task_root)),
        ToolBinding(name="tasks.create", fn=partial(tasks_tool.create_at, task_root)),
        ToolBinding(name="tasks.update", fn=partial(tasks_tool.update_at, task_root)),
        ToolBinding(name="tasks.close", fn=partial(tasks_tool.close_at, task_root)),
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
    agent = _build_root_agent_for_payload(payload)
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
    return schemas.RootOutput(
        narrative=so.get("narrative", ""),
        recommendations=so.get("recommendations", []),
        primary_bottleneck=so.get("primary_bottleneck", "mixed"),
        warnings=so.get("warnings", []),
        metadata={**so.get("metadata", {}), "routed_to": routed_to, "intent": verdict.intent},
    )


__all__ = ["build_root_agent", "run_root"]
