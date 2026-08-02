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

from functools import partial
from pathlib import Path
from typing import Optional

from perfxpert.agents import schemas
from perfxpert.agents.framework import Agent, ToolBinding, run_agent
from perfxpert.runtime import classify_intent
from perfxpert.tools import intent as intent_tool
from perfxpert.tools import tasks as tasks_tool


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
        ToolBinding(name="tasks.next", fn=tasks_tool.next_task),
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
    "explain": "analysis",
    "help": "analysis",
}


def _deterministic_primary_bottleneck(
    payload: schemas.RootInput,
    *,
    routed_to: str,
) -> tuple[str, list[str]]:
    """Best-effort deterministic bottleneck for Root, in either mode.

    The Root template does not invoke any Layer-1 agents, but the batch
    ``perfxpert analyze`` path still expects Root to thread through the
    deterministic Analysis verdict when a profiling DB is present. Without
    this, JSON/webview drift to ``mixed`` even when Analysis already
    classifies the run as latency/compute/memory bound.

    Used by both modes so Root's classification is a rule output rather than
    a model claim — which is what keeps the two modes in agreement.
    """
    if routed_to not in {"analysis", "recommendation"} or not payload.database_path:
        return "mixed", []

    from perfxpert.agents import analysis as analysis_module

    opts = dict(payload.analysis_options or {})
    try:
        result = analysis_module.run_analysis(
            schemas.AnalysisInput(
                database_path=payload.database_path,
                top_kernels=int(opts.get("top_kernels") or 10),
                att_dir=opts.get("att_dir"),
                min_duration=float(opts.get("min_duration") or 0.0),
            ),
            airgap=True,
        )
        return result.primary_bottleneck, []
    except Exception as exc:  # pragma: no cover - defensive fallback
        return "mixed", [f"airgap analysis fallback failed: {exc}"]


def _routing_summary(intent: str, routed_to: str) -> list[dict]:
    """Root's ``recommendations`` entry: a record of where the query went.

    Root does not itself rank optimizations -- Recommendation does. What Root
    can honestly report is its own routing decision, which is a rule output
    and therefore identical in both modes.

    The summary is built from the routing decision rather than sliced out of
    the narrative. Embedding model prose here would put unlabelled model text
    inside the vetted-lane field for any consumer that renders it, and would
    make the two modes disagree. The narrative is carried in its own field
    for callers that want it.
    """
    return [
        {
            "type": intent,
            "target": routed_to,
            "summary": f"Routed to {routed_to} specialist via intent {intent!r}.",
            "source": "root.routing",
        }
    ]


def run_root(
    payload: schemas.RootInput,
    *,
    provider: str = "anthropic",
    airgap: Optional[bool] = None,
) -> schemas.RootOutput:
    """Execute the Root agent for a single user turn."""
    verdict = classify_intent(payload.user_query)
    routed_to = _INTENT_TO_HANDOFF.get(verdict.intent, "analysis")

    agent = _build_root_agent_for_payload(payload)
    raw = run_agent(
        agent,
        input_payload=payload.model_dump(),
        provider=provider,
        airgap=airgap,
    )

    # Classification and routing are rule outputs in both modes. Root is
    # Layer 0 and holds no exploration capability, so nothing it publishes
    # may depend on what the model asserted -- only how it is worded.
    primary_bottleneck, extra_warnings = _deterministic_primary_bottleneck(
        payload,
        routed_to=routed_to,
    )

    if raw.get("_mode") == "airgap":
        return schemas.RootOutput(
            narrative=raw.get("narrative", ""),
            recommendations=_routing_summary(verdict.intent, routed_to),
            primary_bottleneck=primary_bottleneck,
            warnings=["airgap mode; deterministic template used", *extra_warnings],
            metadata={"routed_to": routed_to, "intent": verdict.intent},
        )

    so = raw.get("structured_output") or {}
    narrative = so.get("narrative") or raw.get("text") or ""

    return schemas.RootOutput(
        narrative=narrative,
        # Deliberately not ``so.get("recommendations")``. This field is the
        # vetted lane, and Root is the entry point most callers use, so a
        # model-written entry here would reach a user as advice with nothing
        # marking it unproven. Ranked advice comes from Recommendation.
        recommendations=_routing_summary(verdict.intent, routed_to),
        primary_bottleneck=primary_bottleneck,
        warnings=[*extra_warnings],
        metadata={"routed_to": routed_to, "intent": verdict.intent},
    )


__all__ = ["build_root_agent", "run_root"]
