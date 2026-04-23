"""Correctness decision-maker (Layer 1).

Consumes a GateVerdict from runtime.gate_cascade.evaluate (NEVER invokes
gates directly — spec §5.0). Narrates the verdict to Root, proposes
alternatives on regressions, creates follow-up tasks on rejects.

Tool allowlist (3 of 5 used — intentionally NO execution tools):
  tasks.query_by_kernel, tasks.create, trace_fingerprint.fingerprint

Handoff whitelist: [] (Layer-1 returns to Root).

Absorbs former Revert-Advisor agent per design-review N3: on `regressed`,
propose an alternative technique NOT already present in
tasks.query_by_kernel(kernel_name) history (closes FMEA gap on
Revert-Advisor recycling tried ideas).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from perfxpert.agents import schemas
from perfxpert.agents.framework import Agent, ToolBinding, run_agent
from perfxpert.tools import tasks as tasks_tool
from perfxpert.tools import trace_fingerprint


_FENCE_PATH = Path(__file__).parent / "fence" / "correctness.md"


# -- Module-level delegators (for test injection) -------------------------

def _tasks_query_by_kernel(kernel_name: str) -> List[Dict[str, Any]]:
    return tasks_tool.query_by_kernel(kernel_name)


def _tasks_create(**kw) -> str:
    return tasks_tool.create(**kw)


# -- Builder --------------------------------------------------------------

def build_correctness_agent() -> Agent:
    tools = [
        ToolBinding(name="tasks.query_by_kernel", fn=_tasks_query_by_kernel),
        ToolBinding(name="tasks.create", fn=_tasks_create),
        ToolBinding(name="trace_fingerprint.fingerprint", fn=trace_fingerprint.fingerprint),
    ]
    return Agent(
        name="Correctness",
        layer=1,
        fence_path=str(_FENCE_PATH) if _FENCE_PATH.exists() else None,
        input_schema=schemas.CorrectnessInput,
        output_schema=schemas.CorrectnessOutput,
        tools=tools,
        allowed_handoffs=[],   # returns to Root
        token_budget=3072,
    )


# -- Deterministic narrative template -------------------------------------

def _airgap_narrative(v: schemas.GateVerdictModel) -> str:
    gate = v.failing_gate or "all"
    return f"Gate {gate} {v.status}: {v.detail}"


# -- Runner ---------------------------------------------------------------

def run_correctness(
    payload: schemas.CorrectnessInput,
    *,
    provider: str = "anthropic",
    airgap: Optional[bool] = None,
) -> schemas.CorrectnessOutput:
    verdict = payload.gate_verdict

    # Map verdict → action (deterministic).
    if verdict.status == "pass":
        action = "accept"
    elif verdict.status == "regressed":
        action = "revert"
    else:  # "reject"
        action = "reject_and_log"

    # For regression: propose alternative not in history.
    alternative: Optional[str] = None
    if verdict.status == "regressed":
        history = _tasks_query_by_kernel(payload.kernel_name or "")
        tried = {h.get("technique") for h in history}
        tried.add(payload.last_technique)
        # In airgap/fallback, no LLM to suggest — leave empty; the spec allows
        # a structured warning instead of a forced suggestion.
        alternative = None
        for candidate in history:
            c = candidate.get("candidate_alternative")
            if c and c not in tried:
                alternative = c
                break

    # For reject: create follow-up task.
    follow_up_task_id: Optional[str] = None
    if verdict.status == "reject":
        follow_up_task_id = _tasks_create(
            title=f"Investigate rejected optimization: {verdict.failing_gate}",
            meta={"verdict": verdict.model_dump(), "kernel": payload.kernel_name},
        )

    # Narrative: template in airgap, LLM otherwise.
    agent = build_correctness_agent()
    raw = run_agent(
        agent,
        input_payload={**payload.model_dump()},
        provider=provider,
        airgap=airgap,
    )

    if raw.get("_mode") == "airgap":
        narrative = _airgap_narrative(verdict)
    else:
        so = raw.get("structured_output") or {}
        narrative = so.get("narrative", _airgap_narrative(verdict))
        alternative = so.get("alternative_technique", alternative)

    return schemas.CorrectnessOutput(
        verdict=verdict.status,
        action=action,
        narrative=narrative,
        alternative_technique=alternative,
        follow_up_task_id=follow_up_task_id,
    )


__all__ = ["build_correctness_agent", "run_correctness"]
