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

import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional

from perfxpert.agents import schemas
from perfxpert.agents.framework import Agent, ToolBinding, run_agent
from perfxpert.tools import tasks as tasks_tool
from perfxpert.tools import trace_fingerprint


_FENCE_PATH = Path(__file__).parent / "fence" / "correctness.md"


# -- Module-level delegators (for test injection) -------------------------

def _tasks_query_by_kernel(kernel_name: str, root: Optional[str] = None) -> List[Dict[str, Any]]:
    if root:
        return tasks_tool.query_by_kernel_at(root, kernel_name)
    return tasks_tool.query_by_kernel(kernel_name)


def _tasks_create(*, root: Optional[str] = None, **kw) -> str:
    if root:
        return tasks_tool.create_at(root, **kw)
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


def _history_value(entry: Dict[str, Any], key: str) -> Any:
    if key in entry:
        return entry.get(key)
    meta = entry.get("meta")
    if isinstance(meta, dict):
        return meta.get(key)
    return None


def _validate_tasks_create_binding() -> None:
    sig = inspect.signature(_tasks_create)
    params = sig.parameters.values()
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params):
        return
    names = {param.name for param in sig.parameters.values()}
    required = {"title", "meta"}
    missing = sorted(required - names)
    if missing:
        raise TypeError(
            "tasks.create binding must accept title and meta keyword arguments; "
            f"missing {', '.join(missing)}"
        )


def _create_reject_follow_up(
    verdict: schemas.GateVerdictModel,
    kernel_name: Optional[str],
    source_dir: Optional[str],
) -> str:
    _validate_tasks_create_binding()
    title = f"Investigate rejected optimization: {verdict.failing_gate}"
    meta = {"verdict": verdict.model_dump(), "kernel": kernel_name}
    if not isinstance(title, str) or not title:
        raise TypeError("follow-up task title must be a non-empty string")
    if not isinstance(meta, dict):
        raise TypeError("follow-up task meta must be a dict")
    return _tasks_create(root=source_dir, title=title, meta=meta)


def _select_untried_alternative(
    candidate: Optional[str], tried: set[str], fallback: Optional[str]
) -> Optional[str]:
    if candidate and candidate not in tried:
        return candidate
    return fallback


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
    tried: set[str] = set()
    if verdict.status == "regressed":
        history = _tasks_query_by_kernel(
            payload.kernel_name or "",
            root=payload.source_dir,
        )
        tried = {_history_value(h, "technique") for h in history}
        tried.add(payload.last_technique)
        tried.discard(None)
        # In airgap/fallback, no LLM to suggest — leave empty; the spec allows
        # a structured warning instead of a forced suggestion. Regressions do
        # not create a new task here: the current change is reverted first,
        # then Root can decide whether to schedule a follow-up attempt.
        alternative = None
        for candidate in history:
            c = _history_value(candidate, "candidate_alternative")
            if c and c not in tried:
                alternative = c
                break

    # For reject: create follow-up task.
    follow_up_task_id: Optional[str] = None
    if verdict.status == "reject":
        follow_up_task_id = _create_reject_follow_up(
            verdict,
            payload.kernel_name,
            payload.source_dir,
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
        alternative = _select_untried_alternative(
            so.get("alternative_technique"),
            tried,
            alternative,
        )

    return schemas.CorrectnessOutput(
        verdict=verdict.status,
        action=action,
        narrative=narrative,
        alternative_technique=alternative,
        follow_up_task_id=follow_up_task_id,
    )


__all__ = ["build_correctness_agent", "run_correctness"]
