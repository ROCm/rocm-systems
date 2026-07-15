"""Recommendation decision-maker (Layer 1).

Dispatches to one of the three Layer-2 specialists (compute / memory / latency)
based on findings.primary_bottleneck; ranks + dedups outputs.

Tool allowlist (3 of 5 used):
  plateau.check, trace_fingerprint.fingerprint, profiling.fill_gap

Handoff whitelist: compute_specialist, memory_specialist, latency_specialist
(Layer 2 only). Cannot handoff to Layer 1 peers (Analysis / Correctness).

Dedup strategy: hash each technique dict (sorted keys) and drop if hash is
in RecommendationInput.seen_recommendation_hashes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from perfxpert.agents import compute_specialist, latency_specialist, memory_specialist, schemas
from perfxpert.agents.framework import Agent, ToolBinding, run_agent
from perfxpert.tools import plateau, profiling, trace_fingerprint


_FENCE_PATH = Path(__file__).parent / "fence" / "recommendation.md"


_DATA_INSUFFICIENT_WARNING = """\
╔══════════════════════════════════════════════════════════════════════════╗
║  WARNING: Bottleneck classifier returned data_insufficient              ║
║                                                                          ║
║  The trace database contains no hardware counter data (--pmc was not    ║
║  used when profiling). PerfXpert cannot classify the bottleneck type     ║
║  without counter evidence, and will NOT generate recommendations that   ║
║  might be wrong.                                                         ║
║                                                                          ║
║  ACTION: Re-profile your workload with hardware counters enabled.        ║
║  Run three separate passes (TCC isolation rule — FETCH_SIZE and          ║
║  WRITE_SIZE must each be in their own --pmc pass):                      ║
║                                                                          ║
║    Pass 1 (compute utilization):                                        ║
║      rocprofv3 --sys-trace \\                                            ║
║        --pmc SQ_WAVES,SQ_INSTS_VALU,SQ_INSTS_VALU_MFMA,GRBM_GUI_ACTIVE,GRBM_COUNT \\
║        -- ./your_app                                                    ║
║                                                                          ║
║    Pass 2 (HBM fetch bandwidth — isolated):                             ║
║      rocprofv3 --sys-trace --pmc FETCH_SIZE -- ./your_app              ║
║                                                                          ║
║    Pass 3 (HBM write bandwidth — isolated):                             ║
║      rocprofv3 --sys-trace --pmc WRITE_SIZE -- ./your_app              ║
║                                                                          ║
║  Then merge the output databases before re-running analysis.            ║
╚══════════════════════════════════════════════════════════════════════════╝
"""


def _warn_data_insufficient() -> None:
    print(_DATA_INSUFFICIENT_WARNING, file=sys.stderr, flush=True)


def _run_specialist_compute(payload, **kw):
    return compute_specialist.run_compute_specialist(payload, **kw)


def _run_specialist_memory(payload, **kw):
    return memory_specialist.run_memory_specialist(payload, **kw)


def _run_specialist_latency(payload, **kw):
    return latency_specialist.run_latency_specialist(payload, **kw)


def _plateau_check(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    return plateau.check(history=history)


def build_recommendation_agent() -> Agent:
    tools = [
        ToolBinding(name="plateau.check", fn=_plateau_check),
        ToolBinding(name="trace_fingerprint.fingerprint", fn=trace_fingerprint.fingerprint),
        ToolBinding(name="profiling.fill_gap", fn=profiling.fill_gap),
    ]
    return Agent(
        name="Recommendation",
        layer=1,
        fence_path=str(_FENCE_PATH) if _FENCE_PATH.exists() else None,
        input_schema=schemas.RecommendationInput,
        output_schema=schemas.RecommendationOutput,
        tools=tools,
        allowed_handoffs=["compute_specialist", "memory_specialist", "latency_specialist"],
        token_budget=4096,
    )


def _hash_technique(t: Dict[str, Any]) -> str:
    key = {"name": t.get("name", "")}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


def _dedup(techniques: List[Dict[str, Any]], seen: List[str]) -> List[Dict[str, Any]]:
    seen_set = set(seen)
    return [t for t in techniques if _hash_technique(t) not in seen_set]


def _require_gfx_id(payload: schemas.RecommendationInput) -> str:
    if not payload.gfx_id:
        raise ValueError("RecommendationInput.gfx_id is required for specialist routing")
    return payload.gfx_id


def run_recommendation(
    payload: schemas.RecommendationInput,
    *,
    provider: str = "anthropic",
    airgap: Optional[bool] = None,
) -> schemas.RecommendationOutput:
    """Route to the right specialist based on bottleneck, dedup results."""
    bottleneck = payload.findings.primary_bottleneck
    plateau_info = _plateau_check(payload.edit_history)
    plateau_detected = bool(plateau_info.get("plateau_detected", False))

    if bottleneck == "data_insufficient":
        _warn_data_insufficient()
        specialist_used = "none"
        techniques = []
    elif bottleneck == "compute":
        gfx_id = _require_gfx_id(payload)
        specialist_used = "compute"
        spec_input = schemas.ComputeSpecialistInput(
            gfx_id=gfx_id,
            hot_kernels=payload.findings.hot_kernels,
            counter_data={},
        )
        spec_out = _run_specialist_compute(spec_input, provider=provider, airgap=airgap)
        techniques = list(spec_out.techniques)
    elif bottleneck == "memory_transfer":
        gfx_id = _require_gfx_id(payload)
        specialist_used = "memory"
        spec_input = schemas.MemorySpecialistInput(
            gfx_id=gfx_id,
            hot_kernels=payload.findings.hot_kernels,
        )
        spec_out = _run_specialist_memory(spec_input, provider=provider, airgap=airgap)
        techniques = list(spec_out.techniques)
    elif bottleneck in ("latency", "api_overhead"):
        gfx_id = _require_gfx_id(payload)
        specialist_used = "latency"
        spec_input = schemas.LatencySpecialistInput(
            gfx_id=gfx_id,
            hot_kernels=payload.findings.hot_kernels,
            api_overhead_pct=payload.findings.time_breakdown.get("api_pct", 0.0),
        )
        spec_out = _run_specialist_latency(spec_input, provider=provider, airgap=airgap)
        techniques = list(spec_out.techniques)
    elif bottleneck == "mixed":
        specialist_used = "none"
        techniques = [
            {
                "name": "mixed_bottleneck_triage",
                "category": "triage",
                "priority": "medium",
                "title": "Mixed bottleneck — no single dominant stall source",
                "description": (
                    "The classifier did not find one dominant bottleneck. "
                    "Multiple subsystems are contributing roughly equally. "
                    "Re-profile with ATT (--att) to identify the dominant stall "
                    "source at instruction level, or run `perfxpert doctor` to "
                    "verify counter coverage and ensure all required --pmc passes "
                    "were collected."
                ),
                "rationale": (
                    "Bottleneck classification is mixed. "
                    "ATT or broader counter coverage needed to isolate root cause."
                ),
                "estimated_impact": "Unknown until dominant stall source is identified",
            }
        ]
    else:
        raise ValueError(
            f"unhandled bottleneck type {bottleneck!r}. "
            "Known types: compute | memory_transfer | latency | api_overhead | "
            "mixed | data_insufficient. Add a branch or fix the classifier."
        )

    techniques = _dedup(techniques, payload.seen_recommendation_hashes)
    return schemas.RecommendationOutput(
        recommendations=techniques,
        specialist_used=specialist_used,
        plateau_detected=plateau_detected,
    )


__all__ = ["build_recommendation_agent", "run_recommendation"]
