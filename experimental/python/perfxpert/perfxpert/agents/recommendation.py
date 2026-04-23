"""Recommendation decision-maker (Layer 1).

Dispatches to one of the three Layer-2 specialists (compute / memory / latency)
based on findings.primary_bottleneck; ranks + dedups outputs.

Tool allowlist (3 of 5 used):
  plateau.check, trace.fingerprint, profiling.fill_gap

Handoff whitelist: compute_specialist, memory_specialist, latency_specialist
(Layer 2 only). Cannot handoff to Layer 1 peers (Analysis / Correctness).

Dedup strategy: hash each technique dict (sorted keys) and drop if hash is
in RecommendationInput.seen_recommendation_hashes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from perfxpert.agents import (
    compute_specialist, latency_specialist, memory_specialist, schemas,
)
from perfxpert.agents.framework import Agent, ToolBinding, run_agent
from perfxpert.tools import plateau, profiling, trace_fingerprint


_FENCE_PATH = Path(__file__).parent / "fence" / "recommendation.md"


# -- Thin delegators (module-level for test injection) --------------------

def _run_specialist_compute(payload, **kw):
    return compute_specialist.run_compute_specialist(payload, **kw)


def _run_specialist_memory(payload, **kw):
    return memory_specialist.run_memory_specialist(payload, **kw)


def _run_specialist_latency(payload, **kw):
    return latency_specialist.run_latency_specialist(payload, **kw)


def _plateau_check(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    return plateau.check(history=history)


# -- Builder --------------------------------------------------------------

def build_recommendation_agent() -> Agent:
    tools = [
        ToolBinding(name="plateau.check", fn=_plateau_check),
        ToolBinding(name="trace.fingerprint", fn=trace_fingerprint.fingerprint),
        ToolBinding(name="profiling.fill_gap", fn=profiling.fill_gap),
    ]
    return Agent(
        name="Recommendation",
        layer=1,
        fence_path=str(_FENCE_PATH) if _FENCE_PATH.exists() else None,
        input_schema=schemas.RecommendationInput,
        output_schema=schemas.RecommendationOutput,
        tools=tools,
        allowed_handoffs=[
            "compute_specialist", "memory_specialist", "latency_specialist",
        ],
        token_budget=4096,
    )


# -- Dispatch + dedup -----------------------------------------------------

def _hash_technique(t: Dict[str, Any]) -> str:
    """Stable hash for dedup."""
    # Use only the "name" field as the key to avoid over-specifying; matches
    # the existing _hash_recommendation pattern in interactive.py.
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

    # Plateau check (always runs — informs output flag)
    plateau_info = _plateau_check(payload.edit_history)
    plateau_detected = bool(plateau_info.get("plateau_detected", False))

    # Dispatch
    if bottleneck == "compute":
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
    else:
        specialist_used = "none"
        techniques = []

    # Dedup against seen hashes
    techniques = _dedup(techniques, payload.seen_recommendation_hashes)

    return schemas.RecommendationOutput(
        recommendations=techniques,
        specialist_used=specialist_used,
        plateau_detected=plateau_detected,
    )


__all__ = ["build_recommendation_agent", "run_recommendation"]
