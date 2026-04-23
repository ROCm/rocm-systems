"""Compute-Techniques Specialist (Layer 2).

Filters + ranks compute optimization techniques for a compute-bound kernel.

Tool allowlist (4 of 5 cap used):
  compute_techniques.catalog, arch.lookup_peaks, roofline.classify,
  compiler.lookup_flags

Layer-2 rule: no Layer-2 → Layer-2 handoffs; returns to Recommendation.
Air-gap fallback: sort catalog by expected_impact × (1/effort_factor),
tie-break by risk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from perfxpert.agents import schemas
from perfxpert.agents.framework import Agent, ToolBinding, run_agent
from perfxpert.tools import arch, compiler, roofline


_FENCE_PATH = Path(__file__).parent / "fence" / "compute_specialist.md"


def _fetch_catalog(gfx_id: str) -> List[Dict[str, Any]]:
    """Delegate to tools.compute_techniques.catalog. Module-level for test injection."""
    try:
        from perfxpert.tools import compute_techniques  # type: ignore
        return compute_techniques.catalog(gfx_id=gfx_id)
    except ImportError:
        return []  # Phase 4 deliverable


def build_compute_specialist() -> Agent:
    tools = [
        ToolBinding(name="compute_techniques.catalog", fn=_fetch_catalog),
        ToolBinding(name="arch.lookup_peaks", fn=arch.lookup_peaks),
        ToolBinding(name="roofline.classify", fn=roofline.classify),
        ToolBinding(name="compiler.lookup_flags", fn=compiler.lookup_flags),
    ]
    return Agent(
        name="ComputeTechniquesSpecialist",
        layer=2,
        fence_path=str(_FENCE_PATH) if _FENCE_PATH.exists() else None,
        input_schema=schemas.ComputeSpecialistInput,
        output_schema=schemas.ComputeSpecialistOutput,
        tools=tools,
        allowed_handoffs=[],    # Layer-2 returns to Recommendation
        token_budget=3072,
    )


def _rank_catalog_deterministic(catalog: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Air-gap sorting: score = expected_impact × (1/effort_factor); tie-break by risk."""
    risk_order = {"low": 0, "medium": 1, "high": 2}

    def score(entry: Dict[str, Any]) -> tuple:
        impact = entry.get("expected_impact", 0.0)
        effort = entry.get("effort_factor", 1.0) or 1.0
        risk = risk_order.get(entry.get("risk", "medium"), 1)
        return (-(impact / effort), risk)

    return sorted(catalog, key=score)


def run_compute_specialist(
    payload: schemas.ComputeSpecialistInput,
    *,
    provider: str = "anthropic",
    airgap: Optional[bool] = None,
) -> schemas.ComputeSpecialistOutput:
    catalog = _fetch_catalog(payload.gfx_id)

    agent = build_compute_specialist()
    raw = run_agent(
        agent,
        input_payload={**payload.model_dump(), "catalog": catalog},
        provider=provider,
        airgap=airgap,
    )

    if raw.get("_mode") == "airgap":
        ranked = _rank_catalog_deterministic(catalog)
        return schemas.ComputeSpecialistOutput(
            techniques=ranked,
            confidence=0.6,
            citations=[],
        )

    so = raw.get("structured_output") or {}
    return schemas.ComputeSpecialistOutput(
        techniques=so.get("techniques", _rank_catalog_deterministic(catalog)),
        confidence=so.get("confidence", 0.6),
        citations=so.get("citations", []),
    )


__all__ = ["build_compute_specialist", "run_compute_specialist"]
