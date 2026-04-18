"""Latency-Techniques Specialist (Layer 2).

Tool allowlist (2 of 5 used):
  latency_techniques.catalog, arch.lookup_peaks
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from perfxpert.agents import schemas
from perfxpert.agents.compute_specialist import _rank_catalog_deterministic
from perfxpert.agents.framework import Agent, ToolBinding, run_agent
from perfxpert.tools import arch


_FENCE_PATH = Path(__file__).parent / "fence" / "latency_specialist.md"


def _fetch_catalog(gfx_id: str) -> List[Dict[str, Any]]:
    try:
        from perfxpert.tools import latency_techniques  # type: ignore
        return latency_techniques.catalog(gfx_id=gfx_id)
    except ImportError:
        return []  # Phase 4 deliverable


def build_latency_specialist() -> Agent:
    tools = [
        ToolBinding(name="latency_techniques.catalog", fn=_fetch_catalog),
        ToolBinding(name="arch.lookup_peaks", fn=arch.lookup_peaks),
    ]
    return Agent(
        name="LatencyTechniquesSpecialist",
        layer=2,
        fence_path=str(_FENCE_PATH) if _FENCE_PATH.exists() else None,
        input_schema=schemas.LatencySpecialistInput,
        output_schema=schemas.LatencySpecialistOutput,
        tools=tools,
        allowed_handoffs=[],
        token_budget=3072,
    )


def run_latency_specialist(
    payload: schemas.LatencySpecialistInput,
    *,
    provider: str = "anthropic",
    airgap: Optional[bool] = None,
) -> schemas.LatencySpecialistOutput:
    catalog = _fetch_catalog(payload.gfx_id)
    agent = build_latency_specialist()
    raw = run_agent(
        agent,
        input_payload={**payload.model_dump(), "catalog": catalog},
        provider=provider,
        airgap=airgap,
    )

    if raw.get("_mode") == "airgap":
        return schemas.LatencySpecialistOutput(
            techniques=_rank_catalog_deterministic(catalog),
            confidence=0.6,
            citations=[],
        )

    so = raw.get("structured_output") or {}
    return schemas.LatencySpecialistOutput(
        techniques=so.get("techniques", _rank_catalog_deterministic(catalog)),
        confidence=so.get("confidence", 0.6),
        citations=so.get("citations", []),
    )


__all__ = ["build_latency_specialist", "run_latency_specialist"]
