"""Memory-Techniques Specialist (Layer 2).

Tool allowlist (5 of 5 used after Phase 10 C):
  memory_techniques.catalog, arch.lookup_peaks, bottleneck.lookup_signatures,
  predict_impact.predict_change_impact, unified_memory.analyze_paging
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from perfxpert.agents import schemas
from perfxpert.agents._predict_attach import attach_predictions_to_techniques
from perfxpert.agents.compute_specialist import (
    _promote_named_technique,
    _rank_catalog_deterministic,
)
from perfxpert.agents.framework import Agent, ToolBinding, run_agent
from perfxpert.tools import arch, bottleneck, predict_impact, unified_memory


_FENCE_PATH = Path(__file__).parent / "fence" / "memory_specialist.md"


def _fetch_catalog(gfx_id: str) -> List[Dict[str, Any]]:
    try:
        from perfxpert.tools import memory_techniques  # type: ignore
        return memory_techniques.catalog(gfx_id=gfx_id)
    except ImportError:
        return []  # defensive fallback if memory_techniques tool is absent


def _rank_memory_catalog(catalog: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked = _rank_catalog_deterministic(catalog)
    # Recommendation routes to this specialist only for the memory_transfer
    # bottleneck, so overlap recommendations should lead the deterministic path.
    return _promote_named_technique(ranked, "hip_stream_overlap")


def build_memory_specialist() -> Agent:
    tools = [
        ToolBinding(name="memory_techniques.catalog", fn=_fetch_catalog),
        ToolBinding(name="arch.lookup_peaks", fn=arch.lookup_peaks),
        ToolBinding(name="bottleneck.lookup_signatures", fn=bottleneck.lookup_signatures),
        # Phase 10 — change-impact prediction. The specialist calls this
        # once per surfaced technique before returning so each rec card
        # carries a speedup bracket + confidence.
        ToolBinding(
            name="predict_impact.predict_change_impact",
            fn=predict_impact.predict_change_impact,
        ),
        # Phase 10 C — MI300X unified-memory / cross-die penalty scan.
        # Fills the 5th slot; called once per invocation to derive
        # paging hot-spots + XCD fabric traffic totals for the
        # memory_patterns.yaml signatures.
        ToolBinding(
            name="unified_memory.analyze_paging",
            fn=unified_memory.analyze_paging,
        ),
    ]
    return Agent(
        name="MemoryTechniquesSpecialist",
        layer=2,
        fence_path=str(_FENCE_PATH) if _FENCE_PATH.exists() else None,
        input_schema=schemas.MemorySpecialistInput,
        output_schema=schemas.MemorySpecialistOutput,
        tools=tools,
        allowed_handoffs=[],
        token_budget=3072,
    )


def run_memory_specialist(
    payload: schemas.MemorySpecialistInput,
    *,
    provider: str = "anthropic",
    airgap: Optional[bool] = None,
) -> schemas.MemorySpecialistOutput:
    catalog = _fetch_catalog(payload.gfx_id)
    agent = build_memory_specialist()
    raw = run_agent(
        agent,
        input_payload={**payload.model_dump(), "catalog": catalog},
        provider=provider,
        airgap=airgap,
    )

    if raw.get("_mode") == "airgap":
        techniques = attach_predictions_to_techniques(
            _rank_memory_catalog(catalog), payload
        )
        return schemas.MemorySpecialistOutput(
            techniques=techniques,
            confidence=0.6,
            citations=[],
        )

    so = raw.get("structured_output") or {}
    raw_techniques = so.get("techniques", _rank_memory_catalog(catalog))
    techniques = attach_predictions_to_techniques(raw_techniques, payload)
    return schemas.MemorySpecialistOutput(
        techniques=techniques,
        confidence=so.get("confidence", 0.6),
        citations=so.get("citations", []),
    )


__all__ = ["build_memory_specialist", "run_memory_specialist"]
