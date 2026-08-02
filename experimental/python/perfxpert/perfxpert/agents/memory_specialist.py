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
from perfxpert.agents import creativity
from perfxpert.agents.creativity import CreativityTier
from perfxpert.agents.framework import (
    AgentCapability,
    Agent,
    ToolBinding,
    airgap_enabled,
    run_agent,
)
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
        capability=AgentCapability.ADDITIVE_EXPLORATION,
    )


def run_memory_specialist(
    payload: schemas.MemorySpecialistInput,
    *,
    provider: str = "anthropic",
    airgap: Optional[bool] = None,
) -> schemas.MemorySpecialistOutput:
    catalog = _fetch_catalog(payload.gfx_id)
    agent = build_memory_specialist()

    # The vetted lane is deterministic in both modes. It used to be
    # `so.get("techniques", ...)`, which let a live model replace the whole
    # catalog ranking with invented entries that then carried the same
    # authority as a proven technique. A model now contributes through the
    # exploratory lane instead, where its output is labelled as unproven.
    techniques = attach_predictions_to_techniques(
        _rank_memory_catalog(catalog), payload
    )

    # RFC 0001 freezes the deterministic core before any model call, and
    # returns here without one under the default strict tier: nothing in this
    # schema is model-supplied, so the call was made and its result discarded
    # in full -- paying tokens, latency, and a failure surface for a run the
    # model cannot contribute to.
    airgapped = airgap_enabled(airgap)
    if creativity.effective_tier(agent, airgap=airgapped) is not CreativityTier.EXPLORATORY:
        return schemas.MemorySpecialistOutput(
            techniques=techniques,
            confidence=0.6,
            citations=[],
        )

    raw = run_agent(
        agent,
        input_payload={**payload.model_dump(), "catalog": catalog},
        provider=provider,
        airgap=airgap,
    )

    return schemas.MemorySpecialistOutput(
        techniques=techniques,
        confidence=0.6,
        citations=[],
        exploratory_proposals=creativity.proposals_from_response(
            agent,
            raw,
            specialist="memory",
            airgap=airgapped,
            manifest=creativity.manifest_from_run(
                agent,
                raw,
                kernels=[k.get("name", "") for k in payload.hot_kernels],
                catalog_entries=[t.get("name", "") for t in techniques],
            ),
            provider=provider,
        ),
    )


__all__ = ["build_memory_specialist", "run_memory_specialist"]
