"""Latency-Techniques Specialist (Layer 2).

Tool allowlist (5 of 5 used after Phase 10 D):
  latency_techniques.catalog, arch.lookup_peaks,
  rccl_analysis.analyze_collectives, interconnect.lookup_peaks,
  dependency_graph.reconstruct_dag

Note: Phase 10 B introduced `gpu_runtime_monitor` (amd-smi / rocm-smi JSON
ingest + thermal analysis). It is reachable via the MCP surface and
consumed opportunistically by the specialist via the fence narrative —
but we do not spend a cap slot on it because thermal envelope analysis is
diagnostic (out-of-band), not per-invocation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from perfxpert.agents import schemas
from perfxpert.agents._predict_attach import attach_predictions_to_techniques
from perfxpert.agents.compute_specialist import _rank_catalog_deterministic
from perfxpert.agents.framework import Agent, ToolBinding, run_agent
from perfxpert.tools import arch, dependency_graph, interconnect, rccl_analysis


_FENCE_PATH = Path(__file__).parent / "fence" / "latency_specialist.md"


def _fetch_catalog(gfx_id: str) -> List[Dict[str, Any]]:
    try:
        from perfxpert.tools import latency_techniques  # type: ignore
        return latency_techniques.catalog(gfx_id=gfx_id)
    except ImportError:
        return []  # defensive fallback if latency_techniques tool is absent


def build_latency_specialist() -> Agent:
    tools = [
        ToolBinding(name="latency_techniques.catalog", fn=_fetch_catalog),
        ToolBinding(name="arch.lookup_peaks", fn=arch.lookup_peaks),
        # Phase 10 — RCCL / NIC communication analysis. The specialist can now
        # classify bus-bandwidth efficiency vs peak and comm/compute overlap
        # for latency-bound kernels that fire inside RCCL collectives.
        ToolBinding(
            name="rccl_analysis.analyze_collectives",
            fn=rccl_analysis.analyze_collectives,
        ),
        ToolBinding(
            name="interconnect.lookup_peaks",
            fn=interconnect.lookup_peaks,
        ),
        # Phase 10 D — dependency-graph DAG reconstruction + bubble
        # detection. Core to latency advice: tells the specialist which
        # idle gaps are unnecessary sync vs inherent dependency.
        ToolBinding(
            name="dependency_graph.reconstruct_dag",
            fn=dependency_graph.reconstruct_dag,
        ),
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
        techniques = attach_predictions_to_techniques(
            _rank_catalog_deterministic(catalog), payload
        )
        return schemas.LatencySpecialistOutput(
            techniques=techniques,
            confidence=0.6,
            citations=[],
        )

    so = raw.get("structured_output") or {}
    raw_techniques = so.get("techniques", _rank_catalog_deterministic(catalog))
    techniques = attach_predictions_to_techniques(raw_techniques, payload)
    return schemas.LatencySpecialistOutput(
        techniques=techniques,
        confidence=so.get("confidence", 0.6),
        citations=so.get("citations", []),
    )


__all__ = ["build_latency_specialist", "run_latency_specialist"]
