"""Memory-Techniques Specialist (Layer 2).

Tool allowlist (3 of 5 used):
  memory_techniques.catalog, arch.lookup_peaks, bottleneck.lookup_signatures
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from perfxpert.agents import schemas
from perfxpert.agents.compute_specialist import _rank_catalog_deterministic
from perfxpert.agents.framework import Agent, ToolBinding, run_agent
from perfxpert.tools import arch, bottleneck


_FENCE_PATH = Path(__file__).parent / "fence" / "memory_specialist.md"


def _fetch_catalog(gfx_id: str) -> List[Dict[str, Any]]:
    try:
        from perfxpert.tools import memory_techniques  # type: ignore
        return memory_techniques.catalog(gfx_id=gfx_id)
    except ImportError:
        return []  # Phase 4 deliverable


def build_memory_specialist() -> Agent:
    tools = [
        ToolBinding(name="memory_techniques.catalog", fn=_fetch_catalog),
        ToolBinding(name="arch.lookup_peaks", fn=arch.lookup_peaks),
        ToolBinding(name="bottleneck.lookup_signatures", fn=bottleneck.lookup_signatures),
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
        return schemas.MemorySpecialistOutput(
            techniques=_rank_catalog_deterministic(catalog),
            confidence=0.6,
            citations=[],
        )

    so = raw.get("structured_output") or {}
    return schemas.MemorySpecialistOutput(
        techniques=so.get("techniques", _rank_catalog_deterministic(catalog)),
        confidence=so.get("confidence", 0.6),
        citations=so.get("citations", []),
    )


__all__ = ["build_memory_specialist", "run_memory_specialist"]
