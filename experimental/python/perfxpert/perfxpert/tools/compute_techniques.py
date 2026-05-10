"""Compute technique catalog shim backed by proven_optimizations.yaml."""

from __future__ import annotations

from typing import Any, Dict, List

from perfxpert.tools._class import ToolClass, tool_class
from perfxpert.tools._technique_catalog import catalog_for


_COMPUTE_NAMES = [
    "vgpr_reduction_compute_bound",
    "mfma_enablement",
    "fast_math_compiler_flag",
]


@tool_class(ToolClass.READ_ONLY)
def catalog(gfx_id: str) -> List[Dict[str, Any]]:
    return catalog_for(_COMPUTE_NAMES, gfx_id, category="compute")


__all__ = ["catalog"]
