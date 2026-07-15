"""Memory technique catalog shim backed by proven_optimizations.yaml."""

from __future__ import annotations

from typing import Any, Dict, List

from perfxpert.tools._class import ToolClass, tool_class
from perfxpert.tools._technique_catalog import catalog_for


_MEMORY_NAMES = [
    "memory_coalescing_stride_fix",
    "lds_tiling_matmul",
    "hip_stream_overlap",
    "warp_primitives_reduction",
    "cache_blocking_kernel",
]


@tool_class(ToolClass.READ_ONLY)
def catalog(gfx_id: str) -> List[Dict[str, Any]]:
    return catalog_for(_MEMORY_NAMES, gfx_id, category="memory")


__all__ = ["catalog"]
