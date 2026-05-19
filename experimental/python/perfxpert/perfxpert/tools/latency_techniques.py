"""Latency technique catalog shim backed by proven_optimizations.yaml."""

from __future__ import annotations

from typing import Any, Dict, List

from perfxpert.tools._class import ToolClass, tool_class
from perfxpert.tools._technique_catalog import catalog_for


_LATENCY_NAMES = [
    "hip_stream_overlap",
    "kernel_fusion_small_launches",
    "device_sync_removal",
]


@tool_class(ToolClass.READ_ONLY)
def catalog(gfx_id: str) -> List[Dict[str, Any]]:
    return catalog_for(_LATENCY_NAMES, gfx_id, category="latency")


__all__ = ["catalog"]
