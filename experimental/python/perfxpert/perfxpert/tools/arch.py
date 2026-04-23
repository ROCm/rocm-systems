"""arch — GPU architecture lookup tool.

Replaces free-text GPU spec recitation in the fence with structured lookup.
Agents call lookup_peaks(gfx_id) instead of embedding specs in prompts.

Tool class: READ_ONLY (MCP-safe).

See design spec Appendix A; knowledge/gpu_specs.yaml is the source of truth.
"""

from typing import Any, Dict

from perfxpert.knowledge import load_yaml
from perfxpert.tools._class import ToolClass, tool_class


@tool_class(ToolClass.READ_ONLY)
def lookup_peaks(gfx_id: str) -> Dict[str, Any]:
    """Return hardware peak specs for a given gfx architecture.

    Args:
        gfx_id: Architecture identifier, e.g., "gfx942" for MI300X.

    Returns:
        Dict with keys: name, codename, peak_fp64_tflops, peak_fp32_tflops,
        optional matrix-peak fields (where applicable), peak_bf16_tflops,
        memory_bandwidth_tbs, cu_count, lds_kb, wave_size,
        max_vgprs_per_thread, ridge_point, and optional sku_variants.

    Raises:
        KeyError: if gfx_id is not recognized. Error includes list of known archs.

    Example:
        >>> from perfxpert.tools.arch import lookup_peaks
        >>> mi300x = lookup_peaks("gfx942")
        >>> mi300x["peak_fp64_tflops"]
        81.7
    """
    specs = load_yaml("gpu_specs")
    if gfx_id not in specs:
        known = ", ".join(sorted(specs.keys()))
        raise KeyError(f"Unknown gfx_id {gfx_id!r}; known archs: {known}")
    return specs[gfx_id]
