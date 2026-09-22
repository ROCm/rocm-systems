"""occupancy — VGPR → waves/EU lookup + reduction suggestions.

Reads knowledge/vgpr_occupancy_tables.yaml.

Tool class: READ_ONLY.
"""

from typing import Any, Dict

from perfxpert.knowledge import load_yaml
from perfxpert.tools._class import ToolClass, tool_class


@tool_class(ToolClass.READ_ONLY)
def lookup_waves_per_eu(vgpr_count: int, gfx_id: str) -> int:
    """Look up max waves/EU for a given VGPR count + architecture."""
    tables = load_yaml("vgpr_occupancy_tables")
    if gfx_id not in tables:
        raise KeyError(f"Unknown gfx_id {gfx_id!r}; known: {sorted(tables.keys())}")

    # Find the first threshold entry where vgpr_count <= max_vgprs
    for entry in tables[gfx_id]:
        if vgpr_count <= entry["max_vgprs"]:
            return entry["waves_per_eu"]
    return 1   # fallback — too many VGPRs for any category


@tool_class(ToolClass.READ_ONLY)
def suggest_vgpr_reduction(
    current_vgpr: int, occupancy_pct: float, kernel_time_pct: float, gfx_id: str = "gfx942"
) -> Dict[str, Any]:
    """Suggest VGPR reduction targets + expected impact.

    Only fires when current VGPR is high AND kernel is hot enough to matter.
    """
    if kernel_time_pct < 0.05:
        return {"applicable": False, "reason": "kernel too small (Amdahl)"}

    if current_vgpr <= 32:
        return {"applicable": False, "reason": "already at optimal VGPR count"}

    current_waves = lookup_waves_per_eu(current_vgpr, gfx_id)
    target_vgpr = 32 if current_vgpr > 64 else 40
    target_waves = lookup_waves_per_eu(target_vgpr, gfx_id)

    if target_waves <= current_waves:
        return {"applicable": False, "reason": "no occupancy gain"}

    occupancy_ratio = target_waves / current_waves
    return {
        "applicable": True,
        "current_vgpr": current_vgpr,
        "target_vgpr": target_vgpr,
        "current_waves": current_waves,
        "target_waves": target_waves,
        "expected_occupancy_multiplier": occupancy_ratio,
        "suggested_flag": f"__launch_bounds__(256, {target_waves})",
    }
