"""interconnect — XGMI / PCIe / NIC peak lookup tool.

Structured lookup for RCCL communication analysis. Returns per-arch XGMI
aggregate + per-link bandwidths, the PCIe tier, and an "achievable" RCCL
AllReduce busBW number (typical field observation) so the Latency
Specialist can classify bus-bandwidth efficiency.

Tool class: READ_ONLY (MCP-safe).

knowledge/interconnect_specs.yaml is the source of truth. Inter-node NIC
entries (e.g. "connectx7") live alongside the GPU keys and are selected
explicitly by the caller — there is no auto-detection.
"""

from typing import Any, Dict

from perfxpert.knowledge import load_yaml
from perfxpert.tools._class import ToolClass, tool_class


@tool_class(ToolClass.READ_ONLY)
def lookup_peaks(gfx_id: str) -> Dict[str, Any]:
    """Return interconnect peak specs for a given GPU architecture or NIC.

    Args:
        gfx_id: Architecture identifier (e.g. ``"gfx942"`` for MI300X) or
            NIC codename (e.g. ``"connectx7"``).

    Returns:
        Dict with keys: name, xgmi_peak_gbps, xgmi_links,
        xgmi_per_link_gbps, pcie_tier, pcie_peak_gbps, achievable_gbps,
        source_url, measured_with.

    Raises:
        KeyError: when ``gfx_id`` is not in the spec table. Message includes
            the list of known keys for easy discovery.

    Example:
        >>> from perfxpert.tools.interconnect import lookup_peaks
        >>> mi300x = lookup_peaks("gfx942")
        >>> mi300x["xgmi_peak_gbps"]
        448.0
    """
    specs = load_yaml("interconnect_specs")
    if gfx_id not in specs:
        known = ", ".join(sorted(specs.keys()))
        raise KeyError(
            f"Unknown interconnect id {gfx_id!r}; known: {known}"
        )
    return specs[gfx_id]
