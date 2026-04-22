"""Trace-analysis helpers used by the agentic analysis path.

These wrappers expose the existing deterministic analysis core through the
`perfxpert.tools` namespace so the Analysis agent can bind them as READ_ONLY
tools while still reusing the legacy implementation underneath.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from perfxpert.analysis import compute_time_breakdown, identify_hotspots
from perfxpert.connection import PerfxpertConnection
from perfxpert.tools._class import ToolClass, tool_class


def _fractional_percent(value: float) -> float:
    return max(float(value or 0.0), 0.0) / 100.0


@tool_class(ToolClass.READ_ONLY)
def time_breakdown(database_path: str) -> Dict[str, float]:
    """Return normalized trace fractions for the agentic analysis path."""
    with PerfxpertConnection(str(Path(database_path))) as conn:
        breakdown = compute_time_breakdown(conn)

    return {
        "kernel_pct": _fractional_percent(breakdown.get("kernel_percent", 0.0)),
        "memcpy_pct": _fractional_percent(breakdown.get("memcpy_percent", 0.0)),
        "api_pct": _fractional_percent(breakdown.get("overhead_percent", 0.0)),
        "idle_pct": 0.0,
    }


@tool_class(ToolClass.READ_ONLY)
def hotspots(database_path: str, top_n: int = 10) -> List[Dict[str, Any]]:
    """Return hotspot metadata in the agentic-schema shape."""
    with PerfxpertConnection(str(Path(database_path))) as conn:
        rows = identify_hotspots(conn, top_n=top_n)

    return [
        {
            "name": row.get("name"),
            "pct": _fractional_percent(row.get("percent_of_total", 0.0)),
            "duration_ns": int(row.get("total_duration", 0) or 0),
            "calls": int(row.get("calls", 0) or 0),
            "avg_duration_ns": int(row.get("avg_duration", 0) or 0),
        }
        for row in rows
    ]


__all__ = ["time_breakdown", "hotspots"]
