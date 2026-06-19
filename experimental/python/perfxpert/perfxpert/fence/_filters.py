"""Role → YAML filter map: decides which knowledge keys each role sees.

Narrow-scope principle: each role gets only the YAML slices it needs.
Every role receives {gpu_specs, metric_thresholds, bottleneck_types} as
a core floor; role-specific additions are layered on top.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import yaml

from perfxpert.knowledge import load_yaml


_CORE = {
    "gpu_specs": ["all"],
    "metric_thresholds": ["all"],
    "bottleneck_types": ["all"],
}


ROLE_YAML_MAP: Dict[str, Dict[str, List[str]]] = {
    "root": {
        **_CORE,
    },
    "analysis": {
        **_CORE,
        "derived_metrics": ["all"],
        "sol_metrics": ["all"],
    },
    "recommendation": {
        **_CORE,
        "optimization_techniques": ["all"],
        "compiler_flags": ["all"],
        "amdahl_thresholds": ["all"],
    },
    "correctness": {
        **_CORE,
        "amdahl_thresholds": ["all"],
    },
    "compute_specialist": {
        **_CORE,
        "vgpr_occupancy_tables": ["all"],
        "derived_metrics": ["all"],
        "optimization_techniques": ["compute"],
    },
    "memory_specialist": {
        **_CORE,
        "memory_hierarchy": ["all"],
        "derived_metrics": ["all"],
        "optimization_techniques": ["memory"],
    },
    "latency_specialist": {
        **_CORE,
        "top_down_analysis": ["all"],
        "pc_sampling_stall_reasons": ["all"],
        "optimization_techniques": ["latency"],
    },
}


def get_yaml_keys_for_role(role: str) -> Dict[str, List[str]]:
    """Return the yaml→keys map for a role, or {} if unknown."""
    return dict(ROLE_YAML_MAP.get(role, {}))


def _render_dict(data, gfx_id: Optional[str], bottleneck: Optional[str]) -> str:
    if isinstance(data, dict) and gfx_id and gfx_id in data:
        filtered = {gfx_id: data[gfx_id]}
        return yaml.safe_dump(filtered, sort_keys=True)
    if isinstance(data, list) and bottleneck:
        filtered = [entry for entry in data if entry.get("name") == bottleneck] or data
        return yaml.safe_dump(filtered, sort_keys=False)
    return yaml.safe_dump(data, sort_keys=False)


def format_yaml_excerpt(
    yaml_name: str,
    keys: List[str],
    *,
    gfx_id: Optional[str] = None,
    bottleneck: Optional[str] = None,
) -> str:
    """Render a YAML excerpt as a fenced markdown block. Returns '' if unavailable."""
    if not keys:
        return ""
    try:
        data = load_yaml(yaml_name)
    except FileNotFoundError:
        return ""

    body = _render_dict(data, gfx_id, bottleneck)
    return f"## Knowledge: {yaml_name}\n\n```yaml\n{body}```"


__all__ = ["ROLE_YAML_MAP", "get_yaml_keys_for_role", "format_yaml_excerpt"]
