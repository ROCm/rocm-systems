"""counters — HW counter lookup + per-block limit validation.

Replaces LLM recitation of counter definitions and per-block limits with
structured lookup. Reads knowledge/counter_catalog.yaml + pmc_limits.yaml.

Tool class: READ_ONLY.
"""

from typing import Any, Dict, List

from perfxpert.knowledge import load_yaml
from perfxpert.tools._class import ToolClass, tool_class

# TCC-derived metrics must be isolated to their own rocprofv3 pass
# Ref: CLAUDE.md — FETCH_SIZE/WRITE_SIZE each need own pass
_TCC_DERIVED = frozenset({"FETCH_SIZE", "WRITE_SIZE"})


def _build_multi_pass_escalation(
    passes: List[List[str]], gpu_arch: str
) -> Dict[str, Any]:
    """Return concrete guidance for multi-pass counter collection."""
    pass_specs = [
        {
            "index": idx,
            "counters": list(counter_pass),
            "pmc": " ".join(counter_pass),
        }
        for idx, counter_pass in enumerate(passes, start=1)
    ]
    pmc_groups = [f"pmc: {spec['pmc']}" for spec in pass_specs]
    return {
        "required": True,
        "reason": (
            f"Requested counters require {len(pass_specs)} profiling passes on "
            f"{gpu_arch}; a single rocprofv3 --pmc invocation would exceed "
            "per-block or isolation limits."
        ),
        "gpu_arch": gpu_arch,
        "pass_count": len(pass_specs),
        "passes": pass_specs,
        "pmc_groups_path": "pmc_groups.txt",
        "pmc_groups": pmc_groups,
        "commands": [
            {
                "tool": "rocprof-compute",
                "description": (
                    "Use rocprof-compute when you want the profiler to handle "
                    "the required multi-pass hardware collection for you."
                ),
                "full_command": "rocprof-compute profile -- ./app",
            },
            {
                "tool": "rocprofv3",
                "description": (
                    "Write one pmc group per line, then collect the requested "
                    "passes with rocprofv3 -i."
                ),
                "full_command": "rocprofv3 -i pmc_groups.txt -- ./app",
            },
        ],
    }


@tool_class(ToolClass.READ_ONLY)
def lookup_info(name: str, gfx_id: str = None) -> Dict[str, Any]:
    """Return structured info for an HW counter by name.

    Args:
        name: Counter name (e.g., "SQ_WAVES", "GRBM_COUNT").
        gfx_id: Optional architecture qualifier (future use).

    Returns:
        {"name", "block", "unit", "description"}

    Raises:
        KeyError: if counter not in catalog.
    """
    catalog = load_yaml("counter_catalog")
    # Catalog is a flat list — each entry has name + block + unit + description
    for entry in catalog:
        if entry["name"] == name:
            return {
                "name": name,
                "block": entry["block"],
                "unit": entry.get("unit", "count"),
                "description": entry.get("description", ""),
            }
    known = [e["name"] for e in catalog]
    raise KeyError(f"Unknown counter {name!r}; {len(known)} known counters")


@tool_class(ToolClass.READ_ONLY)
def validate_for_gpu(counter_list: List[str], gpu_arch: str) -> Dict[str, Any]:
    """Validate counter list against per-block hardware limits.

    Returns a validated grouping of counters into passes — each pass respects
    per-block limits AND the FETCH_SIZE/WRITE_SIZE isolation rule.

    Args:
        counter_list: List of counter names.
        gpu_arch: Architecture (e.g., "gfx942").

    Returns:
        {
          "ok": bool,
          "violations": [...],
          "fixed_passes": [[counter,...], ...],
          "escalation": {...} | None,
        }
    """
    limits_cfg = load_yaml("pmc_limits")["per_block_limits"]
    catalog = load_yaml("counter_catalog")

    # Build name → block index from the flat catalog
    name_to_block: Dict[str, str] = {entry["name"]: entry["block"] for entry in catalog}
    # TCC-derived counters are implicitly TCC block
    for derived in _TCC_DERIVED:
        name_to_block.setdefault(derived, "TCC")

    def _limit_for(block: str) -> int:
        """Look up per-pass limit, preferring arch-specific override."""
        info = limits_cfg.get(block, {})
        arch_key = f"{gpu_arch}_limit"
        return int(info.get(arch_key, info.get("limit", 4)))

    # Separate TCC-derived counters — each gets its own pass (anti-Sakana)
    derived = [c for c in counter_list if c in _TCC_DERIVED]
    regular = [c for c in counter_list if c not in _TCC_DERIVED]

    # Group regular by block, respecting per-block limit
    passes: List[List[str]] = []
    by_block: Dict[str, List[str]] = {}
    for c in regular:
        block = name_to_block.get(c, "UNKNOWN")
        by_block.setdefault(block, []).append(c)

    # Pack one pass at a time, up to per-block limit
    active: List[str] = []
    active_counts: Dict[str, int] = {}
    for block, names in by_block.items():
        lim = _limit_for(block)
        for c in names:
            if active_counts.get(block, 0) >= lim:
                passes.append(active)
                active, active_counts = [], {}
            active.append(c)
            active_counts[block] = active_counts.get(block, 0) + 1
    if active:
        passes.append(active)

    # Each TCC-derived counter gets its own pass (spec §5.8 anti-Sakana rule)
    for d in derived:
        passes.append([d])

    escalation = _build_multi_pass_escalation(passes, gpu_arch) if len(passes) > 1 else None
    return {
        "ok": True,
        "violations": [],
        "fixed_passes": passes,
        "escalation": escalation,
    }
