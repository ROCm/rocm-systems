"""counters — HW counter lookup + per-block limit validation.

Replaces LLM recitation of counter definitions and per-block limits with
structured lookup. Reads knowledge/counter_catalog.yaml + pmc_limits.yaml.

Tool class: READ_ONLY.
"""

from collections import defaultdict
from typing import Any, Dict, List

from perfxpert.knowledge import load_yaml
from perfxpert.tools._class import ToolClass, tool_class

# TCC-derived metrics must be isolated to their own rocprofv3 pass
# Ref: CLAUDE.md — FETCH_SIZE/WRITE_SIZE each need own pass
_TCC_DERIVED = frozenset({"FETCH_SIZE", "WRITE_SIZE"})


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
        {"ok": bool, "violations": [...], "fixed_passes": [[counter,...], ...]}
    """
    limits_cfg = load_yaml("pmc_limits")["per_block_limits"]
    isolation_rules = load_yaml("rocprofv3_counter_limits")["isolation_rules"]
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

    violations: List[Dict[str, Any]] = []
    unknown = sorted(
        {c for c in counter_list if c not in name_to_block and c not in _TCC_DERIVED}
    )
    if unknown:
        violations.append(
            {
                "severity": "error",
                "rule": "All requested counters must exist in the counter catalog",
                "reason": f"Unknown counters: {', '.join(unknown)}",
            }
        )

    known = [c for c in counter_list if c not in unknown]
    derived = [c for c in known if c in _TCC_DERIVED]
    regular = [c for c in known if c not in _TCC_DERIVED]

    rule_map = {rule["rule"]: rule for rule in isolation_rules}
    regular_sq = [c for c in regular if name_to_block.get(c) == "SQ"]

    if "FETCH_SIZE" in derived and len(known) > 1:
        rule = rule_map["FETCH_SIZE requires its own dedicated pass"]
        violations.append({**rule, "auto_fixed": True})
    if "WRITE_SIZE" in derived and len(known) > 1:
        rule = rule_map["WRITE_SIZE requires its own dedicated pass"]
        violations.append({**rule, "auto_fixed": True})
    if {"FETCH_SIZE", "WRITE_SIZE"}.issubset(set(derived)):
        rule = rule_map["FETCH_SIZE and WRITE_SIZE MUST NOT share a pass"]
        violations.append({**rule, "auto_fixed": True})
    if derived and regular_sq:
        rule = rule_map["FETCH_SIZE/WRITE_SIZE MUST NOT share a pass with any SQ counter"]
        violations.append({**rule, "auto_fixed": True})

    by_block: Dict[str, List[str]] = defaultdict(list)
    for c in regular:
        by_block[name_to_block[c]].append(c)

    passes: List[List[str]] = []
    if by_block:
        n_passes = max(
            (len(names) + _limit_for(block) - 1) // _limit_for(block)
            for block, names in by_block.items()
        )
        pass_counters: List[List[str]] = [[] for _ in range(n_passes)]
        for block, names in by_block.items():
            limit = _limit_for(block)
            for pass_idx in range(n_passes):
                chunk = names[pass_idx * limit : (pass_idx + 1) * limit]
                pass_counters[pass_idx].extend(chunk)
        passes.extend([p for p in pass_counters if p])

    for d in derived:
        passes.append([d])

    ok = not any(
        v["severity"] == "error" and not v.get("auto_fixed", False) for v in violations
    )
    return {"ok": ok, "violations": violations, "fixed_passes": passes}
