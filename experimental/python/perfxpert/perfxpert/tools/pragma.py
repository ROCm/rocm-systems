"""pragma — LLVM loop-hint advice (Phase 10 advanced specialists).

Reads ``knowledge/compiler_pragmas.yaml``. Offers three READ_ONLY MCP
tools:

* :func:`lookup_pragmas`              — list the catalog.
* :func:`explain_pragma`              — describe one pragma.
* :func:`suggest_pragmas_for_kernel`  — apply the decision rules.

All three tools are auto-discovered by
``mcp_server._registry.discover_read_only_tools``. The fence slice for
the Compute Specialist (``perfxpert/agents/fence/compute_specialist.md``)
binds ``pragma.suggest_pragmas_for_kernel`` into its 5-tool allowlist.

Design invariants (enforced here; fence re-states them for the LLM):

* **Amdahl gate.** Kernels with ``hotspot_pct < 5`` never receive a
  pragma rec — the deterministic Amdahl threshold lives in
  ``knowledge/amdahl_thresholds.yaml`` and is mirrored here as a
  module-level constant so the tool stays pure-Python.
* **Tier-0 source anchor.** The caller must supply a ``source_path``
  inside ``signals`` when the kernel has one. If the path matches the
  Triton JIT cache (``".triton/"``) the tool skips the kernel — Triton
  regenerates kernels per run and the hint would vanish on the next
  launch.
* **No free-form factor invention.** ``suggest_pragmas_for_kernel``
  returns ``factor_sweep`` verbatim from YAML; the Compute Specialist
  is never allowed to invent other values.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List

from perfxpert.knowledge import load_yaml
from perfxpert.tools._class import ToolClass, tool_class


# Amdahl gate: kernel must be at least this fraction of total runtime
# before we even consider emitting a pragma rec.
_MIN_HOTSPOT_PCT = 5.0

# Triton JIT cache directory fragment — Triton regenerates the
# containing .cu/.hip file on every launch so any hint we emit would
# evaporate. See knowledge/compiler_pragmas.yaml header.
_TRITON_PATH_FRAGMENT = ".triton/"


def _by_name(entries: List[Dict[str, Any]], name: str) -> Dict[str, Any]:
    for e in entries:
        if e.get("pragma") == name:
            return e
    raise KeyError(f"Unknown pragma {name!r}")


@tool_class(ToolClass.READ_ONLY)
def lookup_pragmas(gpu_only: bool = True) -> List[Dict[str, Any]]:
    """Return the pragma catalog.

    Args:
        gpu_only: When True (default), return only entries with
            ``gpu_applicable=True`` AND ``allowlist=True`` — the subset
            the Compute Specialist is ever allowed to suggest. When
            False, return the full catalog (including rejected entries)
            so the fence can *see* why those are off-limits.
    """
    catalog = load_yaml("compiler_pragmas") or []
    if gpu_only:
        return [
            e for e in catalog
            if e.get("gpu_applicable", False) and e.get("allowlist", False)
        ]
    return list(catalog)


@tool_class(ToolClass.READ_ONLY)
def explain_pragma(pragma_name: str) -> Dict[str, Any]:
    """Return the full catalog entry for a named pragma.

    Raises:
        KeyError: when ``pragma_name`` is not in
            ``compiler_pragmas.yaml``. Applies to allowlisted AND
            rejected entries.
    """
    catalog = load_yaml("compiler_pragmas") or []
    return copy.deepcopy(_by_name(catalog, pragma_name))


@tool_class(ToolClass.READ_ONLY)
def suggest_pragmas_for_kernel(
    kernel_name: str,
    signals: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Apply the three decision rules and return 0-N pragma candidates.

    ``signals`` is a flat dict; recognised keys:

    * ``hotspot_pct`` (float, default 0)            — % of total runtime
    * ``source_path`` (str,   default "")           — Tier-0 anchor path
    * ``loop_trip_count`` (int, optional)           — compile-time trip
    * ``valu_util_pct`` (float, optional)           — 0..1
    * ``vgpr_per_thread`` (int, optional)
    * ``loop_body_size`` (int, optional)            — instruction count
    * ``arch_max_vgpr`` (int, default 256)          — per-thread arch cap
    * ``scratch_size`` (int, default 0)
    * ``waves_per_eu`` (int, default 8)

    The function returns a list of dicts; each dict is a deep copy of
    the catalog entry with two extras: ``kernel_name`` (echoed back)
    and ``triggered_by`` (the ``trigger_signal`` that fired). The
    catalog's ``factor_sweep`` is echoed verbatim — the caller MUST NOT
    invent other factors.

    Returns ``[]`` when:

    * the kernel's hotspot share is below the Amdahl gate,
    * the source path looks like Triton's JIT cache,
    * no decision rule fires.
    """
    sigs = signals or {}

    # -- Amdahl gate --------------------------------------------------
    hotspot = float(sigs.get("hotspot_pct", 0) or 0)
    if hotspot < _MIN_HOTSPOT_PCT:
        return []

    # -- Triton skip --------------------------------------------------
    src_path = str(sigs.get("source_path", "") or "")
    if _TRITON_PATH_FRAGMENT in src_path:
        return []

    catalog = load_yaml("compiler_pragmas") or []
    # Only iterate allowlisted GPU-applicable entries.
    allowlisted = [
        e for e in catalog
        if e.get("gpu_applicable", False) and e.get("allowlist", False)
    ]

    results: List[Dict[str, Any]] = []

    for entry in allowlisted:
        trig = entry.get("trigger_signal", "")
        fired = False

        if trig == "literal_trip_count_on_hotspot":
            trip = sigs.get("loop_trip_count")
            if trip is not None:
                try:
                    trip_i = int(trip)
                except (TypeError, ValueError):
                    trip_i = -1
                if 0 < trip_i <= 32 and hotspot >= 10.0:
                    fired = True

        elif trig == "valu_stalled_small_body":
            valu_util = sigs.get("valu_util_pct")
            vgpr = sigs.get("vgpr_per_thread")
            body = sigs.get("loop_body_size")
            if valu_util is not None and vgpr is not None and body is not None:
                try:
                    if (
                        float(valu_util) > 0.50
                        and int(vgpr) < 64
                        and int(body) < 20
                    ):
                        fired = True
                except (TypeError, ValueError):
                    fired = False

        elif trig == "high_vgpr_pressure":
            vgpr = sigs.get("vgpr_per_thread")
            arch_max = sigs.get("arch_max_vgpr", 256)
            scratch = sigs.get("scratch_size", 0)
            waves = sigs.get("waves_per_eu", 8)
            if vgpr is not None:
                try:
                    if (
                        int(vgpr) >= 0.80 * int(arch_max)
                        and (int(scratch) > 0 or int(waves) <= 2)
                    ):
                        fired = True
                except (TypeError, ValueError):
                    fired = False

        if fired:
            candidate = copy.deepcopy(entry)
            candidate["kernel_name"] = kernel_name
            candidate["triggered_by"] = trig
            results.append(candidate)

    return results


__all__ = [
    "lookup_pragmas",
    "explain_pragma",
    "suggest_pragmas_for_kernel",
]
