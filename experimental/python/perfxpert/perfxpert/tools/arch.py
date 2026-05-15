"""arch — GPU architecture lookup tool.

Replaces free-text GPU spec recitation in the fence with structured lookup.
Agents call lookup_peaks(gfx_id) instead of embedding specs in prompts.

Tool class: READ_ONLY (MCP-safe).

Static knowledge remains the default for known architectures so trace analysis
does not accidentally use controller-host specs for a remote target. Runtime
GPU facts are used when explicitly requested by local init, or when a local
runtime-only architecture is not present in the static table.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

from perfxpert.knowledge import load_yaml
from perfxpert.tools._class import ToolClass, tool_class


_RIDGE_PEAK_KEYS = {
    "fp64": "peak_fp64_tflops",
    "fp32": "peak_fp32_tflops",
    "fp16": "peak_fp16_tflops",
    "bf16": "peak_bf16_tflops",
    "fp8": "peak_fp8_tflops",
    "int8": "peak_int8_tops",
}


@lru_cache(maxsize=1)
def _gpu_specs() -> Dict[str, Dict[str, Any]]:
    return load_yaml("gpu_specs")


def _ridge_point_from_specs(specs: Dict[str, Any], dtype: str = "fp32") -> float:
    peak_key = _RIDGE_PEAK_KEYS.get(dtype, "peak_fp32_tflops")
    peak_tflops = float(specs.get(peak_key) or specs.get("peak_fp32_tflops") or 0.0)
    bandwidth_tbs = float(specs.get("memory_bandwidth_tbs") or 0.0)
    if bandwidth_tbs <= 0:
        return float(specs.get("ridge_point") or 0.0)
    return round(peak_tflops / bandwidth_tbs, 1)


def _runtime_specs_for_gfx(gfx_id: str) -> Dict[str, Any]:
    from perfxpert.tools.gpu_discovery import runtime_specs_for_gfx

    return runtime_specs_for_gfx(gfx_id) or {}


def _merge_static_and_runtime_specs(
    static_specs: Dict[str, Any],
    runtime_specs: Dict[str, Any],
) -> Dict[str, Any]:
    result = dict(static_specs)
    static_keys = set(result.keys())
    spec_sources = {key: "gpu_specs.yaml" for key in result}

    for key, value in runtime_specs.items():
        if key == "spec_sources":
            continue
        if value is None or value == "":
            continue
        source = runtime_specs.get("spec_sources", {}).get(key, "runtime")
        if (
            key in result
            and source.startswith("derived-from-")
            and (key.startswith("peak_") or key in {"peak_int8_tops", "memory_bandwidth_tbs"})
        ):
            continue
        result[key] = value
        spec_sources[key] = source

    if runtime_specs:
        result["runtime_discovered"] = True
        result["static_fallback_keys"] = sorted(static_keys - set(runtime_specs.keys()))
    else:
        result["runtime_discovered"] = False
        result["static_fallback_keys"] = []
    result["spec_sources"] = spec_sources
    return result


def lookup_ridge_point(gfx_id: str, dtype: str = "fp32") -> float:
    """Return the arch ridge point for a specific dtype.

    The stored ``ridge_point`` is the default FP32/vector ridge. Dtype-specific
    callers (for example live roofline plots) should use this helper so the
    chart ceiling and the scalar classifier do not drift apart.
    """
    return _ridge_point_from_specs(lookup_peaks(gfx_id), dtype=dtype)


@lru_cache(maxsize=1)
def occupancy_specs_table() -> Dict[str, Dict[str, Any]]:
    """Return occupancy-specific arch caps derived from ``gpu_specs.yaml``."""
    table: Dict[str, Dict[str, Any]] = {}
    for gfx_id, spec in _gpu_specs().items():
        table[gfx_id] = {
            "max_waves_per_simd": int(spec["max_waves_per_simd"]),
            "vgprs_per_simd": int(spec["vgprs_per_simd"]),
            "lds_per_cu_kb": int(spec["lds_per_cu_kb"]),
            "wavefront_size": int(spec["wave_size"]),
            "simds_per_cu": int(spec["simds_per_cu"]),
        }
    return table


@tool_class(ToolClass.READ_ONLY)
def lookup_peaks(gfx_id: str, prefer_runtime: bool = False) -> Dict[str, Any]:
    """Return hardware peak specs for a given gfx architecture.

    Args:
        gfx_id: Architecture identifier, e.g., "gfx942" for MI300X.
        prefer_runtime: Use local runtime facts for known architectures. Leave
            false for trace/remote analysis where the profiled host may differ
            from the controller running PerfXpert.

    Returns:
        Dict with keys: name, codename, peak_fp64_tflops, peak_fp32_tflops,
        peak_bf16_tflops (where applicable), memory_bandwidth_tbs, cu_count,
        lds_kb, wave_size, max_vgprs_per_thread, ridge_point.

    Raises:
        KeyError: if gfx_id is not recognized. Error includes list of known archs.

    Example:
        >>> from perfxpert.tools.arch import lookup_peaks
        >>> mi300x = lookup_peaks("gfx942")
        >>> mi300x["peak_fp64_tflops"]
        81.7
    """
    specs = _gpu_specs()
    runtime_specs = _runtime_specs_for_gfx(gfx_id) if prefer_runtime or gfx_id not in specs else {}
    if gfx_id not in specs and not runtime_specs:
        known = ", ".join(sorted(specs.keys()))
        raise KeyError(f"Unknown gfx_id {gfx_id!r}; known archs: {known}")

    result = _merge_static_and_runtime_specs(specs.get(gfx_id, {}), runtime_specs)
    result["ridge_point"] = _ridge_point_from_specs(result, dtype="fp32")
    result["ridge_points"] = {dtype: _ridge_point_from_specs(result, dtype=dtype) for dtype in _RIDGE_PEAK_KEYS}
    return result
