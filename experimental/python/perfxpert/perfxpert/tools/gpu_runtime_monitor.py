###############################################################################
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
###############################################################################

"""gpu_runtime_monitor — parse pre-captured amd-smi / rocm-smi JSON logs.

We do NOT shell out to ``amd-smi`` / ``rocm-smi`` at analyze time: users
capture the JSON in advance (``amd-smi monitor --json > log.json``) and
we ingest it offline. Eliminates the need for sudo, live sampling, and
host-side privilege escalation inside analyze.

Users can either pass the log path directly to the parsers or set
``PERFXPERT_GPU_MONITOR_LOG=<path>`` to let other modules auto-pick it up.

Tool class: READ_ONLY.
"""

from __future__ import annotations

import json
import os
import statistics
from typing import Any, Dict, List

from perfxpert.tools._class import ToolClass, tool_class

# Env var surface (Latency / Memory specialists may read this).
PERFXPERT_GPU_MONITOR_LOG = "PERFXPERT_GPU_MONITOR_LOG"

# Default thermal thresholds — overridable per-arch via thermal_specs.yaml.
_DEFAULT_TJMAX_C = 105.0
_DEFAULT_TJ_THROTTLE_MARGIN_C = 5.0   # throttling begins within 5 C of TjMax
_DEFAULT_POWER_HEADROOM_PCT = 0.95    # 95 % of TDP = power-throttle suspect


def _coerce_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@tool_class(ToolClass.READ_ONLY)
def parse_amd_smi_json(path: str) -> Dict[str, Any]:
    """Parse a captured ``amd-smi monitor --json`` log.

    Args:
        path: Path to the amd-smi JSON file. The file may be a single
            snapshot (dict) or a time-series (list of snapshots).

    Returns:
        Normalized dict::

            {
              "source": "amd-smi",
              "samples": [
                 {"gpu_id": int, "temp_c": float, "power_w": float,
                  "sclk_mhz": float, "mclk_mhz": float,
                  "gfx_busy_pct": float},
                 ...
              ],
              "gpu_count": int,
            }

        Returns ``{"source": "amd-smi", "samples": [], "gpu_count": 0}``
        on parse failure — the tool never raises.
    """
    try:
        raw = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return {"source": "amd-smi", "samples": [], "gpu_count": 0}

    snapshots = raw if isinstance(raw, list) else [raw]
    samples: List[Dict[str, Any]] = []
    gpu_ids: set = set()

    for snap in snapshots:
        if not isinstance(snap, dict):
            continue
        # amd-smi JSON nests per-GPU records under "gpu" or uses list form.
        gpus = snap.get("gpu") or snap.get("gpus") or []
        if isinstance(gpus, dict):
            gpus = [gpus]
        for rec in gpus:
            if not isinstance(rec, dict):
                continue
            gpu_id = rec.get("gpu_id", rec.get("id", 0))
            temp_c = _coerce_float(
                rec.get("temp_edge_c",
                        rec.get("temperature", {}).get("edge", 0)
                        if isinstance(rec.get("temperature"), dict) else
                        rec.get("temp", 0))
            )
            power_w = _coerce_float(
                rec.get("power_w",
                        rec.get("power", {}).get("socket", 0)
                        if isinstance(rec.get("power"), dict) else
                        rec.get("power", 0))
            )
            sclk_mhz = _coerce_float(rec.get("sclk_mhz", rec.get("sclk", 0)))
            mclk_mhz = _coerce_float(rec.get("mclk_mhz", rec.get("mclk", 0)))
            gfx_busy = _coerce_float(rec.get("gfx_busy_pct", rec.get("utilization", 0)))
            samples.append({
                "gpu_id": int(gpu_id) if isinstance(gpu_id, (int, str)) else 0,
                "temp_c": temp_c,
                "power_w": power_w,
                "sclk_mhz": sclk_mhz,
                "mclk_mhz": mclk_mhz,
                "gfx_busy_pct": gfx_busy,
            })
            gpu_ids.add(int(gpu_id) if isinstance(gpu_id, (int, str)) else 0)

    return {
        "source": "amd-smi",
        "samples": samples,
        "gpu_count": len(gpu_ids),
    }


@tool_class(ToolClass.READ_ONLY)
def parse_rocm_smi_json(path: str) -> Dict[str, Any]:
    """Parse a captured ``rocm-smi --json`` log (fallback for older stacks).

    rocm-smi emits ``{"card0": {...}, "card1": {...}, ...}`` at the top
    level; we flatten it into the same shape as :func:`parse_amd_smi_json`.
    """
    try:
        raw = _read_json(path)
    except (OSError, json.JSONDecodeError):
        return {"source": "rocm-smi", "samples": [], "gpu_count": 0}

    if not isinstance(raw, dict):
        return {"source": "rocm-smi", "samples": [], "gpu_count": 0}

    samples: List[Dict[str, Any]] = []
    for key, rec in raw.items():
        if not key.startswith("card"):
            continue
        if not isinstance(rec, dict):
            continue
        try:
            gpu_id = int(key.replace("card", ""))
        except ValueError:
            gpu_id = 0
        # rocm-smi keys carry suffixes like "Temperature (Sensor edge) (C)".
        temp_c = 0.0
        for k, v in rec.items():
            if "Temperature" in k and "edge" in k.lower():
                temp_c = _coerce_float(v)
                break
        power_w = _coerce_float(rec.get("Average Graphics Package Power (W)", 0))
        sclk_raw = rec.get("sclk clock level", rec.get("sclk", "0Mhz"))
        sclk_mhz = _coerce_float(str(sclk_raw).replace("Mhz", "").replace("MHz", ""))
        mclk_raw = rec.get("mclk clock level", rec.get("mclk", "0Mhz"))
        mclk_mhz = _coerce_float(str(mclk_raw).replace("Mhz", "").replace("MHz", ""))
        gfx_busy = _coerce_float(rec.get("GPU use (%)", rec.get("GPU use", 0)))
        samples.append({
            "gpu_id": gpu_id,
            "temp_c": temp_c,
            "power_w": power_w,
            "sclk_mhz": sclk_mhz,
            "mclk_mhz": mclk_mhz,
            "gfx_busy_pct": gfx_busy,
        })

    return {
        "source": "rocm-smi",
        "samples": samples,
        "gpu_count": len({s["gpu_id"] for s in samples}),
    }


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(int(round((pct / 100.0) * (len(s) - 1))), 0)
    return float(s[k])


@tool_class(ToolClass.READ_ONLY)
def analyze_thermal(
    metrics: Dict[str, Any],
    tjmax_c: float = _DEFAULT_TJMAX_C,
    tdp_w: float = 0.0,
) -> Dict[str, Any]:
    """Derive a thermal / throttle-envelope summary from parsed samples.

    Args:
        metrics: Output of :func:`parse_amd_smi_json` or
            :func:`parse_rocm_smi_json`.
        tjmax_c: Junction-temp ceiling (arch-dependent — see
            thermal_specs.yaml).
        tdp_w: TDP in watts (0 = unknown, skips power-throttle check).

    Returns:
        Dict::

            {
              "max_temp_c": float,
              "avg_temp_c": float,
              "p95_temp_c": float,
              "max_power_w": float,
              "throttle_events": int,
              "power_throttle_suspected": bool,
              "thermal_headroom_c": float,
              "verdict": "healthy" | "hot" | "throttling",
            }
    """
    samples = metrics.get("samples", []) if isinstance(metrics, dict) else []
    if not samples:
        return {
            "max_temp_c": 0.0, "avg_temp_c": 0.0, "p95_temp_c": 0.0,
            "max_power_w": 0.0, "throttle_events": 0,
            "power_throttle_suspected": False,
            "thermal_headroom_c": tjmax_c,
            "verdict": "healthy",
        }

    temps = [_coerce_float(s.get("temp_c", 0)) for s in samples]
    powers = [_coerce_float(s.get("power_w", 0)) for s in samples]

    max_t = max(temps) if temps else 0.0
    avg_t = statistics.fmean(temps) if temps else 0.0
    p95_t = _percentile(temps, 95.0)
    max_p = max(powers) if powers else 0.0

    throttle_margin = tjmax_c - _DEFAULT_TJ_THROTTLE_MARGIN_C
    throttle_events = sum(1 for t in temps if t >= throttle_margin)
    power_throttle = bool(tdp_w > 0 and max_p >= tdp_w * _DEFAULT_POWER_HEADROOM_PCT)

    headroom = max(tjmax_c - max_t, 0.0)
    if throttle_events > 0 or power_throttle:
        verdict = "throttling"
    elif max_t >= throttle_margin - 5.0:
        verdict = "hot"
    else:
        verdict = "healthy"

    return {
        "max_temp_c": round(max_t, 2),
        "avg_temp_c": round(avg_t, 2),
        "p95_temp_c": round(p95_t, 2),
        "max_power_w": round(max_p, 2),
        "throttle_events": int(throttle_events),
        "power_throttle_suspected": power_throttle,
        "thermal_headroom_c": round(headroom, 2),
        "verdict": verdict,
    }


def resolve_monitor_log_path() -> str:
    """Return the user-provided monitor log path or empty string."""
    return os.environ.get(PERFXPERT_GPU_MONITOR_LOG, "").strip()


__all__ = [
    "parse_amd_smi_json",
    "parse_rocm_smi_json",
    "analyze_thermal",
    "resolve_monitor_log_path",
    "PERFXPERT_GPU_MONITOR_LOG",
]
