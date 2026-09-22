"""roofline — arithmetic intensity + regime classification + live-chart points.

Pure arithmetic. Replaces the LLM's roofline-model reasoning with a single
rule: if AI > ridge_point -> compute; else memory.

Tool class: READ_ONLY.

Public tools
------------
- ``classify(flops, bytes, gfx_id)`` — scalar regime lookup (existing).
- ``plot_points(db_path, top_k, dtype_hint)`` — NEW (phase-10, Live Roofline):
  returns per-kernel ``(AI, achieved_flops_per_s)`` points plus the
  arch's ridge point so the webview formatter can render a
  self-contained log-log roofline SVG. READ_ONLY against a rocpd DB —
  no network, no writes, no subprocess.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from perfxpert.tools._class import ToolClass, tool_class
# Imported as a private alias so the tool registry (which walks
# perfxpert.tools.* and collects public READ_ONLY callables) does not
# re-register this as `roofline.lookup_peaks`. Canonical name is
# `arch.lookup_peaks`; this module re-uses it internally only.
from perfxpert.tools.arch import (
    lookup_peaks as _lookup_peaks,
    lookup_ridge_point as _lookup_ridge_point,
)


_BALANCED_TOLERANCE = 0.05  # +/-5% around ridge = "balanced"


@tool_class(ToolClass.READ_ONLY)
def classify(flops: float, bytes: float, gfx_id: str) -> Dict[str, Any]:
    """Classify a kernel's regime using the roofline model.

    Args:
        flops: Total floating-point operations for the kernel.
        bytes: Total memory bytes read+written (HBM traffic).
        gfx_id: Architecture identifier (e.g., "gfx942").

    Returns:
        {
            "arithmetic_intensity": FLOPS/Byte,
            "ridge_point": ridge for this arch,
            "regime": "compute" | "memory" | "balanced",
            "distance_to_roof": 0.0-1.0 (how far below peak)
        }

    Raises:
        ValueError: if bytes == 0.
        KeyError: if gfx_id is not recognized.

    Example:
        >>> classify(flops=1e12, bytes=1e11, gfx_id="gfx942")
        {"arithmetic_intensity": 10.0, "ridge_point": 30.8, "regime": "memory", ...}
    """
    if bytes == 0:
        raise ValueError("bytes must be > 0 — divide-by-zero in arithmetic intensity")

    ridge = _lookup_ridge_point(gfx_id, dtype="fp32")
    ai = flops / bytes

    if abs(ai - ridge) / ridge <= _BALANCED_TOLERANCE:
        regime = "balanced"
    elif ai > ridge:
        regime = "compute"
    else:
        regime = "memory"

    if regime == "compute":
        distance = 0.5
    else:
        distance = min(ai / ridge, 1.0)

    return {
        "arithmetic_intensity": ai,
        "ridge_point": ridge,
        "regime": regime,
        "distance_to_roof": distance,
    }


# ---------------------------------------------------------------------------
# plot_points — phase-10 Live Roofline
# ---------------------------------------------------------------------------


_DTYPE_PATTERNS: List[tuple] = [
    (re.compile(r"(_bf16|_bfloat16)", re.IGNORECASE), "bf16"),
    (re.compile(r"(_mfma_f8|_fp8|_f8\b)", re.IGNORECASE), "fp8"),
    (re.compile(r"(_int8|_i8\b)", re.IGNORECASE), "int8"),
    (re.compile(r"(_fp16|_f16\b|_half)", re.IGNORECASE), "fp16"),
]

_LEGACY_MFMA_COUNTER = "SQ_INSTS_VALU_MFMA"
_MFMA_MOPS_FLOPS_PER_COUNT = 512.0
_MOPS_COUNTERS = {
    "SQ_INSTS_VALU_MFMA_MOPS_BF16": "bf16",
    "SQ_INSTS_VALU_MFMA_MOPS_F16": "fp16",
    "SQ_INSTS_VALU_MFMA_MOPS_F32": "fp32",
    "SQ_INSTS_VALU_MFMA_MOPS_F64": "fp64",
    "SQ_INSTS_VALU_MFMA_MOPS_I8": "int8",
    "SQ_INSTS_VALU_MFMA_MOPS_F8": "fp8",
}


def _detect_dtype(kernel_name: str) -> str:
    if not kernel_name:
        return "fp32"
    for rx, label in _DTYPE_PATTERNS:
        if rx.search(kernel_name):
            return label
    return "fp32"


def _peak_for_dtype(specs: Dict[str, Any], dtype: str) -> float:
    key_map = {
        "fp32": "peak_fp32_tflops",
        "fp16": "peak_fp16_tflops",
        "bf16": "peak_bf16_tflops",
        "fp8":  "peak_fp8_tflops",
        "int8": "peak_int8_tops",
    }
    k = key_map.get(dtype, "peak_fp32_tflops")
    val = specs.get(k) or specs.get("peak_fp32_tflops") or 0.0
    return float(val) * 1e12


def _detect_gfx_id(conn: sqlite3.Connection) -> str:
    try:
        cur = conn.execute(
            "SELECT name FROM rocpd_info_agent WHERE name LIKE 'gfx%' LIMIT 1"
        )
        row = cur.fetchone()
    except sqlite3.DatabaseError:
        row = None
    if row is None:
        return "gfx942"
    name = row[0] if not hasattr(row, "keys") else row["name"]
    m = re.search(r"gfx\w+", str(name or ""))
    return m.group(0) if m else "gfx942"


def _fetch_kernel_counters(conn: sqlite3.Connection) -> Dict[str, Dict[str, Any]]:
    wanted = (
        "SQ_INSTS_VALU",
        _LEGACY_MFMA_COUNTER,
        *_MOPS_COUNTERS,
        "FETCH_SIZE",
        "WRITE_SIZE",
    )
    placeholders = ",".join("?" for _ in wanted)
    q = (
        "SELECT name, counter_name, SUM(counter_value), SUM(end - start) "
        "FROM pmc_events "
        f"WHERE counter_name IN ({placeholders}) "
        "GROUP BY name, counter_name"
    )
    out: Dict[str, Dict[str, Any]] = {}
    try:
        cur = conn.execute(q, wanted)
    except sqlite3.DatabaseError:
        return out
    for row in cur.fetchall():
        name, cname, total, duration = (
            row[0], row[1], row[2], row[3]
        ) if not hasattr(row, "keys") else (
            row["name"], row["counter_name"], row[2], row[3]
        )
        k = str(name or "unknown")
        slot = out.setdefault(k, {"_duration_ns": 0})
        slot[cname] = float(total or 0)
        if int(duration or 0) > slot["_duration_ns"]:
            slot["_duration_ns"] = int(duration or 0)
    return out


def _mfma_flops_from_counters(
    ctrs: Dict[str, Any], fp_type: str, mfma_table: Dict[str, Any]
) -> float:
    mops_flops = sum(
        float(ctrs.get(counter_name, 0)) * _MFMA_MOPS_FLOPS_PER_COUNT
        for counter_name in _MOPS_COUNTERS
    )
    if mops_flops > 0:
        return mops_flops

    legacy_mfma = float(ctrs.get(_LEGACY_MFMA_COUNTER, 0))
    mfma_flops_per_inst = float(mfma_table.get(fp_type, 0))
    return legacy_mfma * mfma_flops_per_inst


@tool_class(ToolClass.READ_ONLY)
def plot_points(
    db_path: str,
    top_k: int = 10,
    dtype_hint: Optional[str] = None,
) -> Dict[str, Any]:
    """Return per-kernel roofline points for a rocpd database.

    Formula (deterministic — no LLM involved)::

        flops = SQ_INSTS_VALU * 64
              + sum(SQ_INSTS_VALU_MFMA_MOPS_* * 512)
        # fallback when only legacy MFMA instruction counts are present
              + SQ_INSTS_VALU_MFMA * mfma_flops_per_inst[dtype]
        bytes = (FETCH_SIZE + WRITE_SIZE) * 1024      # TCC KiB -> bytes
        ai    = flops / bytes
        rate  = flops / (duration_ns / 1e9)

    Args:
        db_path: Path to a rocpd-produced SQLite database.
        top_k: Maximum number of kernels to return, ranked by
            ``duration_pct`` descending. Defaults to 10.
        dtype_hint: When provided, force this dtype for every kernel
            instead of regex-detecting from the kernel name. Expected
            values: ``fp32 / fp16 / bf16 / fp8 / int8``.

    Returns:
        The ``roofline`` payload dict. Schema:
            {
              "schema_version": "0.3.x",
              "arch": str,
              "arch_peaks": {"fp32": ..., "fp16": ..., "bf16": ...,
                             "fp64": ..., "fp8": ..., "int8": ...},
              "hbm_bandwidth_bytes_per_s": float,
              "dtype": str,
              "dtype_confidence": "from_kernel_name" | "default",
              "kernels": [
                {"name": str, "ai": float, "achieved_flops_per_s": float,
                 "flops": float, "bytes": int,
                 "duration_ns": int, "duration_pct": float,
                 "fp_type": str,
                 "bottleneck_class": "compute"|"memory"|"balanced",
                 "confidence": "high"|"low"}, ...
              ],
              "ridge_point": {"ai": float, "flops_per_s": float}
            }

        When the database has no ``pmc_events`` view or no rows for the
        wanted counters, ``kernels`` is an empty list but the arch +
        peaks are still filled in so the formatter can render an empty
        chart labelled with the correct ridge point.

    Raises:
        FileNotFoundError: if ``db_path`` does not exist.
    """
    p = Path(db_path)
    if not p.exists():
        raise FileNotFoundError(f"rocpd database not found: {db_path}")

    conn = sqlite3.connect(str(p))
    try:
        gfx_id = _detect_gfx_id(conn)
        specs = _lookup_peaks(gfx_id)
        per_kernel = _fetch_kernel_counters(conn)
    finally:
        conn.close()

    mfma_table = dict(specs.get("mfma_flops_per_inst") or {})

    kernels_out: List[Dict[str, Any]] = []
    dtype_votes: Dict[str, float] = {}

    total_duration_ns = sum(
        int(v.get("_duration_ns", 0)) for v in per_kernel.values()
    )

    for name, ctrs in per_kernel.items():
        valu = float(ctrs.get("SQ_INSTS_VALU", 0))
        fetch = float(ctrs.get("FETCH_SIZE", 0))
        write = float(ctrs.get("WRITE_SIZE", 0))
        duration_ns = int(ctrs.get("_duration_ns", 0))
        if duration_ns <= 0:
            continue

        if dtype_hint:
            fp_type = str(dtype_hint).lower()
        else:
            fp_type = _detect_dtype(name)

        kib = fetch + write
        bytes_total = int(kib * 1024)
        if bytes_total <= 0:
            continue
        confidence = "low" if (fetch == 0) ^ (write == 0) else "high"

        flops = valu * 64.0 + _mfma_flops_from_counters(ctrs, fp_type, mfma_table)
        if flops <= 0:
            continue

        ai = flops / bytes_total
        rate = flops / (duration_ns / 1e9)

        try:
            regime = classify(flops=flops, bytes=bytes_total, gfx_id=gfx_id)[
                "regime"
            ]
        except (ValueError, KeyError):
            regime = "balanced"

        duration_pct = (
            (duration_ns / total_duration_ns * 100.0) if total_duration_ns > 0 else 0.0
        )

        kernels_out.append(
            {
                "name": name,
                "ai": ai,
                "achieved_flops_per_s": rate,
                "flops": flops,
                "bytes": bytes_total,
                "duration_ns": duration_ns,
                "duration_pct": duration_pct,
                "fp_type": fp_type,
                "bottleneck_class": regime,
                "confidence": confidence,
            }
        )
        dtype_votes[fp_type] = dtype_votes.get(fp_type, 0.0) + duration_pct

    kernels_out.sort(key=lambda k: k["duration_pct"], reverse=True)
    kernels_out = kernels_out[: max(0, int(top_k))]

    if dtype_hint:
        dominant_dtype = str(dtype_hint).lower()
        dtype_confidence = "from_kernel_name"
    elif dtype_votes:
        dominant_dtype = max(dtype_votes.items(), key=lambda kv: kv[1])[0]
        dtype_confidence = (
            "from_kernel_name" if dominant_dtype != "fp32" else "default"
        )
    else:
        dominant_dtype = "fp32"
        dtype_confidence = "default"

    arch_peaks = {
        "fp32": _peak_for_dtype(specs, "fp32"),
        "fp16": _peak_for_dtype(specs, "fp16"),
        "bf16": _peak_for_dtype(specs, "bf16"),
        "fp64": float(specs.get("peak_fp64_tflops") or 0.0) * 1e12,
        "fp8":  _peak_for_dtype(specs, "fp8"),
        "int8": _peak_for_dtype(specs, "int8"),
    }
    hbm_bps = float(specs.get("memory_bandwidth_tbs") or 0.0) * 1e12

    dom_peak = arch_peaks.get(dominant_dtype) or arch_peaks["fp32"]
    ridge_ai = _lookup_ridge_point(gfx_id, dtype=dominant_dtype)

    return {
        "schema_version": "0.3.x",
        "arch": gfx_id,
        "arch_peaks": arch_peaks,
        "hbm_bandwidth_bytes_per_s": hbm_bps,
        "dtype": dominant_dtype,
        "dtype_confidence": dtype_confidence,
        "kernels": kernels_out,
        "ridge_point": {"ai": ridge_ai, "flops_per_s": dom_peak},
    }
