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

"""kernel_fusion — find adjacent-short-kernel fusion candidates (Phase-10 A).

Scans the rocprofiler-sdk kernel dispatch timeline for pairs of short,
adjacent kernels (< 10 us each by default) that share a tensor-shape
signature derived from:

  * mangled-name template-argument hash (stripped of address literals)
  * launch dims (``blockDim.x/y/z`` + ``gridDim.x/y/z``)

Each candidate pair comes back with an estimated speedup range ((lo, hi))
computed heuristically from the launch-overhead that fusion would save
relative to the total pair wall-time. Confidence is a bounded blend of
"signature stability" and "gap tightness".

The tool is READ_ONLY — it only queries the database. It never mutates
source, nor does it shell out to compilers. Fusion *candidacy* != fusion
correctness; the Compute Specialist + user review each candidate.

Shared brain contract: the same function backs the MCP surface, the
Compute Specialist allowlist, and CLI integration so every caller gets
identical output for identical input.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from perfxpert.tools._class import ToolClass, tool_class


_DEFAULT_MAX_GAP_NS = 500
_SHORT_KERNEL_NS = 10_000  # 10 us
# Launch overhead saved per fused pair (empirical — HIP kernel launch).
_LAUNCH_OVERHEAD_NS = 6_000
# Signature-hash stability contribution to confidence.
_SIG_CONFIDENCE_BASE = 0.6
_SIG_CONFIDENCE_TIGHT_GAP_BONUS = 0.2


def _find_kernel_dispatch_table(conn: sqlite3.Connection) -> Optional[Tuple[str, str]]:
    """Return (kernel_dispatch_table, info_kernel_symbol_table) if present.

    rocprofiler-sdk tables carry a UUID suffix; both tables share it.
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name LIKE 'rocpd_kernel_dispatch%'"
    )
    row = cur.fetchone()
    if not row:
        return None
    kt = row[0]
    uuid = kt.replace("rocpd_kernel_dispatch_", "")
    info_kt = f"rocpd_info_kernel_symbol_{uuid}"
    return kt, info_kt


def _kernel_signature(
    display_name: str,
    block_x: int, block_y: int, block_z: int,
    grid_x: int, grid_y: int, grid_z: int,
) -> str:
    """Derive a tensor-shape signature for a single dispatch.

    Strategy: drop address literals + numeric template instantiation
    sizes from the mangled name (keeps the "shape" of the template) and
    fold launch dims into the hash. Two dispatches with the same
    signature are fusion-compatible candidates.
    """
    # Strip hex addresses and pointer-like numeric suffixes.
    cleaned = re.sub(r"0x[0-9a-fA-F]+", "#addr", display_name or "")
    # Collapse consecutive digits inside template brackets.
    cleaned = re.sub(r"<\s*[0-9,\s]+\s*>", "<#dims>", cleaned)
    base = f"{cleaned}|b={block_x},{block_y},{block_z}|g={grid_x},{grid_y},{grid_z}"
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
    return digest


def _fetch_dispatches(db_path: str) -> List[Dict[str, Any]]:
    """Pull ordered kernel dispatches with launch dims + names.

    Returns a list of dicts sorted by start time. Falls back gracefully
    when launch-dim columns are absent (legacy schema) — fusion detection
    still works, just with less confidence since sig relies on the name.
    """
    conn = sqlite3.connect(db_path)
    try:
        tables = _find_kernel_dispatch_table(conn)
        if not tables:
            return []
        kt, info_kt = tables

        # Discover which launch-dim columns the dispatch table has.
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({kt})")
        cols = {r[1] for r in cur.fetchall()}

        dim_cols = []
        for c in ("workgroup_size_x", "workgroup_size_y", "workgroup_size_z",
                  "grid_size_x", "grid_size_y", "grid_size_z"):
            dim_cols.append(c if c in cols else "NULL")

        dim_select = ", ".join(dim_cols)
        try:
            rows = conn.execute(f"""
                SELECT kd.start, kd.end, iks.display_name, {dim_select}
                FROM {kt} kd
                JOIN {info_kt} iks ON kd.kernel_id = iks.id
                ORDER BY kd.start ASC
            """).fetchall()
        except sqlite3.OperationalError:
            rows = conn.execute(f"""
                SELECT start, end, name, {dim_select}
                FROM {kt}
                ORDER BY start ASC
            """).fetchall()

        out: List[Dict[str, Any]] = []
        for r in rows:
            start, end, name = r[0], r[1], r[2]
            bx, by, bz, gx, gy, gz = (r[3] or 0, r[4] or 0, r[5] or 0,
                                     r[6] or 0, r[7] or 0, r[8] or 0)
            out.append({
                "start_ns": int(start),
                "end_ns": int(end),
                "duration_ns": int(end) - int(start),
                "name": name or "",
                "block_dim": (int(bx), int(by), int(bz)),
                "grid_dim": (int(gx), int(gy), int(gz)),
            })
        return out
    finally:
        conn.close()


def _estimate_speedup_bounds(
    pair_duration_ns: int,
    gap_ns: int,
) -> Tuple[float, float]:
    """Return (lo, hi) speedup bound estimates for a fused pair.

    lo = 1.0 + (overhead saved) / (pair_total_incl_gap)
    hi = 1.0 + (overhead saved + gap_reclaim) / (pair_total_incl_gap)
    """
    total = max(pair_duration_ns + gap_ns, 1)
    lo = 1.0 + _LAUNCH_OVERHEAD_NS / total
    hi = 1.0 + (_LAUNCH_OVERHEAD_NS + gap_ns) / total
    # Cap at 3x — fusion alone rarely exceeds this without algorithmic change.
    return (round(min(lo, 3.0), 3), round(min(hi, 3.0), 3))


def _confidence(gap_ns: int, max_gap_ns: int) -> float:
    if max_gap_ns <= 0:
        return _SIG_CONFIDENCE_BASE
    tight = 1.0 - min(gap_ns, max_gap_ns) / max_gap_ns
    return round(
        min(_SIG_CONFIDENCE_BASE + _SIG_CONFIDENCE_TIGHT_GAP_BONUS * tight, 0.95),
        3,
    )


@tool_class(ToolClass.READ_ONLY)
def find_fusion_candidates(
    db_path: str,
    max_gap_ns: int = _DEFAULT_MAX_GAP_NS,
) -> List[Dict[str, Any]]:
    """Scan the kernel timeline for fusion-candidate pairs.

    Args:
        db_path: rocprofiler-sdk .db path.
        max_gap_ns: Max ns between consecutive kernels to still count as
            "adjacent". Defaults to 500.

    Returns:
        List of dicts, ranked by ``est_speedup_hi`` descending::

            {
              "pair": [name1, name2],
              "signature": "<12-hex signature>",
              "gap_ns": int,
              "est_speedup_lo": float,
              "est_speedup_hi": float,
              "expected_impact_units": "speedup_multiplier",
              "confidence": float,  # 0..1
            }

        ``expected_impact_units`` disambiguates the numeric meaning of
        the ``est_speedup_*`` bounds the way
        :func:`perfxpert.knowledge.fusion_patterns` entries carry a
        ``units`` field for their ``expected_impact`` value.
        ``est_speedup_{lo,hi}`` here are always
        ``speedup_multiplier`` (``1.0 + overhead_saved / total``), never
        a time-saved fraction; surfacing the label prevents
        downstream consumers from re-interpreting the number.

        Empty list if the DB has no short-adjacent kernels meeting both
        duration (< 10 us each) and gap (``< max_gap_ns``) thresholds.
    """
    dispatches = _fetch_dispatches(db_path)
    candidates: List[Dict[str, Any]] = []

    for i in range(len(dispatches) - 1):
        a = dispatches[i]
        b = dispatches[i + 1]
        if a["duration_ns"] >= _SHORT_KERNEL_NS:
            continue
        if b["duration_ns"] >= _SHORT_KERNEL_NS:
            continue
        gap = b["start_ns"] - a["end_ns"]
        if gap < 0 or gap > max_gap_ns:
            continue

        sig_a = _kernel_signature(a["name"], *a["block_dim"], *a["grid_dim"])
        sig_b = _kernel_signature(b["name"], *b["block_dim"], *b["grid_dim"])
        if sig_a != sig_b:
            continue

        pair_dur = a["duration_ns"] + b["duration_ns"]
        lo, hi = _estimate_speedup_bounds(pair_dur, gap)
        candidates.append({
            "pair": [a["name"], b["name"]],
            "signature": sig_a,
            "gap_ns": int(gap),
            "est_speedup_lo": lo,
            "est_speedup_hi": hi,
            # Propagate the same ``units`` vocabulary that
            # knowledge/fusion_patterns.yaml uses so downstream consumers
            # never have to guess whether a number is a fractional time
            # saved or a multiplicative speedup.
            "expected_impact_units": "speedup_multiplier",
            "confidence": _confidence(gap, max_gap_ns),
        })

    candidates.sort(key=lambda c: c["est_speedup_hi"], reverse=True)
    return candidates


__all__ = ["find_fusion_candidates"]
