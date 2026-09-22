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

"""unified_memory — MI300X paging + cross-die penalty analysis (Phase-10 C).

Detects two classes of unified-memory pathology visible in a
rocprofiler-sdk trace:

  1. CPU-resident pages that the GPU touches (HtoD / DtoH memcpy spikes
     and, when present, page-fault / HMM migration events).
  2. Cross-die traffic on MI300X where XCD-to-XCD access pays a small
     (~30 ns baseline) latency penalty vs in-die HBM.

Returns a lightweight summary dict + recommendations; the Memory
Specialist consumes the shape + folds it into its per-kernel advice.

Tool class: READ_ONLY. DB-only — we never probe live device state.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from perfxpert.tools._class import ToolClass, tool_class


# MI300X cross-die (XCD<->XCD via AID fabric) latency overhead vs local HBM.
# Empirical per-access penalty (ns). Extend via knowledge/memory_patterns.yaml.
_MI300X_CROSS_DIE_NS_PER_ACCESS = 30

# Threshold for flagging HtoD/DtoH bandwidth as "paging-like".
_PAGING_MEMCPY_MIN_BYTES = 1 << 20  # 1 MiB


def _find_memcpy_table(conn: sqlite3.Connection) -> Optional[str]:
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name LIKE 'rocpd_memory_copy%'"
    )
    row = cur.fetchone()
    return row[0] if row else None


def _find_page_fault_table(conn: sqlite3.Connection) -> Optional[str]:
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name LIKE '%page_fault%'"
    )
    row = cur.fetchone()
    return row[0] if row else None


def _count_paging_memcpy(conn: sqlite3.Connection, memcpy_table: str) -> Tuple[int, int]:
    """Return (paging_event_count, total_bytes_transferred)."""
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({memcpy_table})")
    cols = {r[1] for r in cur.fetchall()}

    # Columns vary: rocprofv3 uses (size, kind), legacy uses (bytes, direction).
    size_col = "size" if "size" in cols else ("bytes" if "bytes" in cols else None)
    kind_col = "kind" if "kind" in cols else ("direction" if "direction" in cols else None)
    if not size_col:
        return 0, 0

    try:
        if kind_col:
            rows = conn.execute(
                f"SELECT {size_col}, {kind_col} FROM {memcpy_table}"
            ).fetchall()
        else:
            rows = conn.execute(f"SELECT {size_col}, NULL FROM {memcpy_table}").fetchall()
    except sqlite3.OperationalError:
        return 0, 0

    count = 0
    total = 0
    for size, kind in rows:
        size = int(size or 0)
        total += size
        # "HtoD" / "DtoH" or numeric kind codes that indicate host-device
        # migrations — anything not pure DtoD counts as a "paging-like" move.
        kstr = str(kind or "").upper()
        is_host_dev = (
            "HTOD" in kstr or "DTOH" in kstr or "HOST" in kstr
            or kstr in ("1", "2")  # legacy numeric kinds
        )
        if is_host_dev and size >= _PAGING_MEMCPY_MIN_BYTES:
            count += 1
    return count, total


def _cross_die_bytes_estimate(
    conn: sqlite3.Connection, memcpy_table: str
) -> int:
    """Best-effort estimate of XCD<->XCD bytes on MI300X.

    Without PCIe-peer column data we approximate using DtoD memcpys that
    cross GPU ids (or when ``src_agent_id != dst_agent_id``). Falls back
    to 0 when the schema lacks agent columns.
    """
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({memcpy_table})")
    cols = {r[1] for r in cur.fetchall()}
    size_col = "size" if "size" in cols else ("bytes" if "bytes" in cols else None)
    if not size_col:
        return 0

    src = next((c for c in ("src_agent_id", "src_device_id", "src_gpu_id") if c in cols), None)
    dst = next((c for c in ("dst_agent_id", "dst_device_id", "dst_gpu_id") if c in cols), None)
    if not src or not dst:
        return 0

    try:
        rows = conn.execute(
            f"SELECT {size_col} FROM {memcpy_table} WHERE {src} != {dst}"
        ).fetchall()
    except sqlite3.OperationalError:
        return 0
    return int(sum(r[0] or 0 for r in rows))


def _page_fault_count(conn: sqlite3.Connection, table: str) -> int:
    try:
        row = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        return int(row[0]) if row else 0
    except sqlite3.OperationalError:
        return 0


def _recommendations(
    paging_events: int,
    cross_die_bytes: int,
    page_faults: int,
) -> List[str]:
    recs: List[str] = []
    if paging_events > 0 or page_faults > 0:
        recs.append(
            "Pin host-shared buffers with hipHostMalloc(kHostNonCoherent) or "
            "pre-touch via hipMemAdvise(hipMemAdviseSetPreferredLocation) to "
            "keep pages GPU-resident."
        )
        recs.append(
            "For repeated HtoD transfers of the same buffer, promote to a "
            "persistent device allocation + single upload at init."
        )
    if cross_die_bytes >= (1 << 30):  # 1 GiB
        recs.append(
            "MI300X: pin hot tensors to a single XCD via ROCR_VISIBLE_DEVICES "
            "partitioning; cross-die DtoD >= 1 GiB pays the XCD fabric tax."
        )
        recs.append(
            "Consider torch.cuda.set_device() partitioning or GROUPED device "
            "selection so collectives stay in-die when possible."
        )
    if not recs:
        recs.append("No unified-memory pathology detected — current pinning looks healthy.")
    return recs


@tool_class(ToolClass.READ_ONLY)
def analyze_paging(db_path: str) -> Dict[str, Any]:
    """Quantify unified-memory paging + cross-die penalty from a trace DB.

    Args:
        db_path: rocprofiler-sdk .db path.

    Returns:
        Dict::

            {
              "paging_events": int,          # HtoD/DtoH moves >= 1 MiB
              "cross_die_bytes": int,        # DtoD across agent ids
              "page_faults": int,            # kernel-visible page faults (0 if absent)
              "estimated_penalty_ns": int,   # cross-die access penalty estimate
              "recommendations": [str, ...],
            }

        Returns a zero-filled shape on missing tables (never raises).
    """
    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.Error:
        return {
            "paging_events": 0, "cross_die_bytes": 0, "page_faults": 0,
            "estimated_penalty_ns": 0,
            "recommendations": ["DB unreadable — skipping unified-memory analysis."],
        }

    try:
        memcpy_table = _find_memcpy_table(conn)
        pf_table = _find_page_fault_table(conn)

        if memcpy_table:
            paging_events, _total_bytes = _count_paging_memcpy(conn, memcpy_table)
            cross_die_bytes = _cross_die_bytes_estimate(conn, memcpy_table)
        else:
            paging_events, cross_die_bytes = 0, 0

        page_faults = _page_fault_count(conn, pf_table) if pf_table else 0

        # Penalty model: assume 64B-per-access fabric crossing.
        access_est = max(cross_die_bytes // 64, 0)
        estimated_penalty_ns = access_est * _MI300X_CROSS_DIE_NS_PER_ACCESS

        return {
            "paging_events": int(paging_events),
            "cross_die_bytes": int(cross_die_bytes),
            "page_faults": int(page_faults),
            "estimated_penalty_ns": int(estimated_penalty_ns),
            "recommendations": _recommendations(
                paging_events, cross_die_bytes, page_faults
            ),
        }
    finally:
        conn.close()


__all__ = ["analyze_paging"]
