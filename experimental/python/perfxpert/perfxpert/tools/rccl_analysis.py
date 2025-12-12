"""rccl_analysis — per-collective RCCL bus-bandwidth analysis.

Computes per-collective metrics (duration, bus BW, peak, efficiency,
overlap) + a summary from a rocpd SQLite database.

Tool class: READ_ONLY (MCP-safe).

Data sources
------------
Primary:  ``rocpd_region WHERE category='RCCL'`` JOINed with ``rocpd_arg``
          for the ``count`` and ``datatype`` args (so ``msg_bytes`` can be
          reconstructed). rocprofv3 >= 6.2 emits these natively and carries
          the opType in the region name (``AllReduce``, ``AllGather``, …).

Fallback: when no RCCL regions exist but RCCL kernels do fire, match kernel
          names against the TraceLens regex — this is the "capture
          incomplete" path. The result's summary carries
          ``capture_incomplete: true`` so formatters can surface a hint.

Formulas
--------
  busBW = msg_bytes * factor / duration_s
  factor = { AllReduce: 2(N-1)/N,
             AllGather / ReduceScatter / AllToAll: (N-1)/N,
             Broadcast / Reduce: 1 }
  efficiency_pct = busBW / peak_busBW * 100

Efficiency classification: <40% = poor, 40-70% = fair, >70% = good.
Regime: <1 MB = latency-bound, >16 MB = bandwidth-bound, else algo-dependent.

Overlap = fraction of RCCL-kernel interval that coincides with non-RCCL
kernel activity, computed via ``tracelens_port._merge_intervals`` +
``_subtract_intervals``.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from perfxpert.tools._class import ToolClass, tool_class
from perfxpert.tracelens_port import (
    _merge_intervals,
    _subtract_intervals,
    _total_ns,
    categorize_kernel_name,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# dtype size table — the ``datatype`` arg RCCL records is a small enum name.
# Keep this list conservative: anything unknown falls back to 4 bytes so a
# missing mapping never silently zeroes out the message size.
_DTYPE_BYTES: Dict[str, int] = {
    "int8": 1, "uint8": 1, "char": 1,
    "float16": 2, "half": 2, "bfloat16": 2, "bf16": 2, "fp16": 2, "int16": 2,
    "float32": 4, "float": 4, "fp32": 4, "int32": 4, "uint32": 4,
    "float64": 8, "double": 8, "fp64": 8, "int64": 8, "uint64": 8,
}

_OP_RE = re.compile(
    r"(AllReduce|AllGather|ReduceScatter|Broadcast|Reduce|AllToAll|SendRecv)",
    re.IGNORECASE,
)

# Kernel-name regex for the fallback path (matches TraceLens NCCL category).
_RCCL_KERNEL_RE = re.compile(
    r"ncclKernel|rccl|AllReduce|AllGather|ReduceScatter|Broadcast",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_op_type(name: str) -> str:
    """Map a region / kernel name to a canonical collective op type."""
    m = _OP_RE.search(name or "")
    if m:
        op = m.group(1)
        # Normalise capitalisation — "allreduce" → "AllReduce" etc.
        fixed = {
            "allreduce": "AllReduce",
            "allgather": "AllGather",
            "reducescatter": "ReduceScatter",
            "broadcast": "Broadcast",
            "reduce": "Reduce",
            "alltoall": "AllToAll",
            "sendrecv": "SendRecv",
        }
        return fixed.get(op.lower(), op)
    return "Unknown"


def _busbw_factor(op_type: str, n_ranks: int) -> float:
    """Ring busBW factor for RCCL collectives.

    Returns 0 when ``n_ranks <= 1`` (no meaningful ring) so callers surface
    busBW = 0 rather than division noise.
    """
    if n_ranks <= 1:
        return 0.0
    n = float(n_ranks)
    op = op_type.lower()
    if op == "allreduce":
        return 2.0 * (n - 1.0) / n
    if op in {"allgather", "reducescatter", "alltoall"}:
        return (n - 1.0) / n
    if op in {"broadcast", "reduce", "sendrecv"}:
        return 1.0
    return (n - 1.0) / n  # conservative default


def _regime(msg_bytes: int) -> str:
    """Classify an op as latency-bound / algo-dependent / bandwidth-bound."""
    if msg_bytes < 1 * 1024 * 1024:
        return "latency-bound"
    if msg_bytes > 16 * 1024 * 1024:
        return "bandwidth-bound"
    return "algo-dependent"


def _classify_efficiency(pct: float) -> str:
    if pct < 40.0:
        return "poor"
    if pct < 70.0:
        return "fair"
    return "good"


def _algo_hint(op_type: str, msg_bytes: int) -> str:
    """Best-effort textual algorithm hint (Ring vs Tree vs PXN)."""
    op = op_type.lower()
    if op == "allreduce":
        return "Ring" if msg_bytes >= 1 * 1024 * 1024 else "Tree"
    if op in {"allgather", "reducescatter"}:
        return "Ring"
    if op == "alltoall":
        return "Pairwise"
    return "direct"


def _topology_hint(n_ranks: int) -> str:
    if n_ranks <= 1:
        return "single-rank"
    if n_ranks <= 8:
        return "intra-node"
    return "multi-node"


def _lookup_peak(gfx_id: Optional[str]) -> Optional[float]:
    """Return the achievable XGMI busBW from interconnect_specs.yaml."""
    if not gfx_id:
        return None
    try:
        from perfxpert.tools.interconnect import lookup_peaks as _p
        entry = _p(gfx_id)
    except Exception:
        return None
    peak = entry.get("achievable_gbps") or entry.get("xgmi_peak_gbps")
    return float(peak) if peak else None


def _open_raw(db_path: str) -> sqlite3.Connection:
    """Open a rocpd .db read-only; caller closes."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _rccl_regions(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    """Fetch RCCL regions via the ``regions`` view. Empty list on any error."""
    try:
        cur = conn.execute(
            "SELECT id, name, start, end, pid, tid, event_id "
            "FROM regions WHERE category='RCCL'"
        )
        return cur.fetchall()
    except sqlite3.OperationalError:
        return []


def _args_for_events(
    conn: sqlite3.Connection, event_ids: List[int]
) -> Dict[int, Dict[str, str]]:
    """Return ``{event_id: {arg_name: arg_value}}`` for a list of event_ids.

    Uses parameter binding with chunking so very large event-id lists do
    not hit SQLite's variable-count limit.
    """
    if not event_ids:
        return {}
    out: Dict[int, Dict[str, str]] = {}
    CHUNK = 500
    for i in range(0, len(event_ids), CHUNK):
        chunk = event_ids[i:i + CHUNK]
        placeholders = ",".join(["?"] * len(chunk))
        try:
            cur = conn.execute(
                f"SELECT event_id, name, value FROM rocpd_arg "
                f"WHERE event_id IN ({placeholders})",
                chunk,
            )
            for row in cur.fetchall():
                eid = int(row["event_id"])
                out.setdefault(eid, {})[str(row["name"])] = str(row["value"])
        except sqlite3.OperationalError:
            # rocpd_arg absent or schema differs — return what we have.
            break
    return out


def _infer_msg_bytes(args: Dict[str, str]) -> int:
    """Reconstruct ``msg_bytes`` from (count, datatype) args when present.

    Falls back to 0 when either arg is missing or unparseable.
    """
    count_s = args.get("count") or args.get("sendcount") or args.get("recvcount")
    dtype_s = args.get("datatype") or args.get("dtype") or "float32"
    if not count_s:
        return 0
    try:
        count = int(count_s)
    except (TypeError, ValueError):
        try:
            count = int(float(count_s))
        except (TypeError, ValueError):
            return 0
    dt_bytes = _DTYPE_BYTES.get(dtype_s.lower().strip("'\""), 4)
    return max(0, count * dt_bytes)


def _count_ranks(conn: sqlite3.Connection) -> int:
    """Infer rank count from distinct (nid, pid) tuples in ``rocpd_region``."""
    try:
        cur = conn.execute(
            "SELECT COUNT(DISTINCT (nid * 1000000 + pid)) FROM rocpd_region "
            "WHERE name_id IN (SELECT id FROM rocpd_string)"
        )
        row = cur.fetchone()
        if row and row[0]:
            return int(row[0])
    except sqlite3.OperationalError:
        pass
    # Fallback — use processes table if present.
    try:
        cur = conn.execute("SELECT COUNT(DISTINCT pid) FROM processes")
        row = cur.fetchone()
        if row and row[0]:
            return int(row[0])
    except sqlite3.OperationalError:
        pass
    return 1


def _compute_overlap_ratio(conn: sqlite3.Connection) -> float:
    """Return fraction of RCCL-kernel time overlapping non-RCCL kernel time.

    Uses the ``kernels`` view's (start, end, name) triplet. Any failure →
    return 0.0 (missing data is surfaced, not errored).
    """
    try:
        rows = conn.execute(
            "SELECT name, start, end FROM kernels"
        ).fetchall()
    except sqlite3.OperationalError:
        return 0.0

    rccl: List[Tuple[int, int]] = []
    other: List[Tuple[int, int]] = []
    for r in rows:
        name = r[0] if not isinstance(r, sqlite3.Row) else r["name"]
        start = r[1] if not isinstance(r, sqlite3.Row) else r["start"]
        end = r[2] if not isinstance(r, sqlite3.Row) else r["end"]
        if name is None or start is None or end is None:
            continue
        iv = (int(start), int(end))
        if categorize_kernel_name(str(name)) == "NCCL":
            rccl.append(iv)
        else:
            other.append(iv)

    if not rccl:
        return 0.0
    merged_rccl = _merge_intervals(rccl)
    merged_other = _merge_intervals(other)
    rccl_total = _total_ns(merged_rccl)
    if rccl_total <= 0:
        return 0.0
    # time in RCCL that does NOT overlap other work = exposed comm
    exposed = _subtract_intervals(merged_rccl, merged_other)
    exposed_ns = _total_ns(exposed)
    overlapped = rccl_total - exposed_ns
    return round(100.0 * overlapped / rccl_total, 2)


def _fallback_from_kernels(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Regex-based fallback when no ``category='RCCL'`` rows exist.

    Returns a list with one synthetic entry per distinct RCCL kernel, with
    zeroed message size (we have no arg table binding) but real durations.
    """
    try:
        rows = conn.execute(
            "SELECT name, start, end FROM kernels"
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    out: List[Dict[str, Any]] = []
    for r in rows:
        name = str(r[0] if not isinstance(r, sqlite3.Row) else r["name"])
        if not _RCCL_KERNEL_RE.search(name):
            continue
        start = int(r[1] if not isinstance(r, sqlite3.Row) else r["start"])
        end = int(r[2] if not isinstance(r, sqlite3.Row) else r["end"])
        out.append({"op_type": _detect_op_type(name), "name": name,
                    "start": start, "end": end})
    return out


# ---------------------------------------------------------------------------
# Public tool
# ---------------------------------------------------------------------------

@tool_class(ToolClass.READ_ONLY)
def analyze_collectives(
    db_path: str,
    gfx_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Return per-collective metrics + a summary from a rocpd DB.

    Args:
        db_path: Path to a rocpd SQLite file (single-DB or pre-merged).
        gfx_id: Optional arch id (e.g. ``"gfx942"``) used to look up the
            peak busBW so efficiency percentages can be reported.

    Returns:
        ``CommunicationBlock``-compatible dict (see
        :class:`perfxpert.agents.schemas.CommunicationBlock`) —
        ``{"collectives": [...], "summary": {...}, "capture_incomplete":
        bool}`` where each collective entry carries the 12 fields typed
        by :class:`perfxpert.agents.schemas.CollectiveEntry`
        (``op_type``, ``msg_bytes``, ``duration_ns``,
        ``effective_bw_gbps``, ``peak_bw_gbps``, ``efficiency_pct``,
        ``efficiency_label``, ``overlap_ratio``, ``algo_hint``,
        ``topology_hint``, ``regime``, ``ranks``).

        When the DB contains no RCCL data at all the result is
        ``{"collectives": [], "summary": {"capture_incomplete": False,
        ...}, "capture_incomplete": False}``.

    Example:
        >>> from perfxpert.tools.rccl_analysis import analyze_collectives
        >>> res = analyze_collectives("merged.db", gfx_id="gfx942")
        >>> res["summary"]["op_count"]
        4
    """
    conn = _open_raw(db_path)
    try:
        regions = _rccl_regions(conn)
        n_ranks = _count_ranks(conn)
        peak = _lookup_peak(gfx_id)
        overlap_pct = _compute_overlap_ratio(conn)
        capture_incomplete = False

        collectives: List[Dict[str, Any]] = []

        if regions:
            event_ids = [int(r["event_id"]) for r in regions if r["event_id"] is not None]
            arg_map = _args_for_events(conn, event_ids)
            for r in regions:
                name = str(r["name"])
                op = _detect_op_type(name)
                start = int(r["start"])
                end = int(r["end"])
                duration_ns = max(0, end - start)
                args = arg_map.get(int(r["event_id"]), {}) if r["event_id"] is not None else {}
                msg_bytes = _infer_msg_bytes(args)
                factor = _busbw_factor(op, n_ranks)
                dur_s = duration_ns / 1e9 if duration_ns > 0 else 0.0
                bw_gbps = (msg_bytes * factor / dur_s / 1e9) if dur_s > 0 else 0.0
                eff_pct = (bw_gbps / peak * 100.0) if (peak and peak > 0) else 0.0
                collectives.append({
                    "op_type": op,
                    "msg_bytes": int(msg_bytes),
                    "duration_ns": int(duration_ns),
                    "effective_bw_gbps": round(bw_gbps, 3),
                    "peak_bw_gbps": round(peak, 3) if peak else None,
                    "efficiency_pct": round(eff_pct, 2),
                    "efficiency_label": _classify_efficiency(eff_pct),
                    "overlap_ratio": overlap_pct,
                    "algo_hint": _algo_hint(op, msg_bytes),
                    "topology_hint": _topology_hint(n_ranks),
                    "regime": _regime(msg_bytes),
                    "ranks": n_ranks,
                })
        else:
            # Fallback: kernel-name regex.
            kernel_hits = _fallback_from_kernels(conn)
            if kernel_hits:
                capture_incomplete = True
                for k in kernel_hits:
                    op = k["op_type"]
                    duration_ns = max(0, k["end"] - k["start"])
                    factor = _busbw_factor(op, n_ranks)
                    collectives.append({
                        "op_type": op,
                        "msg_bytes": 0,
                        "duration_ns": int(duration_ns),
                        "effective_bw_gbps": 0.0,
                        "peak_bw_gbps": round(peak, 3) if peak else None,
                        "efficiency_pct": 0.0,
                        "efficiency_label": "unknown",
                        "overlap_ratio": overlap_pct,
                        "algo_hint": _algo_hint(op, 0),
                        "topology_hint": _topology_hint(n_ranks),
                        "regime": _regime(0),
                        "ranks": n_ranks,
                    })
            # else: no RCCL signal at all — return the empty-shape response.

        summary = _build_summary(collectives, n_ranks, peak, overlap_pct,
                                 capture_incomplete)
        # Validate at the boundary via the Pydantic CommunicationBlock
        # model — downstream consumers (formatters, Latency specialist,
        # MCP clients) then get a single source of truth for field names
        # + types. ``model_dump()`` casts back to the dict shape the rest
        # of the codebase already consumes.
        from perfxpert.agents.schemas import CommunicationBlock
        block = CommunicationBlock(
            collectives=collectives,  # Pydantic validates each entry as CollectiveEntry
            summary=summary,
            capture_incomplete=capture_incomplete,
        )
        return block.model_dump()
    finally:
        conn.close()


def _build_summary(
    collectives: List[Dict[str, Any]],
    n_ranks: int,
    peak: Optional[float],
    overlap_pct: float,
    capture_incomplete: bool,
) -> Dict[str, Any]:
    """Compact per-run summary for formatters / the Latency Specialist."""
    if not collectives:
        return {
            "op_count": 0,
            "ranks": n_ranks,
            "dominant_op": None,
            "peak_bw_gbps": peak,
            "avg_bw_gbps": 0.0,
            "avg_efficiency_pct": 0.0,
            "overlap_pct": overlap_pct,
            "capture_incomplete": capture_incomplete,
        }

    # Dominant op by total time.
    by_op: Dict[str, Dict[str, float]] = {}
    total_bw = 0.0
    total_eff = 0.0
    for c in collectives:
        op = c["op_type"]
        entry = by_op.setdefault(op, {"count": 0, "time_ns": 0})
        entry["count"] += 1
        entry["time_ns"] += c["duration_ns"]
        total_bw += c["effective_bw_gbps"]
        total_eff += c["efficiency_pct"]
    dominant = max(by_op.items(), key=lambda kv: kv[1]["time_ns"])[0]
    n = len(collectives)
    return {
        "op_count": n,
        "ranks": n_ranks,
        "dominant_op": dominant,
        "peak_bw_gbps": round(peak, 3) if peak else None,
        "avg_bw_gbps": round(total_bw / n, 3),
        "avg_efficiency_pct": round(total_eff / n, 2),
        "overlap_pct": overlap_pct,
        "capture_incomplete": capture_incomplete,
    }


__all__ = ["analyze_collectives"]
