"""metrics — HW counter arithmetic primitives.

Pure-arithmetic derivations from raw rocprofv3 hardware counters. Replaces
LLM arithmetic (which hallucinates) with deterministic functions.

Counter sources (rocprofv3):
  - GRBM_GUI_ACTIVE, GRBM_COUNT  → GPU utilization
  - TCP_TCC_READ_REQ, TCP_*_READ → L1 miss rate (reads missing to L2)
  - TCC_HIT, TCC_MISS            → L2 hit rate
  - FETCH_SIZE, WRITE_SIZE       → HBM bandwidth (KiB units from rocprofv3)
  - SQ_ACCUM_PREV_HIRES, SQ_INSTS_VMEM, SQ_INSTS_LDS, SQ_BUSY_CYCLES → latency

Tool class: READ_ONLY.
"""

from typing import Any, Dict

from perfxpert.tools._class import ToolClass, tool_class


# Bytes per KiB (binary)
_BYTES_PER_KIB = 1024
# Bytes per GiB (binary)
_BYTES_PER_GIB = 1024 ** 3
# Bytes per GB (decimal, SI) — for bandwidth reporting against vendor specs
_BYTES_PER_GB = 1_000_000_000
# Nanoseconds per second
_NS_PER_S = 1_000_000_000


@tool_class(ToolClass.READ_ONLY)
def compute_gpu_utilization(grbm_gui_active: float, grbm_count: float) -> float:
    """Fraction of total cycles the GPU command processor was busy.

    Args:
        grbm_gui_active: GRBM_GUI_ACTIVE counter (cycles GUI was active).
        grbm_count: GRBM_COUNT counter (total reference cycles).

    Returns:
        Utilization fraction in [0.0, 1.0].

    Raises:
        ValueError: if grbm_count == 0 (divide-by-zero).
    """
    if grbm_count == 0:
        raise ValueError("grbm_count must be > 0 — divide-by-zero in GPU utilization")
    return grbm_gui_active / grbm_count


@tool_class(ToolClass.READ_ONLY)
def compute_l1_miss_rate(tcp_tcc_read_req: float, tcp_total_read: float) -> float:
    """Fraction of L1 (TCP) read requests that missed to L2 (TCC).

    Args:
        tcp_tcc_read_req: TCP→TCC read requests (L1 misses that went to L2).
        tcp_total_read: Total TCP read requests.

    Returns:
        Miss rate fraction in [0.0, 1.0].

    Raises:
        ValueError: if tcp_total_read == 0.
    """
    if tcp_total_read == 0:
        raise ValueError("tcp_total_read must be > 0 — divide-by-zero in L1 miss rate")
    return tcp_tcc_read_req / tcp_total_read


@tool_class(ToolClass.READ_ONLY)
def compute_l2_hit_rate(tcc_hit: float, tcc_miss: float) -> float:
    """Fraction of L2 (TCC) accesses that hit.

    Args:
        tcc_hit: L2 hit counter.
        tcc_miss: L2 miss counter.

    Returns:
        Hit rate fraction in [0.0, 1.0].

    Raises:
        ValueError: if tcc_hit + tcc_miss == 0 (no L2 activity).
    """
    total = tcc_hit + tcc_miss
    if total == 0:
        raise ValueError("tcc_hit + tcc_miss must be > 0 — no L2 activity to measure")
    return tcc_hit / total


@tool_class(ToolClass.READ_ONLY)
def compute_hbm_bandwidth(
    fetch_kib: float,
    write_kib: float,
    duration_ns: float,
) -> Dict[str, float]:
    """HBM bandwidth in GB/s (decimal) and GiB/s (binary) from FETCH_SIZE/WRITE_SIZE.

    rocprofv3 reports FETCH_SIZE and WRITE_SIZE in KiB (1024 bytes). This
    function returns both decimal (GB/s, vs vendor specs like MI300X 5.3 TB/s)
    and binary (GiB/s) forms so callers can choose the convention that matches
    their comparison baseline.

    Args:
        fetch_kib: FETCH_SIZE counter in KiB (read bytes from HBM).
        write_kib: WRITE_SIZE counter in KiB (write bytes to HBM).
        duration_ns: kernel duration in nanoseconds.

    Returns:
        {"gb_per_s": float, "gib_per_s": float}

    Raises:
        ValueError: if duration_ns == 0.
    """
    if duration_ns == 0:
        raise ValueError("duration_ns must be > 0 — divide-by-zero in HBM bandwidth")

    total_bytes = (fetch_kib + write_kib) * _BYTES_PER_KIB
    duration_s = duration_ns / _NS_PER_S
    bytes_per_s = total_bytes / duration_s

    return {
        "gb_per_s": bytes_per_s / _BYTES_PER_GB,
        "gib_per_s": bytes_per_s / _BYTES_PER_GIB,
    }


@tool_class(ToolClass.READ_ONLY)
def compute_latency(
    accum_prev_hires: float,
    insts_vmem: float,
    insts_lds: float,
    busy_cycles: float,
) -> Dict[str, Any]:
    """Average stall latency per VMEM / LDS instruction + busy ratio.

    Per rocprofv3 `SQ_ACCUM_PREV_HIRES` semantics, `accum_prev_hires` is the
    accumulated wait-state cycles on the previous hi-res counter (typically
    VMEM + LDS latency proxies). We approximate per-instruction latency by
    dividing by instruction counts.

    Args:
        accum_prev_hires: SQ_ACCUM_PREV_HIRES (total stall/wait cycles).
        insts_vmem: SQ_INSTS_VMEM (VMEM instruction count).
        insts_lds: SQ_INSTS_LDS (LDS instruction count).
        busy_cycles: SQ_BUSY_CYCLES (cycles wave was busy).

    Returns:
        {
            "vmem_latency_cycles": accum / insts_vmem (or 0.0 if insts_vmem == 0),
            "lds_latency_cycles":  accum / insts_lds  (or 0.0 if insts_lds == 0),
            "busy_ratio":          busy_cycles / (busy_cycles + accum_prev_hires),
                                    or 0.0 if denominator == 0
        }

    Note:
        Zero-instruction branches return 0.0 rather than raising — callers
        iterating over many kernels may legitimately have kernels with no
        VMEM or no LDS, and raising would be noisy.
    """
    vmem_lat = accum_prev_hires / insts_vmem if insts_vmem > 0 else 0.0
    lds_lat = accum_prev_hires / insts_lds if insts_lds > 0 else 0.0
    busy_total = busy_cycles + accum_prev_hires
    busy_ratio = busy_cycles / busy_total if busy_total > 0 else 0.0

    return {
        "vmem_latency_cycles": vmem_lat,
        "lds_latency_cycles": lds_lat,
        "busy_ratio": busy_ratio,
    }
