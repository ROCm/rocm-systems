###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
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
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
###############################################################################

"""
Core analysis functions: time breakdown, hotspot identification, memory copy
analysis, and hardware counter analysis.
"""

import sys
from typing import Any, Dict, List, Optional

from ..connection import PerfxpertConnection as RocpdImportData, execute_statement
from ..tools import arch as arch_tools


def compute_time_breakdown(connection: RocpdImportData) -> Dict[str, Any]:
    """
    Calculate time distribution across kernel execution, memory copies, and overhead.

    Args:
        connection: RocpdImportData database connection

    Returns:
        Dictionary with time breakdown metrics including percentages
    """
    query = """
    WITH kernel_time AS (
        SELECT COALESCE(SUM(duration), 0) as total_kernel_time
        FROM kernels
    ),
    memcpy_time AS (
        SELECT COALESCE(SUM(duration), 0) as total_memcpy_time
        FROM memory_copies
    ),
    total_time AS (
        SELECT MAX(end) - MIN(start) as total_runtime
        FROM (
            SELECT start, end FROM kernels
            UNION ALL
            SELECT start, end FROM memory_copies
        )
    )
    SELECT
        k.total_kernel_time,
        m.total_memcpy_time,
        COALESCE(t.total_runtime, 0) as total_runtime,
        CASE WHEN COALESCE(t.total_runtime, 0) > 0
             THEN (k.total_kernel_time * 100.0 / t.total_runtime)
             ELSE 0 END as kernel_percent,
        CASE WHEN COALESCE(t.total_runtime, 0) > 0
             THEN (m.total_memcpy_time * 100.0 / t.total_runtime)
             ELSE 0 END as memcpy_percent,
        CASE WHEN COALESCE(t.total_runtime, 0) > 0
             THEN ((t.total_runtime - k.total_kernel_time - m.total_memcpy_time) * 100.0 / t.total_runtime)
             ELSE 0 END as overhead_percent
    FROM kernel_time k, memcpy_time m, total_time t
    """

    try:
        result = execute_statement(connection, query).fetchone()

        if not result:
            return {
                "total_kernel_time": 0,
                "total_memcpy_time": 0,
                "total_runtime": 0,
                "kernel_percent": 0,
                "memcpy_percent": 0,
                "overhead_percent": 0,
            }

        breakdown = {
            "total_kernel_time": result[0] or 0,
            "total_memcpy_time": result[1] or 0,
            "total_runtime": result[2] or 0,
            "kernel_percent": result[3] or 0,
            "memcpy_percent": result[4] or 0,
            "overhead_percent": max(0.0, result[5] or 0),
        }

        if getattr(connection, "db_count", 1) > 1:
            normalized_runtime = connection.sum_shard_runtime_envelopes(
                ["kernels", "memory_copies"]
            )
            if normalized_runtime > 0:
                breakdown["normalized_runtime"] = normalized_runtime
                breakdown["kernel_percent"] = (
                    breakdown["total_kernel_time"] * 100.0 / normalized_runtime
                )
                breakdown["memcpy_percent"] = (
                    breakdown["total_memcpy_time"] * 100.0 / normalized_runtime
                )
                breakdown["overhead_percent"] = max(
                    0.0,
                    (
                        normalized_runtime
                        - breakdown["total_kernel_time"]
                        - breakdown["total_memcpy_time"]
                    )
                    * 100.0
                    / normalized_runtime,
                )

        return breakdown

    except Exception as e:
        print(f"Warning: Could not compute time breakdown: {e}", file=sys.stderr)
        return {
            "error": str(e),
            "total_kernel_time": 0,
            "total_memcpy_time": 0,
            "total_runtime": 0,
            "kernel_percent": 0,
            "memcpy_percent": 0,
            "overhead_percent": 0,
        }


def identify_hotspots(
    connection: RocpdImportData, top_n: int = 10, min_duration: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Identify top N kernels by total execution time.

    Args:
        connection: RocpdImportData database connection
        top_n: Number of top kernels to return
        min_duration: Minimum duration threshold in nanoseconds

    Returns:
        List of dictionaries containing kernel statistics
    """
    # Build query with string formatting to avoid parameter binding issues
    # Use f-strings for both min_duration and top_n to avoid SQLite datatype issues
    if min_duration > 0:
        query = f"""
        SELECT
            name,
            COUNT(*) as calls,
            SUM(duration) as total_duration,
            AVG(duration) as avg_duration,
            MIN(duration) as min_duration,
            MAX(duration) as max_duration,
            (SUM(duration) * 100.0 / NULLIF((SELECT SUM(duration) FROM kernels), 0)) as percent_of_total
        FROM kernels
        WHERE duration >= {int(min_duration)}
        GROUP BY name
        ORDER BY total_duration DESC
        LIMIT {int(top_n)}
        """
    else:
        query = f"""
        SELECT
            name,
            COUNT(*) as calls,
            SUM(duration) as total_duration,
            AVG(duration) as avg_duration,
            MIN(duration) as min_duration,
            MAX(duration) as max_duration,
            (SUM(duration) * 100.0 / NULLIF((SELECT SUM(duration) FROM kernels), 0)) as percent_of_total
        FROM kernels
        GROUP BY name
        ORDER BY total_duration DESC
        LIMIT {int(top_n)}
        """

    try:
        # No parameters needed with string formatting
        results = execute_statement(connection, query, ()).fetchall()

        hotspots = []
        for row in results:
            hotspots.append(
                {
                    "name": row[0],
                    "calls": row[1],
                    "total_duration": row[2],
                    "avg_duration": row[3],
                    "min_duration": row[4],
                    "max_duration": row[5],
                    "percent_of_total": row[6] or 0,
                }
            )

        return hotspots

    except Exception as e:
        print(f"Warning: Could not identify hotspots: {e}", file=sys.stderr)
        return []


def analyze_memory_copies(connection: RocpdImportData) -> Dict[str, Dict[str, Any]]:
    """
    Analyze memory copy operations by direction and calculate bandwidth.

    Args:
        connection: RocpdImportData database connection

    Returns:
        Dictionary keyed by direction containing memory copy statistics
    """
    query = """
    SELECT
        CASE
            WHEN category LIKE '%HostToDevice%' OR category LIKE '%H2D%' THEN 'Host-to-Device'
            WHEN category LIKE '%DeviceToHost%' OR category LIKE '%D2H%' THEN 'Device-to-Host'
            WHEN category LIKE '%DeviceToDevice%' OR category LIKE '%D2D%' THEN 'Device-to-Device'
            ELSE category
        END as direction,
        COUNT(*) as count,
        COALESCE(SUM(CAST(size AS INTEGER)), 0) as total_bytes,
        SUM(end - start) as total_duration,
        COALESCE(AVG(CAST(size AS INTEGER)), 0.0) as avg_bytes,
        AVG(end - start) as avg_duration,
        CASE WHEN SUM(end - start) > 0
             THEN (COALESCE(SUM(CAST(size AS INTEGER)), 0) * 1.0e9 / SUM(end - start))
             ELSE 0.0
        END as bandwidth_bytes_per_sec
    FROM memory_copies
    WHERE category IS NOT NULL
    GROUP BY direction
    ORDER BY total_duration DESC
    """

    try:
        results = execute_statement(connection, query).fetchall()

        analysis = {}
        for row in results:
            direction = row[0]
            analysis[direction] = {
                "count": row[1],
                "total_bytes": row[2],
                "total_duration": row[3],
                "avg_bytes": row[4],
                "avg_duration": row[5],
                "bandwidth_bytes_per_sec": row[6],
            }

        return analysis

    except Exception as e:
        print(f"Warning: Could not analyze memory copies: {e}", file=sys.stderr)
        return {}


def analyze_hardware_counters(connection: RocpdImportData) -> Dict[str, Any]:
    """
    Analyze hardware performance counters (Tier 2 analysis).

    Args:
        connection: RocpdImportData database connection

    Returns:
        Dictionary containing hardware counter analysis:
        - has_counters: bool indicating if counter data exists
        - counters: dict of counter statistics by name
        - metrics: derived metrics (occupancy, utilization, etc.)
        - per_kernel: counter analysis by kernel name
    """
    try:
        # Check if pmc_events table exists by trying to query it
        check_query = "SELECT COUNT(*) FROM pmc_events LIMIT 1"
        result = execute_statement(connection, check_query, ()).fetchone()
        if not result or result[0] == 0:
            return {
                "has_counters": False,
                "reason": "pmc_events table exists but contains no data",
            }

        # Get available counters
        counters_query = """
        SELECT
            counter_name,
            COUNT(*) as sample_count,
            AVG(counter_value) as avg_value,
            MIN(counter_value) as min_value,
            MAX(counter_value) as max_value,
            SUM(counter_value) as total_value
        FROM pmc_events
        GROUP BY counter_name
        ORDER BY counter_name
        """

        counter_results = execute_statement(connection, counters_query, ()).fetchall()

        counters = {}
        for row in counter_results:
            counters[row[0]] = {
                "sample_count": row[1],
                "avg_value": row[2],
                "min_value": row[3],
                "max_value": row[4],
                "total_value": row[5],
            }

        # Get per-kernel counter analysis
        per_kernel_query = """
        SELECT
            name,
            counter_name,
            COUNT(DISTINCT dispatch_id) as dispatch_count,
            AVG(counter_value) as avg_value,
            MIN(counter_value) as min_value,
            MAX(counter_value) as max_value
        FROM pmc_events
        GROUP BY name, counter_name
        ORDER BY name, counter_name
        LIMIT 5000
        """

        kernel_results = execute_statement(connection, per_kernel_query, ()).fetchall()

        per_kernel = {}
        for row in kernel_results:
            kernel_name = row[0]
            counter_name = row[1]

            if kernel_name not in per_kernel:
                per_kernel[kernel_name] = {}

            per_kernel[kernel_name][counter_name] = {
                "dispatch_count": row[2],
                "avg_value": row[3],
                "min_value": row[4],
                "max_value": row[5],
            }

        # Calculate derived metrics
        metrics = {}

        # GPU Utilization (GRBM_GUI_ACTIVE / GRBM_COUNT)
        if "GRBM_GUI_ACTIVE" in counters and "GRBM_COUNT" in counters:
            grbm_count = counters["GRBM_COUNT"]["avg_value"]
            grbm_active = counters["GRBM_GUI_ACTIVE"]["avg_value"]
            if grbm_count > 0:
                metrics["gpu_utilization_percent"] = (grbm_active / grbm_count) * 100

        # Average wave occupancy
        if "SQ_WAVES" in counters:
            metrics["avg_waves"] = counters["SQ_WAVES"]["avg_value"]
            metrics["max_waves"] = counters["SQ_WAVES"]["max_value"]
            metrics["min_waves"] = counters["SQ_WAVES"]["min_value"]

        return {
            "has_counters": True,
            "counters": counters,
            "metrics": metrics,
            "per_kernel": per_kernel,
        }

    except Exception as e:
        print(f"Warning: Could not analyze hardware counters: {e}", file=sys.stderr)
        return {"has_counters": False, "reason": str(e)}


def detect_warmup_issues(
    connection: RocpdImportData,
    hotspots: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Detect first-dispatch latency outliers suggesting missing warmup.

    Compares first dispatch duration of each hotspot kernel to its average.
    Returns warmup issue info if any kernel's first dispatch is >2x average.

    Args:
        connection: RocpdImportData database connection
        hotspots: Top kernel hotspots from identify_hotspots()

    Returns:
        Dictionary with ``has_warmup_issues`` bool and ``outliers`` list.
    """
    outliers: List[Dict[str, Any]] = []
    for kernel in hotspots[:5]:  # Top 5 only to limit queries
        kernel_name = kernel.get("name", "")
        avg_duration = kernel.get("avg_duration", 0)
        calls = kernel.get("calls", 0)
        if not kernel_name or calls <= 1 or avg_duration <= 0:
            continue
        safe_name = kernel_name.replace("'", "''")
        query = f"SELECT duration FROM kernels WHERE name = '{safe_name}' ORDER BY start ASC LIMIT 1"
        try:
            row = execute_statement(connection, query).fetchone()
            if not row:
                continue
            first_duration = row[0] or 0
            if first_duration > 0 and avg_duration > 0:
                ratio = first_duration / avg_duration
                if ratio > 2.0:
                    outliers.append({
                        "kernel_name": kernel_name,
                        "first_duration_ns": first_duration,
                        "avg_duration_ns": avg_duration,
                        "ratio": ratio,
                    })
        except Exception:
            continue
    return {"has_warmup_issues": len(outliers) > 0, "outliers": outliers}


# ---------------------------------------------------------------------------
# Architecture specs for occupancy calculation (ROCM-21553 I1)
# ---------------------------------------------------------------------------
#
# Keep one source of truth: ``knowledge/gpu_specs.yaml`` carries both the
# user-facing peak specs and the occupancy-only caps consumed here.

_ARCH_SPECS: Dict[str, Dict[str, Any]] = arch_tools.occupancy_specs_table()


def analyze_kernel_resources(
    connection: RocpdImportData,
    hotspots: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Extract kernel resource usage (VGPR, SGPR, LDS, scratch) and compute
    theoretical occupancy for top hotspot kernels.

    Queries the ``kernels`` view which JOINs ``rocpd_kernel_dispatch`` with
    ``rocpd_info_kernel_symbol``, exposing register counts, LDS/scratch sizes,
    and launch configuration (block/grid dimensions).

    Architecture is detected from ``rocpd_info_agent`` to look up hardware
    limits for occupancy calculation.

    Returns:
        Dictionary with ``arch``, ``arch_specs``, and ``kernels`` list.
    """
    import math
    import re as _re

    # Detect architecture
    arch = None
    arch_specs = None
    try:
        agent_query = "SELECT name FROM rocpd_info_agent WHERE type='GPU' LIMIT 1"
        row = execute_statement(connection, agent_query).fetchone()
        if row and row[0]:
            m = _re.search(r"gfx\d+", row[0])
            if m:
                arch = m.group(0)
                arch_specs = _ARCH_SPECS.get(arch)
    except Exception:
        pass

    kernel_list: List[Dict[str, Any]] = []
    for kernel in hotspots[:10]:
        kernel_name = kernel.get("name", "")
        if not kernel_name:
            continue
        safe_name = kernel_name.replace("'", "''")
        query = (
            f"SELECT DISTINCT name, vgpr_count, accum_vgpr_count, sgpr_count, "
            f"lds_size, scratch_size, workgroup_x, workgroup_y, workgroup_z, "
            f"grid_x, grid_y, grid_z "
            f"FROM kernels WHERE name = '{safe_name}' LIMIT 1"
        )
        try:
            row = execute_statement(connection, query).fetchone()
            if not row:
                continue
        except Exception:
            continue

        vgpr = row[1] or 0
        accum_vgpr = row[2] or 0
        sgpr = row[3] or 0
        lds = row[4] or 0
        scratch = row[5] or 0
        wg_x, wg_y, wg_z = row[6] or 1, row[7] or 1, row[8] or 1
        grid_x, grid_y, grid_z = row[9] or 1, row[10] or 1, row[11] or 1

        entry: Dict[str, Any] = {
            "name": kernel_name,
            "vgpr": vgpr,
            "accum_vgpr": accum_vgpr,
            "sgpr": sgpr,
            "lds_bytes": lds,
            "scratch_bytes": scratch,
            "block": f"{wg_x}x{wg_y}x{wg_z}",
            "grid": f"{grid_x}x{grid_y}x{grid_z}",
        }

        # Compute occupancy if architecture known
        if arch_specs and vgpr > 0:
            max_w = arch_specs["max_waves_per_simd"]
            vgprs_per_simd = arch_specs["vgprs_per_simd"]
            lds_per_cu = arch_specs["lds_per_cu_kb"] * 1024
            wf_size = arch_specs["wavefront_size"]
            simds = arch_specs["simds_per_cu"]

            vgpr_limited = min(math.floor(vgprs_per_simd / max(vgpr, 1)), max_w)

            threads_per_block = wg_x * wg_y * wg_z
            waves_per_block = math.ceil(threads_per_block / wf_size)
            max_threads_per_cu = max_w * simds * wf_size
            max_blocks = math.floor(max_threads_per_cu / max(threads_per_block, 1))
            block_waves = max_blocks * waves_per_block
            block_limited = min(math.floor(block_waves / simds), max_w)

            if lds > 0:
                lds_blocks = math.floor(lds_per_cu / lds)
                lds_limited = min(math.floor(lds_blocks * waves_per_block / simds), max_w)
            else:
                lds_limited = max_w

            achieved = min(vgpr_limited, lds_limited, block_limited)
            occ_pct = achieved / max_w * 100.0

            limiting = "none"
            if achieved == vgpr_limited and achieved < max_w:
                limiting = "VGPR"
            elif achieved == lds_limited and achieved < max_w:
                limiting = "LDS"
            elif achieved == block_limited and achieved < max_w:
                limiting = "block size"

            entry["occupancy"] = {
                "waves_per_simd": achieved,
                "max_waves_per_simd": max_w,
                "percent": round(occ_pct, 1),
                "limiting_resource": limiting,
                "vgpr_limited": vgpr_limited,
                "lds_limited": lds_limited,
                "block_limited": block_limited,
            }

        kernel_list.append(entry)

    return {"arch": arch, "arch_specs": arch_specs, "kernels": kernel_list}


def analyze_api_overhead(connection: RocpdImportData) -> Dict[str, Any]:
    """
    Break down HIP/HSA API overhead by individual API call.

    Queries the ``regions`` SQL view for per-API-call duration aggregates
    so recommendations can distinguish kernel launch overhead from setup
    overhead (e.g., hipSetDevice, hipMalloc).

    Returns:
        Dictionary with ``api_calls`` list (sorted by total time DESC),
        ``launch_overhead_ns``, ``total_api_ns``, and ``has_api_data`` flag.
        Gracefully returns empty structure if ``regions`` view is unavailable.
    """
    query = """
    SELECT name, COUNT(*) as calls, SUM(duration) as total_ns, AVG(duration) as avg_ns
    FROM regions
    WHERE category IN ('HIP_RUNTIME_API_EXT', 'HIP_COMPILER_API_EXT')
    GROUP BY name
    ORDER BY total_ns DESC
    """
    try:
        rows = execute_statement(connection, query).fetchall()
        if not rows:
            return {"api_calls": [], "launch_overhead_ns": 0, "total_api_ns": 0, "has_api_data": False}

        api_calls: List[Dict[str, Any]] = []
        launch_overhead_ns = 0
        total_api_ns = 0

        for row in rows:
            name, calls, total_ns, avg_ns = row[0], row[1], row[2] or 0, row[3] or 0
            api_calls.append({
                "name": name,
                "calls": calls,
                "total_ns": total_ns,
                "avg_ns": avg_ns,
            })
            total_api_ns += total_ns
            if name and "hipLaunchKernel" in name:
                launch_overhead_ns += total_ns

        return {
            "api_calls": api_calls,
            "launch_overhead_ns": launch_overhead_ns,
            "total_api_ns": total_api_ns,
            "has_api_data": True,
        }
    except Exception:
        # Graceful fallback if regions view doesn't exist in older DBs
        return {"api_calls": [], "launch_overhead_ns": 0, "total_api_ns": 0, "has_api_data": False}
