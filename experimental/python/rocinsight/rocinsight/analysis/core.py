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
from typing import Any, Dict, List

from ..connection import RocinsightConnection as RocpdImportData, execute_statement


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

        return {
            "total_kernel_time": result[0] or 0,
            "total_memcpy_time": result[1] or 0,
            "total_runtime": result[2] or 0,
            "kernel_percent": result[3] or 0,
            "memcpy_percent": result[4] or 0,
            "overhead_percent": max(0.0, result[5] or 0),
        }

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
