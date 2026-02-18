#!/usr/bin/env python3
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
AI-powered performance analysis for GPU traces.

This module analyzes rocpd database files and provides human-readable insights,
bottleneck identification, and optimization recommendations.
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

from .importer import RocpdImportData, execute_statement
from . import output_config

__all__ = [
    "compute_time_breakdown",
    "identify_hotspots",
    "analyze_memory_copies",
    "analyze_hardware_counters",
    "generate_recommendations",
    "format_analysis_output",
    "analyze_performance",
    "add_args",
    "execute",
    "main",
]


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
            "overhead_percent": result[5] or 0,
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
        SUM(CAST(0 AS INTEGER)) as total_bytes,
        SUM(end - start) as total_duration,
        AVG(CAST(0 AS INTEGER)) as avg_bytes,
        AVG(end - start) as avg_duration,
        CAST(0 AS REAL) as bandwidth_bytes_per_sec
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
            return {"has_counters": False}

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
        return {"has_counters": False}


def generate_recommendations(
    time_breakdown: Dict[str, Any],
    hotspots: List[Dict[str, Any]],
    memory_analysis: Dict[str, Dict[str, Any]],
    hardware_counters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Generate performance recommendations based on analysis results.

    Args:
        time_breakdown: Time distribution metrics
        hotspots: Top kernel hotspots
        memory_analysis: Memory copy analysis
        hardware_counters: Hardware counter analysis (Tier 2)

    Returns:
        List of recommendation dictionaries with priority, issue, and suggestions
    """
    recommendations = []

    # Tier 2: Hardware counter-based recommendations
    if hardware_counters and hardware_counters.get("has_counters"):
        metrics = hardware_counters.get("metrics", {})

        # Low wave occupancy
        avg_waves = metrics.get("avg_waves", 0)
        if avg_waves > 0 and avg_waves < 16:
            recommendations.append(
                {
                    "priority": "HIGH",
                    "category": "Low Occupancy",
                    "issue": f"Low wave occupancy detected: average {avg_waves:.1f} waves per SIMD",
                    "suggestion": "Increase kernel occupancy to improve GPU utilization",
                    "actions": [
                        "Increase block/workgroup size to launch more waves per CU",
                        "Reduce register usage per thread (check with --save-temps or rocm-llvm-mc)",
                        "Reduce shared memory (LDS) usage per workgroup",
                        "Check for resource limitations preventing more waves with rocprof-compute",
                    ],
                    "estimated_impact": "10-30% throughput improvement depending on occupancy gap",
                    "commands": [
                        {
                            "tool": "rocprofv3",
                            "description": "Collect wave occupancy and cycle counters per kernel dispatch",
                            "flags": ["--sys-trace"],
                            "args": [
                                {"name": "--pmc", "value": "SQ_WAVES SQ_WAVE_CYCLES TA_TA_BUSY"},
                                {"name": "-d", "value": "./occupancy_output"},
                                {"name": "-o", "value": "profile"},
                            ],
                            "full_command": "rocprofv3 --sys-trace --pmc SQ_WAVES SQ_WAVE_CYCLES TA_TA_BUSY -d ./occupancy_output -o profile -- ./app",
                        },
                        {
                            "tool": "rocprof-compute",
                            "description": "Deep-dive occupancy analysis: theoretical vs achieved waves per CU",
                            "flags": [],
                            "args": [
                                {"name": "profile", "value": None},
                                {"name": "--block", "value": "SQ"},
                            ],
                            "full_command": "rocprof-compute profile --block SQ -- ./app",
                        },
                    ],
                }
            )

        # Low GPU utilization
        gpu_util = metrics.get("gpu_utilization_percent", 0)
        if gpu_util > 0 and gpu_util < 70:
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "category": "GPU Utilization",
                    "issue": f"GPU utilization is only {gpu_util:.1f}% (target: >70%)",
                    "suggestion": "Reduce GPU idle time by overlapping work and eliminating synchronization gaps",
                    "actions": [
                        "Launch independent kernels concurrently using hipStreams",
                        "Increase kernel grid size to fill all CUs when problem size allows",
                        "Reduce hipDeviceSynchronize() and hipStreamSynchronize() call frequency",
                        "Overlap host-device transfers with compute using async streams",
                    ],
                    "estimated_impact": f"Up to {100 - gpu_util:.0f}% reduction in idle time",
                    "commands": [
                        {
                            "tool": "rocprofv3",
                            "description": "Collect GPU active vs total cycle counters to confirm utilization",
                            "flags": ["--sys-trace"],
                            "args": [
                                {"name": "--pmc", "value": "GRBM_GUI_ACTIVE GRBM_COUNT"},
                                {"name": "-d", "value": "./utilization_output"},
                                {"name": "-o", "value": "profile"},
                            ],
                            "full_command": "rocprofv3 --sys-trace --pmc GRBM_GUI_ACTIVE GRBM_COUNT -d ./utilization_output -o profile -- ./app",
                        },
                        {
                            "tool": "rocprof-sys",
                            "description": "System-level timeline: identify host/GPU idle gaps and synchronization stalls",
                            "flags": ["--trace"],
                            "args": [],
                            "full_command": "rocprof-sys --trace -- ./app",
                        },
                    ],
                }
            )

    # Tier 1: Trace-level recommendations

    # Rule 1: High memory copy overhead
    memcpy_percent = time_breakdown.get("memcpy_percent", 0)
    if memcpy_percent > 20:
        recommendations.append(
            {
                "priority": "HIGH",
                "category": "Memory Transfer",
                "issue": f"Memory copies consume {memcpy_percent:.1f}% of execution time",
                "suggestion": "Reduce host-device transfer overhead by batching and overlapping transfers",
                "actions": [
                    "Batch multiple small hipMemcpy calls into one large transfer",
                    "Allocate pinned host memory with hipHostMalloc for faster PCIe transfers",
                    "Use hipMemcpyAsync with streams to overlap transfers with kernel execution",
                    "Minimize round-trips: keep data on GPU between consecutive kernels",
                ],
                "estimated_impact": "15-30% reduction in total runtime when transfers dominate",
                "commands": [
                    {
                        "tool": "rocprofv3",
                        "description": "Trace HIP and HSA memory copy operations with timing",
                        "flags": ["--sys-trace", "--hsa-trace"],
                        "args": [
                            {"name": "-d", "value": "./memcpy_output"},
                            {"name": "-o", "value": "profile"},
                        ],
                        "full_command": "rocprofv3 --sys-trace --hsa-trace -d ./memcpy_output -o profile -- ./app",
                    },
                    {
                        "tool": "rocprof-sys",
                        "description": "Detailed memory transfer timeline with PCIe bandwidth and overlap analysis",
                        "flags": [],
                        "args": [
                            {"name": "--trace-gpu-memory", "value": None},
                        ],
                        "full_command": "rocprof-sys --trace-gpu-memory -- ./app",
                    },
                ],
            }
        )

    # Rule 2: High API overhead
    overhead_percent = time_breakdown.get("overhead_percent", 0)
    if overhead_percent > 15:
        recommendations.append(
            {
                "priority": "MEDIUM",
                "category": "API Overhead",
                "issue": f"API and launch overhead is {overhead_percent:.1f}% of total time",
                "suggestion": "Reduce the number of API calls and kernel launches",
                "actions": [
                    "Fuse multiple small kernels into fewer larger launches",
                    "Replace repeated hipMalloc/hipFree with a pre-allocated memory pool",
                    "Batch hipMemcpy calls; use hipMemcpyAsync where possible",
                    "Minimize hipDeviceSynchronize() — synchronize at stream level instead",
                ],
                "estimated_impact": "5-15% reduction when overhead exceeds 15%",
                "commands": [
                    {
                        "tool": "rocprofv3",
                        "description": "Trace all HIP runtime API calls to identify highest-frequency calls",
                        "flags": ["--hip-api-trace", "--hsa-trace"],
                        "args": [
                            {"name": "-d", "value": "./api_output"},
                            {"name": "-o", "value": "profile"},
                        ],
                        "full_command": "rocprofv3 --hip-api-trace --hsa-trace -d ./api_output -o profile -- ./app",
                    },
                    {
                        "tool": "rocprof-sys",
                        "description": "System-level API call frequency and per-call latency breakdown",
                        "flags": ["--trace"],
                        "args": [],
                        "full_command": "rocprof-sys --trace -- ./app",
                    },
                ],
            }
        )

    # Rule 3: Single kernel dominates
    if hotspots and len(hotspots) > 0:
        top_kernel = hotspots[0]
        percent = top_kernel.get("percent_of_total", 0)
        if percent > 50:
            kernel_name = top_kernel.get("name", "unknown")
            recommendations.append(
                {
                    "priority": "HIGH",
                    "category": "Compute Bottleneck",
                    "issue": f"Kernel '{kernel_name}' consumes {percent:.1f}% of GPU time",
                    "suggestion": "Profile this kernel with hardware counters to identify its specific bottleneck",
                    "actions": [
                        "Collect hardware counters to classify compute vs memory bound",
                        "Check memory access patterns for coalescing issues",
                        "Analyze instruction mix: VALU, MFMA, load/store ratios",
                        "Tune occupancy: balance registers, LDS, and block size",
                    ],
                    "estimated_impact": "Highly dependent on bottleneck type; 20-50% improvement possible",
                    "commands": [
                        {
                            "tool": "rocprofv3",
                            "description": f"Collect GPU hardware counters scoped to the dominant kernel",
                            "flags": ["--sys-trace"],
                            "args": [
                                {"name": "--pmc", "value": "GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES"},
                                {"name": "--kernel-names", "value": kernel_name},
                                {"name": "-d", "value": "./kernel_output"},
                                {"name": "-o", "value": "profile"},
                            ],
                            "full_command": (
                                f'rocprofv3 --sys-trace --pmc GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES'
                                f' --kernel-names "{kernel_name}"'
                                f' -d ./kernel_output -o profile -- ./app'
                            ),
                        },
                        {
                            "tool": "rocprof-compute",
                            "description": "Roofline model, instruction mix, and memory bottleneck analysis for this kernel",
                            "flags": [],
                            "args": [
                                {"name": "profile", "value": None},
                                {"name": "--kernel", "value": kernel_name},
                            ],
                            "full_command": f'rocprof-compute profile --kernel "{kernel_name}" -- ./app',
                        },
                    ],
                }
            )

    # Rule 4: Many small kernels
    if hotspots:
        total_calls = sum(k.get("calls", 0) for k in hotspots)
        if total_calls > 1000:
            avg_duration = (
                time_breakdown.get("total_kernel_time", 0) / total_calls
                if total_calls > 0
                else 0
            )
            if avg_duration < 10000:  # Less than 10 microseconds
                recommendations.append(
                    {
                        "priority": "MEDIUM",
                        "category": "Launch Overhead",
                        "issue": f"Many small kernels detected: {total_calls} launches, avg {avg_duration/1000:.1f} μs each",
                        "suggestion": "Fuse kernels or batch work to amortize per-launch overhead (~5-10 μs each)",
                        "actions": [
                            "Combine sequential element-wise kernels (e.g., add + multiply) into a single fused kernel",
                            "Increase problem size per launch to push avg duration above 50 μs",
                            "Use persistent kernels for iterative workloads to eliminate repeated launches",
                        ],
                        "estimated_impact": "Eliminates up to 50% of launch overhead for fine-grained workloads",
                        "commands": [
                            {
                                "tool": "rocprofv3",
                                "description": "Capture full kernel dispatch timeline to visualize launch frequency and gaps",
                                "flags": ["--sys-trace"],
                                "args": [
                                    {"name": "-d", "value": "./launch_output"},
                                    {"name": "-o", "value": "profile"},
                                ],
                                "full_command": "rocprofv3 --sys-trace -d ./launch_output -o profile -- ./app",
                            },
                            {
                                "tool": "rocprof-sys",
                                "description": "Visualize kernel launch timeline and inter-launch gaps in a Perfetto trace",
                                "flags": ["--trace"],
                                "args": [],
                                "full_command": "rocprof-sys --trace -- ./app",
                            },
                        ],
                    }
                )

    # Rule 5: Low memory bandwidth
    for direction, stats in memory_analysis.items():
        bandwidth_gbps = stats.get("bandwidth_bytes_per_sec", 0) / 1e9
        if bandwidth_gbps > 0 and bandwidth_gbps < 10:
            avg_bytes = stats.get("avg_bytes", 0)
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "category": "Memory Bandwidth",
                    "issue": f"{direction} copies achieving only {bandwidth_gbps:.2f} GB/s (avg transfer: {avg_bytes/1024:.1f} KB)",
                    "suggestion": "Increase transfer size per operation to reach PCIe or HBM saturation bandwidth",
                    "actions": [
                        f"Consolidate many {avg_bytes/1024:.1f} KB transfers into fewer large transfers (>1 MB each)",
                        "Use hipHostMalloc with hipHostMallocPinned flag to enable DMA engine transfers",
                        "Consider hipMemcpyAsync with stream to overlap with compute",
                        "For multi-GPU: evaluate hipMemcpyPeer for direct device-to-device transfers",
                    ],
                    "estimated_impact": "2-10x bandwidth improvement by eliminating small-transfer PCIe overhead",
                    "commands": [
                        {
                            "tool": "rocprofv3",
                            "description": "Trace memory copy operations with size and timing data",
                            "flags": ["--hsa-trace"],
                            "args": [
                                {"name": "-d", "value": "./bandwidth_output"},
                                {"name": "-o", "value": "profile"},
                            ],
                            "full_command": "rocprofv3 --hsa-trace -d ./bandwidth_output -o profile -- ./app",
                        },
                        {
                            "tool": "rocprof-compute",
                            "description": "HBM bandwidth utilization analysis for memory-bound kernels",
                            "flags": [],
                            "args": [
                                {"name": "profile", "value": None},
                                {"name": "--block", "value": "TD"},
                            ],
                            "full_command": "rocprof-compute profile --block TD -- ./app",
                        },
                    ],
                }
            )

    # Rule 6: Default if no issues found
    if not recommendations:
        recommendations.append(
            {
                "priority": "INFO",
                "category": "Performance",
                "issue": "No obvious performance issues detected at this analysis tier",
                "suggestion": "Collect deeper profiling data to find optimization opportunities",
                "actions": [
                    "Collect hardware counters to check GPU utilization and occupancy",
                    "Enable PC sampling for instruction-level hotspot analysis",
                    "Profile with rocprof-compute for roofline model and bottleneck classification",
                ],
                "estimated_impact": "Depends on findings from deeper analysis",
                "commands": [
                    {
                        "tool": "rocprofv3",
                        "description": "Collect standard hardware performance counters for Tier 2 analysis",
                        "flags": ["--sys-trace"],
                        "args": [
                            {"name": "--pmc", "value": "GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES"},
                            {"name": "-d", "value": "./counters_output"},
                            {"name": "-o", "value": "profile"},
                        ],
                        "full_command": "rocprofv3 --sys-trace --pmc GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES -d ./counters_output -o profile -- ./app",
                    },
                    {
                        "tool": "rocprof-sys",
                        "description": "Full system trace for comprehensive performance timeline",
                        "flags": ["--trace"],
                        "args": [],
                        "full_command": "rocprof-sys --trace -- ./app",
                    },
                    {
                        "tool": "rocprof-compute",
                        "description": "Complete hardware counter sweep for roofline model and bottleneck classification",
                        "flags": [],
                        "args": [
                            {"name": "profile", "value": None},
                        ],
                        "full_command": "rocprof-compute profile -- ./app",
                    },
                ],
            }
        )

    return recommendations


def _format_as_json(
    time_breakdown: Dict[str, Any],
    hotspots: List[Dict[str, Any]],
    memory_analysis: Dict[str, Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
    hardware_counters: Optional[Dict[str, Any]] = None,
    database_path: str = "",
) -> str:
    """
    Serialize analysis results to JSON conforming to schema v0.1.0.

    The output document contains a top-level ``schema_version`` field that
    consumers MUST check before parsing.  See
    ``rocpd/ai_analysis/docs/analysis-output.schema.json`` for the
    normative schema and ``SCHEMA_CHANGELOG.md`` for migration guidance.
    """
    import json as _json

    breakdown = time_breakdown or {}
    hw = hardware_counters or {}
    total_runtime_ns = int(breakdown.get("total_runtime", 0))
    kernel_time_ns = int(breakdown.get("total_kernel_time", 0))
    memcpy_time_ns = int(breakdown.get("total_memcpy_time", 0))
    kernel_pct = float(breakdown.get("kernel_percent", 0))
    memcpy_pct = float(breakdown.get("memcpy_percent", 0))
    overhead_pct = float(breakdown.get("overhead_percent", 0))
    # Derive api_overhead_ns from the percentage; clamp negative values to 0
    api_overhead_ns = max(0, int(total_runtime_ns * overhead_pct / 100.0))
    idle_time_ns = max(0, total_runtime_ns - kernel_time_ns - memcpy_time_ns - api_overhead_ns)
    idle_pct = float(idle_time_ns / total_runtime_ns * 100.0) if total_runtime_ns > 0 else 0.0

    # --- metadata ---
    has_counters = bool(hw.get("has_counters", False))
    doc: Dict[str, Any] = {
        "schema_version": "0.1.0",
        "metadata": {
            "rocpd_version": "6.3.0",
            "analysis_version": "0.1.0",
            "database_file": database_path,
            "analysis_timestamp": datetime.now().isoformat(),
            "analysis_duration_ms": 0,
            "custom_prompt": None,
        },
        # --- profiling_info ---
        "profiling_info": {
            "total_duration_ns": total_runtime_ns,
            "profiling_mode": "sys_trace_with_counters" if has_counters else "sys_trace_only",
            "analysis_tier": 2 if has_counters else 1,
            "gpus": [],
        },
        # --- summary ---
        "summary": _build_summary(breakdown, hotspots, has_counters),
        # --- execution_breakdown ---
        "execution_breakdown": {
            "total_runtime_ns": total_runtime_ns,
            "kernel_time_ns": kernel_time_ns,
            "kernel_time_pct": round(kernel_pct, 2),
            "memcpy_time_ns": memcpy_time_ns,
            "memcpy_time_pct": round(memcpy_pct, 2),
            "api_overhead_ns": api_overhead_ns,
            "api_overhead_pct": round(overhead_pct, 2),
            "idle_time_ns": idle_time_ns,
            "idle_time_pct": round(idle_pct, 2),
        },
        # --- hotspots ---
        "hotspots": [
            {
                "rank": i + 1,
                "name": k.get("name", "unknown"),
                "calls": int(k.get("calls", 0)),
                "total_duration_ns": int(k.get("total_duration", 0)),
                "avg_duration_ns": float(k.get("avg_duration", 0)),
                "min_duration_ns": int(k.get("min_duration", 0)),
                "max_duration_ns": int(k.get("max_duration", 0)),
                "pct_of_total": round(float(k.get("percent_of_total", 0)), 2),
            }
            for i, k in enumerate(hotspots or [])
        ],
        # --- memory_analysis ---
        "memory_analysis": {
            direction: {
                "count": int(s.get("count", 0)),
                "total_bytes": int(s.get("total_bytes", 0)),
                "total_duration_ns": int(s.get("total_duration", 0)),
                "avg_bytes": float(s.get("avg_bytes", 0)),
                "avg_duration_ns": float(s.get("avg_duration", 0)),
                "bandwidth_gbps": round(
                    float(s.get("bandwidth_bytes_per_sec", 0)) / 1e9, 4
                ),
            }
            for direction, s in (memory_analysis or {}).items()
        },
        # --- hardware_counters ---
        "hardware_counters": _build_hw_counters_json(hw),
        # --- recommendations ---
        "recommendations": _build_recommendations_json(recommendations or []),
        # --- warnings ---
        "warnings": _build_warnings_json(has_counters),
        "errors": [],
        "llm_enhanced_explanation": None,
    }

    return _json.dumps(doc, indent=2)


def _build_summary(
    breakdown: Dict[str, Any],
    hotspots: List[Dict[str, Any]],
    has_counters: bool,
) -> Dict[str, Any]:
    """Derive the summary section from analysis data."""
    memcpy_pct = float(breakdown.get("memcpy_percent", 0))
    kernel_pct = float(breakdown.get("kernel_percent", 0))
    overhead_pct = float(breakdown.get("overhead_percent", 0))

    # Simple bottleneck classification
    if memcpy_pct > 30:
        bottleneck = "memory_transfer"
        confidence = 0.85
    elif memcpy_pct > 20:
        bottleneck = "memory_transfer"
        confidence = 0.70
    elif overhead_pct > 25:
        bottleneck = "latency"
        confidence = 0.75
    elif kernel_pct > 70 and has_counters:
        bottleneck = "compute"
        confidence = 0.80
    elif kernel_pct > 70:
        bottleneck = "compute"
        confidence = 0.60
    else:
        bottleneck = "mixed"
        confidence = 0.50

    top_kernel = hotspots[0].get("name", "N/A") if hotspots else "N/A"
    key_findings = [
        f"Kernel execution: {kernel_pct:.1f}% of total runtime",
        f"Memory copy overhead: {memcpy_pct:.1f}% of total runtime",
        f"Top kernel: {top_kernel}",
    ]
    if has_counters:
        key_findings.append("Hardware counter data available (Tier 2 analysis)")
    else:
        key_findings.append("No hardware counters — Tier 1 trace analysis only")

    return {
        "overall_assessment": (
            f"Workload is {bottleneck.replace('_', ' ')}-bound "
            f"with {len(hotspots)} unique kernels analyzed. "
            f"Kernel time: {kernel_pct:.1f}%, memory copies: {memcpy_pct:.1f}%."
        ),
        "primary_bottleneck": bottleneck,
        "confidence": round(confidence, 2),
        "key_findings": key_findings,
    }


def _build_hw_counters_json(hw: Dict[str, Any]) -> Dict[str, Any]:
    """Convert hardware_counters internal dict to schema-compliant form."""
    has_counters = bool(hw.get("has_counters", False))
    if not has_counters:
        return {"has_counters": False, "metrics": None, "counters": None}

    raw_metrics = hw.get("metrics", {}) or {}
    metrics: Dict[str, Any] = {
        "gpu_utilization_pct": raw_metrics.get("gpu_utilization_percent"),
        "avg_waves": raw_metrics.get("avg_waves"),
        "max_waves": raw_metrics.get("max_waves"),
        "min_waves": raw_metrics.get("min_waves"),
    }

    raw_counters = hw.get("counters", {}) or {}
    counters = {
        name: {
            "sample_count": int(s.get("sample_count", 0)),
            "avg_value": float(s.get("avg_value", 0)),
            "min_value": float(s.get("min_value", 0)),
            "max_value": float(s.get("max_value", 0)),
            "total_value": float(s.get("total_value", 0)),
        }
        for name, s in raw_counters.items()
    }

    return {"has_counters": True, "metrics": metrics, "counters": counters}


# Stable IDs for known recommendation categories.
_CATEGORY_IDS = {
    "Low Occupancy": "ROCPD-OCCUPANCY-001",
    "GPU Utilization": "ROCPD-UTILIZATION-001",
    "Memory Transfer": "ROCPD-MEMCPY-001",
    "API Overhead": "ROCPD-API-001",
    "Compute Bottleneck": "ROCPD-COMPUTE-001",
    "Launch Overhead": "ROCPD-LAUNCH-001",
    "Memory Bandwidth": "ROCPD-MEMBW-001",
    "Performance": "ROCPD-INFO-001",
}


def _build_recommendations_json(
    recommendations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Map internal recommendation dicts to the schema v0.1.0 format."""
    out = []
    seen_ids: Dict[str, int] = {}
    for rec in recommendations:
        category = rec.get("category", "General")
        base_id = _CATEGORY_IDS.get(category, f"ROCPD-{category.upper().replace(' ', '-')[:12]}-001")
        count = seen_ids.get(base_id, 0) + 1
        seen_ids[base_id] = count
        rec_id = base_id if count == 1 else f"{base_id[:-3]}{count:03d}"

        out.append({
            "id": rec_id,
            "priority": rec.get("priority", "INFO"),
            "category": category,
            "issue": rec.get("issue", ""),
            "suggestion": rec.get("suggestion", ""),
            "actions": rec.get("actions", []),
            "estimated_impact": rec.get("estimated_impact", ""),
            "commands": rec.get("commands", []),
        })
    return out


def _build_warnings_json(has_counters: bool) -> List[Dict[str, Any]]:
    """Build the warnings list based on analysis context."""
    if not has_counters:
        return [
            {
                "severity": "warning",
                "message": (
                    "No hardware counters collected. Analysis limited to "
                    "Tier 1 (trace data only)."
                ),
                "recommendation": (
                    "Collect counters with: "
                    "rocprofv3 --pmc GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES -- ./app"
                ),
            }
        ]
    return []


def _format_as_markdown(
    time_breakdown: Dict[str, Any],
    hotspots: List[Dict[str, Any]],
    memory_analysis: Dict[str, Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
    hardware_counters: Optional[Dict[str, Any]] = None,
    database_path: str = "",
) -> str:
    """Format analysis results as Markdown."""
    breakdown = time_breakdown or {}
    hw = hardware_counters or {}
    has_counters = bool(hw.get("has_counters", False))

    total_runtime_ms = breakdown.get("total_runtime", 0) / 1e6
    kernel_pct = breakdown.get("kernel_percent", 0)
    memcpy_pct = breakdown.get("memcpy_percent", 0)
    overhead_pct = breakdown.get("overhead_percent", 0)
    kernel_ms = breakdown.get("total_kernel_time", 0) / 1e6
    memcpy_ms = breakdown.get("total_memcpy_time", 0) / 1e6

    lines = []
    lines.append("# ROCpd AI Performance Analysis")
    lines.append("")
    if database_path:
        lines.append(f"**Database:** `{database_path}`")
    lines.append(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    tier = 2 if has_counters else 1
    lines.append(f"**Analysis Tier:** {tier} ({'Hardware Counters' if has_counters else 'Trace Only'})")
    lines.append("")

    lines.append("## Time Breakdown")
    lines.append("")
    lines.append("| Category | Time (ms) | Percentage |")
    lines.append("|----------|-----------|------------|")
    lines.append(f"| Kernel Execution | {kernel_ms:,.2f} | {kernel_pct:.1f}% |")
    lines.append(f"| Memory Copies | {memcpy_ms:,.2f} | {memcpy_pct:.1f}% |")
    overhead_ms = total_runtime_ms - kernel_ms - memcpy_ms if total_runtime_ms > 0 else 0
    lines.append(f"| API Overhead | {overhead_ms:,.2f} | {overhead_pct:.1f}% |")
    lines.append(f"| **Total** | **{total_runtime_ms:,.2f}** | **100%** |")
    lines.append("")

    if hotspots:
        lines.append("## Top Kernel Hotspots")
        lines.append("")
        lines.append("| Rank | Kernel | Calls | Total (ms) | Avg (μs) | % Total |")
        lines.append("|------|--------|-------|------------|----------|---------|")
        for i, k in enumerate(hotspots, 1):
            name = k.get("name", "unknown")
            if len(name) > 40:
                name = name[:37] + "..."
            lines.append(
                f"| {i} | `{name}` | {k.get('calls', 0)} "
                f"| {k.get('total_duration', 0)/1e6:,.2f} "
                f"| {k.get('avg_duration', 0)/1e3:,.1f} "
                f"| {k.get('percent_of_total', 0):.1f}% |"
            )
        lines.append("")

    if memory_analysis:
        lines.append("## Memory Copy Analysis")
        lines.append("")
        lines.append("| Direction | Count | Total Size | Duration (ms) | Bandwidth (GB/s) |")
        lines.append("|-----------|-------|------------|---------------|-----------------|")
        for direction, s in memory_analysis.items():
            tb = s.get("total_bytes", 0)
            if tb >= 1e9:
                size_str = f"{tb/1e9:.1f} GB"
            elif tb >= 1e6:
                size_str = f"{tb/1e6:.1f} MB"
            elif tb >= 1e3:
                size_str = f"{tb/1e3:.1f} KB"
            else:
                size_str = f"{tb:.0f} B"
            bw = s.get("bandwidth_bytes_per_sec", 0) / 1e9
            lines.append(
                f"| {direction} | {s.get('count', 0)} | {size_str} "
                f"| {s.get('total_duration', 0)/1e6:,.2f} | {bw:.2f} |"
            )
        lines.append("")

    if has_counters:
        metrics = hw.get("metrics", {}) or {}
        lines.append("## Hardware Counters (Tier 2)")
        lines.append("")
        if "gpu_utilization_percent" in metrics:
            lines.append(f"- **GPU Utilization:** {metrics['gpu_utilization_percent']:.1f}%")
        if "avg_waves" in metrics:
            lines.append(f"- **Avg Wave Occupancy:** {metrics['avg_waves']:.1f} waves")
            lines.append(f"- **Max Wave Occupancy:** {metrics.get('max_waves', 0):.1f} waves")
        lines.append("")

    if recommendations:
        lines.append("## Recommendations")
        lines.append("")
        priority_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢", "INFO": "🔵"}
        for rec in recommendations:
            p = rec.get("priority", "INFO")
            emoji = priority_emoji.get(p, "•")
            lines.append(f"### {emoji} [{p}] {rec.get('category', '')}")
            lines.append("")
            lines.append(f"**Issue:** {rec.get('issue', '')}")
            lines.append("")
            lines.append(f"**Suggestion:** {rec.get('suggestion', '')}")
            actions = rec.get("actions", [])
            if actions:
                lines.append("")
                for action in actions:
                    lines.append(f"{action}")
            estimated_impact = rec.get("estimated_impact", "")
            if estimated_impact:
                lines.append("")
                lines.append(f"**Estimated Impact:** {estimated_impact}")
            commands = rec.get("commands", [])
            if commands:
                lines.append("")
                lines.append("**Recommended Commands:**")
                lines.append("")
                for cmd in commands:
                    tool = cmd.get("tool", "")
                    desc = cmd.get("description", "")
                    full_command = cmd.get("full_command", "")
                    flags = cmd.get("flags", [])
                    args = cmd.get("args", [])
                    lines.append(f"*{tool}* — {desc}")
                    if flags:
                        lines.append(f"- Flags: `{' '.join(flags)}`")
                    if args:
                        arg_strs = []
                        for a in args:
                            name = a.get("name", "")
                            value = a.get("value")
                            arg_strs.append(
                                f"{name} {value}" if value is not None else name
                            )
                        lines.append(f"- Args: `{' '.join(arg_strs)}`")
                    if full_command:
                        lines.append(f"```bash\n{full_command}\n```")
                    lines.append("")
            lines.append("")

    lines.append("---")
    lines.append(f"*Generated by rocpd analyze • {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    return "\n".join(lines)


def format_analysis_output(
    time_breakdown: Dict[str, Any],
    hotspots: List[Dict[str, Any]],
    memory_analysis: Dict[str, Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
    hardware_counters: Optional[Dict[str, Any]] = None,
    database_path: str = "",
    output_format: str = "text",
) -> str:
    """
    Format analysis results for display.

    Args:
        time_breakdown: Time distribution metrics
        hotspots: Top kernel hotspots
        memory_analysis: Memory copy analysis
        recommendations: Performance recommendations
        database_path: Path to analyzed database
        output_format: Output format (currently only "text" supported)

    Returns:
        Formatted string output
    """
    if output_format == "json":
        return _format_as_json(
            time_breakdown=time_breakdown,
            hotspots=hotspots,
            memory_analysis=memory_analysis,
            recommendations=recommendations,
            hardware_counters=hardware_counters,
            database_path=database_path,
        )

    if output_format == "markdown":
        return _format_as_markdown(
            time_breakdown=time_breakdown,
            hotspots=hotspots,
            memory_analysis=memory_analysis,
            recommendations=recommendations,
            hardware_counters=hardware_counters,
            database_path=database_path,
        )

    # Default: text
    lines = []
    width = 80

    # Header
    lines.append("=" * width)
    lines.append("ROCPD AI PERFORMANCE ANALYSIS".center(width))
    lines.append("=" * width)
    if database_path:
        lines.append(f"Database: {database_path}")
    lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    total_runtime_ms = time_breakdown.get("total_runtime", 0) / 1e6
    lines.append(f"Total Runtime: {total_runtime_ms:,.2f} ms")
    lines.append("")

    # Time Breakdown
    lines.append("━" * width)
    lines.append("TIME BREAKDOWN".center(width))
    lines.append("━" * width)
    lines.append("")

    def make_bar(percent: float, bar_width: int = 30) -> str:
        """Create a visual percentage bar."""
        filled = int(percent / 100.0 * bar_width)
        return "█" * filled

    kernel_pct = time_breakdown.get("kernel_percent", 0)
    memcpy_pct = time_breakdown.get("memcpy_percent", 0)
    overhead_pct = time_breakdown.get("overhead_percent", 0)

    kernel_time_ms = time_breakdown.get("total_kernel_time", 0) / 1e6
    memcpy_time_ms = time_breakdown.get("total_memcpy_time", 0) / 1e6
    overhead_time_ms = (total_runtime_ms - kernel_time_ms - memcpy_time_ms) if total_runtime_ms > 0 else 0

    lines.append(
        f"  Kernel Execution:  {kernel_time_ms:10,.2f} ms  ({kernel_pct:5.1f}%)  {make_bar(kernel_pct)}"
    )
    lines.append(
        f"  Memory Copies:     {memcpy_time_ms:10,.2f} ms  ({memcpy_pct:5.1f}%)  {make_bar(memcpy_pct)}"
    )
    lines.append(
        f"  API Overhead:      {overhead_time_ms:10,.2f} ms  ({overhead_pct:5.1f}%)  {make_bar(overhead_pct)}"
    )
    lines.append("")

    # Hotspots
    if hotspots:
        lines.append("━" * width)
        lines.append("HOTSPOTS".center(width))
        lines.append("━" * width)
        lines.append("")
        lines.append(f"Top {len(hotspots)} Kernels by Duration:")
        lines.append("")

        # Table header
        lines.append(
            f" #  {'Kernel Name':<30}  {'Calls':>6}  {'Total (ms)':>10}  {'Avg (μs)':>9}  {'% Total':>7}"
        )
        lines.append("─" * width)

        # Table rows
        for i, kernel in enumerate(hotspots, 1):
            name = kernel.get("name", "unknown")
            if len(name) > 30:
                name = name[:27] + "..."

            calls = kernel.get("calls", 0)
            total_ms = kernel.get("total_duration", 0) / 1e6
            avg_us = kernel.get("avg_duration", 0) / 1e3
            percent = kernel.get("percent_of_total", 0)

            lines.append(
                f"{i:2}  {name:<30}  {calls:6}  {total_ms:10,.2f}  {avg_us:9,.1f}  {percent:6.1f}%"
            )

        lines.append("")

    # Memory Analysis
    if memory_analysis:
        lines.append("━" * width)
        lines.append("MEMORY COPY ANALYSIS".center(width))
        lines.append("━" * width)
        lines.append("")

        # Table header
        lines.append(
            f"{'Direction':<20}  {'Count':>6}  {'Total Size':>12}  {'Duration':>10}  {'Bandwidth':>10}"
        )
        lines.append("─" * width)

        # Table rows
        for direction, stats in memory_analysis.items():
            count = stats.get("count", 0)
            total_bytes = stats.get("total_bytes", 0)
            duration_ms = stats.get("total_duration", 0) / 1e6
            bandwidth_gbps = stats.get("bandwidth_bytes_per_sec", 0) / 1e9

            # Format size
            if total_bytes >= 1e9:
                size_str = f"{total_bytes/1e9:.1f} GB"
            elif total_bytes >= 1e6:
                size_str = f"{total_bytes/1e6:.1f} MB"
            elif total_bytes >= 1e3:
                size_str = f"{total_bytes/1e3:.1f} KB"
            else:
                size_str = f"{total_bytes:.0f} B"

            lines.append(
                f"{direction:<20}  {count:6}  {size_str:>12}  {duration_ms:9,.2f} ms  {bandwidth_gbps:8.2f} GB/s"
            )

        lines.append("")

    # Hardware Counters (Tier 2)
    if hardware_counters and hardware_counters.get("has_counters"):
        lines.append("━" * width)
        lines.append("HARDWARE COUNTERS (Tier 2)".center(width))
        lines.append("━" * width)
        lines.append("")

        metrics = hardware_counters.get("metrics", {})
        counters = hardware_counters.get("counters", {})

        # Display derived metrics
        if metrics:
            lines.append("Derived Metrics:")
            lines.append("")

            if "gpu_utilization_percent" in metrics:
                util_pct = metrics["gpu_utilization_percent"]
                lines.append(f"  GPU Utilization:        {util_pct:6.1f}%  {make_bar(util_pct)}")

            if "avg_waves" in metrics:
                avg_waves = metrics["avg_waves"]
                max_waves = metrics.get("max_waves", 0)
                lines.append(f"  Avg Wave Occupancy:     {avg_waves:6.1f} waves")
                lines.append(f"  Max Wave Occupancy:     {max_waves:6.1f} waves")

            lines.append("")

        # Display raw counters
        if counters:
            lines.append("Collected Counters:")
            lines.append("")
            lines.append(f"{'Counter Name':<25}  {'Avg Value':>15}  {'Min Value':>15}  {'Max Value':>15}")
            lines.append("─" * width)

            for counter_name, stats in counters.items():
                avg = stats.get("avg_value", 0)
                min_val = stats.get("min_value", 0)
                max_val = stats.get("max_value", 0)

                lines.append(
                    f"{counter_name:<25}  {avg:15,.1f}  {min_val:15,.1f}  {max_val:15,.1f}"
                )

            lines.append("")

    # Recommendations
    lines.append("━" * width)
    lines.append("RECOMMENDATIONS".center(width))
    lines.append("━" * width)
    lines.append("")

    for rec in recommendations:
        priority = rec.get("priority", "INFO")
        category = rec.get("category", "")
        issue = rec.get("issue", "")
        suggestion = rec.get("suggestion", "")
        actions = rec.get("actions", [])
        commands = rec.get("commands", [])
        estimated_impact = rec.get("estimated_impact", "")

        lines.append(f"[{priority}] {category}")
        lines.append("─" * width)
        lines.append(f"  Issue: {issue}")
        lines.append("")
        if suggestion:
            lines.append(f"  Suggestion: {suggestion}")
            if actions:
                for action in actions:
                    lines.append(f"    {action}")
            lines.append("")
        if estimated_impact:
            lines.append(f"  Estimated Impact: {estimated_impact}")
            lines.append("")
        if commands:
            lines.append(f"  Recommended Commands:")
            for cmd in commands:
                tool = cmd.get("tool", "")
                desc = cmd.get("description", "")
                full_command = cmd.get("full_command", "")
                flags = cmd.get("flags", [])
                args = cmd.get("args", [])
                lines.append(f"    [{tool}] {desc}")
                if flags:
                    lines.append(f"      Flags: {' '.join(flags)}")
                if args:
                    arg_strs = []
                    for a in args:
                        name = a.get("name", "")
                        value = a.get("value")
                        arg_strs.append(f"{name} {value}" if value is not None else name)
                    lines.append(f"      Args:  {' '.join(arg_strs)}")
                if full_command:
                    lines.append(f"      $ {full_command}")
            lines.append("")
        lines.append("")

    # Footer
    lines.append("=" * width)
    lines.append("Analysis complete.".center(width))
    lines.append("=" * width)

    return "\n".join(lines)


def analyze_performance(
    connection: RocpdImportData,
    prompt: Optional[str] = None,
    top_kernels: int = 10,
    min_duration: float = 0.0,
    output_format: str = "text",
    database_path: str = "",
    llm: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    verbose: bool = False,
    **kwargs: Any,
) -> str:
    """
    Main analysis orchestrator that runs all analyses and formats output.

    Args:
        connection: RocpdImportData database connection
        prompt: Optional custom analysis prompt
        top_kernels: Number of top kernels to analyze
        min_duration: Minimum kernel duration threshold
        output_format: Output format (text, json, markdown)
        database_path: Path to database file
        llm: LLM provider (anthropic or openai)
        llm_api_key: API key for LLM provider
        verbose: Enable verbose logging
        **kwargs: Additional arguments

    Returns:
        Formatted analysis output string
    """
    # Run all analyses
    time_breakdown = compute_time_breakdown(connection)
    hotspots = identify_hotspots(connection, top_n=top_kernels, min_duration=min_duration)
    memory_analysis = analyze_memory_copies(connection)
    hardware_counters = analyze_hardware_counters(connection)  # Tier 2

    # Generate recommendations
    recommendations = generate_recommendations(
        time_breakdown, hotspots, memory_analysis, hardware_counters
    )

    # Format output
    output = format_analysis_output(
        time_breakdown=time_breakdown,
        hotspots=hotspots,
        memory_analysis=memory_analysis,
        recommendations=recommendations,
        hardware_counters=hardware_counters,
        database_path=database_path,
        output_format=output_format,
    )

    # LLM enhancement (if enabled)
    if llm:
        try:
            if verbose:
                print(f"[LLM] Enabling {llm} enhancement...")

            from .ai_analysis.llm_analyzer import LLMAnalyzer

            # Initialize LLM analyzer
            analyzer = LLMAnalyzer(
                provider=llm,
                api_key=llm_api_key,
                verbose=verbose,
            )

            # Prepare data for LLM
            analysis_data = {
                "gpu": {"name": "AMD GPU", "arch": "unknown"},  # TODO: Extract from DB
                "execution_breakdown": {
                    "kernel_time_pct": time_breakdown.get("kernel_percent", 0),
                    "memcpy_time_pct": time_breakdown.get("memcpy_percent", 0),
                    "api_overhead_pct": time_breakdown.get("overhead_percent", 0),
                },
                "kernels": [
                    {
                        "name": h.get("name", "unknown"),
                        "dispatch_count": h.get("calls", 0),
                        "pct_total_time": h.get("percent_of_total", 0),
                        "avg_duration_ns": h.get("avg_duration", 0),
                    }
                    for h in hotspots[:5]  # Top 5 kernels
                ],
                "memory_ops": {
                    direction: {
                        "count": data.get("count", 0),
                        "total_bytes": data.get("total_bytes", 0),
                        "bandwidth_gbps": data.get("bandwidth_bytes_per_sec", 0) / 1e9,
                    }
                    for direction, data in memory_analysis.items()
                },
                "has_counters": bool(hardware_counters),
                "has_pc_sampling": False,
            }

            # Get LLM enhancement
            llm_explanation = analyzer.analyze_with_llm(
                analysis_data=analysis_data,
                custom_prompt=prompt,
            )

            # Append LLM explanation to output
            if output_format == "text":
                output += "\n\n" + "=" * 80 + "\n"
                output += "AI-ENHANCED EXPLANATION (powered by {})".format(llm.upper()).center(80) + "\n"
                output += "=" * 80 + "\n\n"
                output += llm_explanation
                output += "\n\n" + "=" * 80 + "\n"
            elif output_format == "json":
                # Parse JSON and add LLM explanation
                import json
                try:
                    output_dict = json.loads(output)
                    output_dict["llm_enhanced_explanation"] = llm_explanation
                    output = json.dumps(output_dict, indent=2)
                except:
                    pass  # If parsing fails, keep original output

            if verbose:
                print(f"[LLM] Enhancement complete")

        except Exception as e:
            # Always show LLM failures on console (even without --verbose)
            import sys
            error_msg = f"⚠️  LLM enhancement failed: {e}"
            print(error_msg, file=sys.stderr)

            # Also add to output file
            warning_msg = f"\n\n{error_msg}\n(Analysis completed with local results only)\n"
            if output_format == "text":
                output += warning_msg

            # Show full traceback only in verbose mode
            if verbose:
                import traceback
                traceback.print_exc()

    return output


def add_args(parser: argparse.ArgumentParser):
    """
    Add command-line arguments for AI analysis.

    Args:
        parser: Argument parser to add arguments to

    Returns:
        Function to process parsed arguments
    """
    analysis_options = parser.add_argument_group("Analysis options")

    analysis_options.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom analysis prompt/question to guide analysis (e.g., 'Why is my matmul kernel slow?')",
    )

    analysis_options.add_argument(
        "--top-kernels",
        type=int,
        default=10,
        help="Number of top kernels to analyze (default: 10)",
    )

    analysis_options.add_argument(
        "--format",
        type=str,
        choices=["text", "json", "markdown"],
        default="text",
        help="Output format: text, json, or markdown (default: text)",
    )

    analysis_options.add_argument(
        "--min-duration",
        type=float,
        default=0.0,
        help="Minimum kernel duration threshold in microseconds (filter out short kernels)",
    )

    # LLM Enhancement Options
    llm_options = parser.add_argument_group(
        "LLM enhancement options (optional)",
        "Enable natural language explanations via Anthropic Claude or OpenAI GPT. "
        "Requires API key - see https://console.anthropic.com/ or https://platform.openai.com/api-keys"
    )

    llm_options.add_argument(
        "--llm",
        type=str,
        choices=["anthropic", "openai"],
        default=None,
        help="Enable LLM-powered analysis enhancement. Choices: 'anthropic' (Claude) or 'openai' (GPT). "
             "Requires API key set via environment variable or --llm-api-key option. "
             "Local analysis always runs first; LLM provides additional natural language insights.",
    )

    llm_options.add_argument(
        "--llm-api-key",
        type=str,
        default=None,
        help="API key for LLM provider. Alternatively, set environment variable: "
             "ANTHROPIC_API_KEY for Anthropic Claude, or OPENAI_API_KEY for OpenAI GPT. "
             "Example: --llm anthropic --llm-api-key sk-ant-... "
             "Or: export ANTHROPIC_API_KEY='sk-ant-...' && rocpd analyze --llm anthropic",
    )

    llm_options.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging (shows LLM API calls, reference guide loading, etc.)",
    )

    def process_args(input: RocpdImportData, args: argparse.Namespace):
        """Process and return valid arguments as dictionary."""
        valid_args = ["prompt", "top_kernels", "format", "min_duration", "llm", "llm_api_key", "verbose"]
        ret = {}
        for itr in valid_args:
            if hasattr(args, itr):
                val = getattr(args, itr)
                if val is not None:
                    ret[itr] = val
        # Convert min_duration from microseconds to nanoseconds
        if "min_duration" in ret:
            ret["min_duration"] = ret["min_duration"] * 1000
        return ret

    return process_args


def execute(input: RocpdImportData, config: Optional[output_config.output_config] = None, **kwargs: Any) -> RocpdImportData:
    """
    Execute AI analysis on rocpd database.

    Args:
        input: RocpdImportData object with database connection
        config: Optional output configuration
        **kwargs: Analysis parameters

    Returns:
        The input RocpdImportData object (for chaining)
    """
    # Update config if provided
    if config is not None:
        config = config.update(**kwargs)
    else:
        config = output_config.output_config(**kwargs)

    # Get database path for display
    database_path = ""
    if hasattr(input, "_paths") and input._paths:
        database_path = input._paths[0] if isinstance(input._paths, list) else str(input._paths)

    # Run analysis
    output = analyze_performance(
        connection=input,
        database_path=database_path,
        **kwargs,
    )

    # Handle output
    if config and config.output_file and config.output_path:
        output_file = os.path.join(config.output_path, config.output_file)
        os.makedirs(config.output_path, exist_ok=True)
        with open(output_file, "w") as f:
            f.write(output)
        print(f"Analysis written to: {output_file}")
    else:
        print(output)

    return input


def main(argv=None) -> int:
    """
    Main entry point for standalone execution.

    Args:
        argv: Command-line arguments (defaults to sys.argv)

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(
        prog="rocpd.analyze",
        description="AI-powered performance analysis for GPU traces",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        type=str,
        required=True,
        help="Input rocpd database file(s)",
    )

    # Add output config args
    output_config.add_args(parser)

    # Add analysis args
    process_args = add_args(parser)

    # Parse arguments
    args = parser.parse_args(argv)

    try:
        # Create database connection
        input_data = RocpdImportData(args.input)

        # Process arguments
        analysis_args = process_args(input_data, args)

        # Execute analysis
        execute(input_data, **analysis_args)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
