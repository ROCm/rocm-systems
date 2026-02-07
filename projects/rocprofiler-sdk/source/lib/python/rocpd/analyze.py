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
                    "suggestion": "Increase kernel occupancy to improve GPU utilization:",
                    "actions": [
                        "- Increase block/workgroup size to launch more waves",
                        "- Reduce register usage per thread",
                        "- Reduce shared memory (LDS) usage per workgroup",
                        "- Check for resource limitations preventing more waves",
                    ],
                    "next_steps": "Profile occupancy: rocprofv3 --pmc SQ_WAVES SQ_WAVE_CYCLES TA_TA_BUSY -- <your-app>",
                }
            )

        # Low GPU utilization
        gpu_util = metrics.get("gpu_utilization_percent", 0)
        if gpu_util > 0 and gpu_util < 70:
            recommendations.append(
                {
                    "priority": "MEDIUM",
                    "category": "GPU Utilization",
                    "issue": f"GPU utilization is only {gpu_util:.1f}%",
                    "suggestion": "Improve GPU utilization:",
                    "actions": [
                        "- Launch more concurrent kernels using streams",
                        "- Increase kernel grid size if problem permits",
                        "- Reduce kernel launch overhead and synchronization",
                        "- Check for host-side bottlenecks limiting launch rate",
                    ],
                    "next_steps": "Profile GPU activity: rocprofv3 --pmc GRBM_GUI_ACTIVE GRBM_COUNT -- <your-app>",
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
                "suggestion": "Consider reducing host-device transfers by:",
                "actions": [
                    "- Batching multiple small copies into larger transfers",
                    "- Using pinned memory for host allocations",
                    "- Overlapping compute with data transfers using streams",
                ],
                "next_steps": "Run: rocprofv3 --hip-trace --hsa-trace -- <your-app>",
            }
        )

    # Rule 2: High API overhead
    overhead_percent = time_breakdown.get("overhead_percent", 0)
    if overhead_percent > 15:
        recommendations.append(
            {
                "priority": "MEDIUM",
                "category": "API Overhead",
                "issue": f"API overhead is {overhead_percent:.1f}% of total time",
                "suggestion": "Reduce API calls by:",
                "actions": [
                    "- Launching larger kernels instead of many small ones",
                    "- Using batch operations where possible",
                    "- Minimizing synchronization points",
                ],
                "next_steps": "Profile API calls with: rocprofv3 --hip-api-trace -- <your-app>",
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
                    "suggestion": "Profile this kernel in detail:",
                    "actions": [
                        "- Collect hardware counters to identify bottlenecks",
                        "- Check for memory bandwidth limitations",
                        "- Analyze instruction mix and occupancy",
                    ],
                    "next_steps": f'Run: rocprofv3 --kernel-names "{kernel_name}" --pmc GRBM_COUNT GRBM_GUI_ACTIVE SQ_WAVES -- <your-app>',
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
                        "issue": f"Many small kernels detected ({total_calls} launches, avg {avg_duration/1000:.1f}μs)",
                        "suggestion": "Consider kernel fusion or batching to reduce launch overhead",
                        "actions": [
                            "- Combine multiple small kernels into a single launch",
                            "- Increase problem size per kernel invocation",
                        ],
                        "next_steps": "Analyze launch patterns with: rocprofv3 --sys-trace -- <your-app>",
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
                    "issue": f"{direction} copies achieving only {bandwidth_gbps:.1f} GB/s",
                    "suggestion": "Improve transfer efficiency:",
                    "actions": [
                        f"- Use larger transfer sizes (current avg: {avg_bytes/1024:.1f} KB)",
                        "- Enable async copies with streams",
                        "- Consider peer-to-peer transfers for multi-GPU",
                    ],
                    "next_steps": "Profile memory operations: rocprofv3 --hsa-trace -- <your-app>",
                }
            )

    # Rule 6: Default if no issues found
    if not recommendations:
        recommendations.append(
            {
                "priority": "INFO",
                "category": "Performance",
                "issue": "No obvious performance issues detected",
                "suggestion": "Application appears well-optimized. For deeper analysis:",
                "actions": [
                    "- Collect hardware counters: rocprofv3 --pmc <metrics> -- <your-app>",
                    "- Enable PC sampling: rocprofv3 --pc-sampling -- <your-app>",
                    "- Profile memory allocation: rocprofv3 --sys-trace -- <your-app>",
                ],
                "next_steps": "Review AMD optimization guides at: https://rocm.docs.amd.com",
            }
        )

    return recommendations


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
    if output_format != "text":
        return "Only text format is currently supported"

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
        next_steps = rec.get("next_steps", "")

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
        if next_steps:
            lines.append(f"  Next Steps:")
            lines.append(f"    $ {next_steps}")
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
