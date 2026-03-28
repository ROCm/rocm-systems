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
Output formatting functions for ROCInsight analysis results.

Provides formatters for text, JSON, Markdown, and WebView (HTML) output.
This package was extracted from the former ``formatters.py`` flat module.
All public symbols are re-exported here for backward compatibility.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from ._common import _CATEGORY_IDS, _ROCINSIGHT_VERSION
from .json_fmt import (
    _build_hw_counters_json,
    _build_recommendations_json,
    _build_summary,
    _build_warnings_json,
    _format_as_json,
    _format_tier0_json,
    _tier0_to_dict,
)
from .markdown import _format_as_markdown, _format_tier0_markdown
from .text import _format_tier0_text, _tier0_recommendations_text
from .webview import _format_as_webview, _format_tier0_webview


def format_analysis_output(
    time_breakdown: Dict[str, Any],
    hotspots: List[Dict[str, Any]],
    memory_analysis: Dict[str, Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
    hardware_counters: Optional[Dict[str, Any]] = None,
    database_path: str = "",
    output_format: str = "text",
    tier0_result: Optional[Any] = None,
    source_only: bool = False,
    interval_timeline: Optional[Dict[str, Any]] = None,
    kernel_categories: Optional[List[Any]] = None,
    short_kernels: Optional[Dict[str, Any]] = None,
    att_analysis: Optional[Dict[str, Any]] = None,  # Tier 3 ATT
    custom_prompt: Optional[str] = None,
) -> str:
    """
    Format analysis results for display.

    Args:
        time_breakdown: Time distribution metrics
        hotspots: Top kernel hotspots
        memory_analysis: Memory copy analysis
        recommendations: Performance recommendations
        database_path: Path to analyzed database
        output_format: Output format (text, json, markdown, webview)
        tier0_result: Optional Tier 0 source analysis result
        source_only: True when no database was provided (Tier 0 only)

    Returns:
        Formatted string output
    """
    # Source-only mode: dispatch entirely to Tier 0 formatters
    if source_only and tier0_result is not None:
        if output_format == "json":
            return _format_tier0_json(tier0_result)
        if output_format == "markdown":
            return _format_tier0_markdown(tier0_result)
        if output_format == "webview":
            return _format_tier0_webview(tier0_result)
        return _format_tier0_text(tier0_result)

    if output_format == "json":
        output = _format_as_json(
            time_breakdown=time_breakdown,
            hotspots=hotspots,
            memory_analysis=memory_analysis,
            recommendations=recommendations,
            hardware_counters=hardware_counters,
            database_path=database_path,
            interval_timeline=interval_timeline,
            kernel_categories=kernel_categories,
            short_kernels=short_kernels,
            att_analysis=att_analysis,
            custom_prompt=custom_prompt,
        )
        # Combined mode: embed tier0 into JSON document
        if tier0_result is not None:
            import json as _json

            try:
                doc = _json.loads(output)
                doc["tier0"] = _tier0_to_dict(tier0_result)
                output = _json.dumps(doc, indent=2)
            except Exception:
                pass  # Tier0 embedding into combined JSON is non-fatal; return Tier1/2 output unchanged
        return output

    if output_format == "markdown":
        output = _format_as_markdown(
            time_breakdown=time_breakdown,
            hotspots=hotspots,
            memory_analysis=memory_analysis,
            recommendations=recommendations,
            hardware_counters=hardware_counters,
            database_path=database_path,
            interval_timeline=interval_timeline,
            kernel_categories=kernel_categories,
            short_kernels=short_kernels,
        )
        if tier0_result is not None:
            output += "\n\n---\n\n## Tier 0: Source Code Analysis\n\n"
            output += _format_tier0_markdown(tier0_result)
        return output

    if output_format == "webview":
        return _format_as_webview(
            time_breakdown=time_breakdown,
            hotspots=hotspots,
            memory_analysis=memory_analysis,
            recommendations=recommendations,
            hardware_counters=hardware_counters,
            database_path=database_path,
            interval_timeline=interval_timeline,
            kernel_categories=kernel_categories,
            short_kernels=short_kernels,
            att_analysis=att_analysis,
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
    lines.append("\u2501" * width)
    lines.append("TIME BREAKDOWN".center(width))
    lines.append("\u2501" * width)
    lines.append("")

    def make_bar(percent: float, bar_width: int = 30) -> str:
        """Create a visual percentage bar."""
        filled = int(percent / 100.0 * bar_width)
        return "\u2588" * filled

    kernel_pct = time_breakdown.get("kernel_percent", 0)
    memcpy_pct = time_breakdown.get("memcpy_percent", 0)
    overhead_pct = time_breakdown.get("overhead_percent", 0)

    kernel_time_ms = time_breakdown.get("total_kernel_time", 0) / 1e6
    memcpy_time_ms = time_breakdown.get("total_memcpy_time", 0) / 1e6
    overhead_time_ms = (
        max(0.0, total_runtime_ms - kernel_time_ms - memcpy_time_ms)
        if total_runtime_ms > 0
        else 0
    )

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
        lines.append("\u2501" * width)
        lines.append("HOTSPOTS".center(width))
        lines.append("\u2501" * width)
        lines.append("")
        lines.append(f"Top {len(hotspots)} Kernels by Duration:")
        lines.append("")

        # Table header
        _avg_hdr = "Avg (\u03bcs)"
        lines.append(
            f" #  {'Kernel Name':<30}  {'Calls':>6}  {'Total (ms)':>10}  {_avg_hdr:>9}  {'% Total':>7}"
        )
        lines.append("\u2500" * width)

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
        lines.append("\u2501" * width)
        lines.append("MEMORY COPY ANALYSIS".center(width))
        lines.append("\u2501" * width)
        lines.append("")

        # Table header
        lines.append(
            f"{'Direction':<20}  {'Count':>6}  {'Total Size':>12}  {'Duration':>10}  {'Bandwidth':>10}"
        )
        lines.append("\u2500" * width)

        # Table rows
        for direction, stats in memory_analysis.items():
            count = stats.get("count", 0)
            total_bytes = stats.get("total_bytes", 0)
            duration_ms = stats.get("total_duration", 0) / 1e6
            bandwidth_gbps = stats.get("bandwidth_bytes_per_sec", 0) / 1e9

            # Format size
            if total_bytes >= 1e9:
                size_str = f"{total_bytes / 1e9:.1f} GB"
            elif total_bytes >= 1e6:
                size_str = f"{total_bytes / 1e6:.1f} MB"
            elif total_bytes >= 1e3:
                size_str = f"{total_bytes / 1e3:.1f} KB"
            else:
                size_str = f"{total_bytes:.0f} B"

            lines.append(
                f"{direction:<20}  {count:6}  {size_str:>12}  {duration_ms:9,.2f} ms  {bandwidth_gbps:8.2f} GB/s"
            )

        lines.append("")

    # Hardware Counters (Tier 2)
    if hardware_counters and hardware_counters.get("has_counters"):
        lines.append("\u2501" * width)
        lines.append("HARDWARE COUNTERS (Tier 2)".center(width))
        lines.append("\u2501" * width)
        lines.append("")

        metrics = hardware_counters.get("metrics", {})
        counters = hardware_counters.get("counters", {})

        # Display derived metrics
        if metrics:
            lines.append("Derived Metrics:")
            lines.append("")

            if "gpu_utilization_percent" in metrics:
                util_pct = metrics["gpu_utilization_percent"]
                lines.append(
                    f"  GPU Utilization:        {util_pct:6.1f}%  {make_bar(util_pct)}"
                )

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
            lines.append(
                f"{'Counter Name':<25}  {'Avg Value':>15}  {'Min Value':>15}  {'Max Value':>15}"
            )
            lines.append("\u2500" * width)

            for counter_name, stats in counters.items():
                avg = stats.get("avg_value", 0)
                min_val = stats.get("min_value", 0)
                max_val = stats.get("max_value", 0)

                lines.append(
                    f"{counter_name:<25}  {avg:15,.1f}  {min_val:15,.1f}  {max_val:15,.1f}"
                )

            lines.append("")

    # TraceLens: Kernel Category Breakdown
    if kernel_categories:
        lines.append("")
        lines.append("\u2501" * width)
        lines.append("KERNEL CATEGORY BREAKDOWN (TraceLens)".center(width))
        lines.append("\u2501" * width)
        lines.append("")
        max_pct = max((c["pct_of_kernel_time"] for c in kernel_categories), default=1)
        bar_width = 30
        for cat in kernel_categories:
            pct = cat["pct_of_kernel_time"]
            bar = "\u2588" * int(bar_width * pct / max(max_pct, 1))
            cnt = cat["count"]
            avg_us = cat["avg_duration_ns"] / 1_000
            lines.append(
                f"  {cat['category']:<15} {bar:<30} {pct:5.1f}%  ({cnt} kernels, avg {avg_us:.1f}\u03bcs)"
            )
        lines.append("")

    # TraceLens: Short Kernel Analysis
    if short_kernels and short_kernels.get("short_kernel_count", 0) > 0:
        lines.append("\u2501" * width)
        lines.append("SHORT KERNEL ANALYSIS (TraceLens)".center(width))
        lines.append("\u2501" * width)
        lines.append("")
        thresh = short_kernels.get("threshold_us", 10)
        count = short_kernels["short_kernel_count"]
        wasted = short_kernels["wasted_pct_of_kernel_time"]
        lines.append(
            f"  {count} kernels below {thresh}\u03bcs threshold \u2014 {wasted:.1f}% of kernel time wasted"
        )
        if short_kernels.get("histogram"):
            hist_str = "  Histogram: " + "  ".join(
                f"[{b['bucket_label']}]: {b['count']}" for b in short_kernels["histogram"]
            )
            lines.append(hist_str)
        if short_kernels.get("top_offenders"):
            lines.append("  Top offenders:")
            for off in short_kernels["top_offenders"][:5]:
                lines.append(
                    f"    {off['name'][:50]:<52} \u00d7{off['count']}  avg {off['avg_us']:.1f}\u03bcs"
                )
        lines.append("")

    # Recommendations
    lines.append("\u2501" * width)
    lines.append("RECOMMENDATIONS".center(width))
    lines.append("\u2501" * width)
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
        lines.append("\u2500" * width)
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
            lines.append("  Recommended Commands:")
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

    # ATT Thread Trace section (Tier 3)
    if att_analysis and att_analysis.get("has_att_data"):
        lines.append("\u2501" * width)
        lines.append("TIER 3: ADVANCED THREAD TRACE (ATT)".center(width))
        lines.append("\u2501" * width)
        lines.append("")
        att_kernels = att_analysis.get("kernels", [])
        att_summary = att_analysis.get("summary", {})
        lines.append(
            f"  Kernels traced: {att_summary.get('kernel_count', len(att_kernels))}"
            f"   High-stall kernels: {att_summary.get('high_stall_kernels', 0)}"
        )
        lines.append("")
        for k in att_kernels[:5]:  # Top 5 worst kernels
            kname = k.get("name", "?")
            avg_stall = k.get("avg_stall_ratio", 0.0) * 100.0
            category = k.get("stall_category", "").replace("att_", "").replace("_", " ")
            lines.append(f"  Kernel: {kname}")
            lines.append(
                f"    Avg stall ratio: {avg_stall:.1f}%   Category: {category or 'unknown'}"
            )
            top_instrs = k.get("top_stalling_instructions", [])[:3]
            for instr in top_instrs:
                pc = instr.get("pc_offset", "?")
                stall_pct = instr.get("stall_ratio", 0.0) * 100.0
                src = instr.get("source_line", "")
                src_part = f"  {src}" if src else ""
                weighted = instr.get("weighted_stall", 0)
                lines.append(
                    f"    {pc}: stall {stall_pct:.0f}%  weighted={weighted:,}{src_part}"
                )
            lines.append("")
    elif (
        att_analysis
        and not att_analysis.get("has_att_data")
        and att_analysis.get("reason")
    ):
        lines.append("\u2501" * width)
        lines.append("TIER 3: ATT".center(width))
        lines.append("\u2501" * width)
        lines.append(f"  Note: {att_analysis['reason']}")
        lines.append("")

    # Footer
    lines.append("=" * width)
    lines.append("Analysis complete.".center(width))
    lines.append("=" * width)

    return "\n".join(lines)


__all__ = [
    "format_analysis_output",
    "_format_as_json",
    "_build_summary",
    "_build_hw_counters_json",
    "_build_recommendations_json",
    "_build_warnings_json",
    "_format_as_markdown",
    "_format_as_webview",
    "_tier0_recommendations_text",
    "_format_tier0_text",
    "_tier0_to_dict",
    "_format_tier0_json",
    "_format_tier0_markdown",
    "_format_tier0_webview",
    "_CATEGORY_IDS",
    "_ROCINSIGHT_VERSION",
]
