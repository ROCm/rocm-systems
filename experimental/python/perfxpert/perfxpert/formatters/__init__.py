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
Output formatting functions for PerfXpert analysis results.

Provides formatters for text, JSON, Markdown, and WebView (HTML) output.
This package was extracted from the former ``formatters.py`` flat module.
All public symbols are re-exported here for backward compatibility.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from ._common import _CATEGORY_IDS, _PERFXPERT_VERSION
from .json_fmt import (
    _build_hw_counters_json,
    _build_recommendations_json,
    _build_summary,
    _build_warnings_json,
    _format_as_json,
    _format_tier0_json,
    _normalize_hw_counter_escalation,
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
    kernel_resources: Optional[Dict[str, Any]] = None,
    api_overhead: Optional[Dict[str, Any]] = None,
    communication: Optional[Dict[str, Any]] = None,  # Phase 10 RCCL / NIC
    roofline: Optional[Dict[str, Any]] = None,       # Phase 10 Live Roofline
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
            kernel_resources=kernel_resources,
            api_overhead=api_overhead,
            communication=communication,
            roofline=roofline,
        )
        # Combined mode: embed tier0 into JSON document
        if tier0_result is not None:
            import json as _json

            try:
                doc = _json.loads(output)
                doc["tier0"] = _tier0_to_dict(tier0_result, has_profiling=bool(database_path))
                output = _json.dumps(doc, indent=2)
            except Exception:
                pass  # Tier0 embedding into combined JSON is non-fatal; return Tier1/2 output unchanged
        return output

    if output_format == "markdown":
        _detected_kernels = (
            getattr(tier0_result, "detected_kernels", None)
            if tier0_result is not None
            else None
        )
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
            detected_kernels=_detected_kernels,
            communication=communication,
            roofline=roofline,
        )
        if tier0_result is not None:
            output += "\n\n---\n\n## Tier 0: Source Code Analysis\n\n"
            output += _format_tier0_markdown(tier0_result, has_profiling=bool(database_path))
        return output

    if output_format == "webview":
        # Surface Tier-0 detected_kernels (if any) so the Top Kernel
        # Hotspots table can correlate each row with its source location.
        _detected_kernels = None
        if tier0_result is not None:
            _detected_kernels = getattr(tier0_result, "detected_kernels", None)
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
            detected_kernels=_detected_kernels,
            communication=communication,
            roofline=roofline,
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
    if "normalized_runtime" in time_breakdown:
        normalized_runtime_ms = time_breakdown.get("normalized_runtime", 0) / 1e6
        lines.append(f"Normalized Runtime: {normalized_runtime_ms:,.2f} ms")
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
    runtime_for_breakdown_ms = (
        time_breakdown.get("normalized_runtime", time_breakdown.get("total_runtime", 0)) / 1e6
    )
    overhead_time_ms = (
        max(0.0, runtime_for_breakdown_ms - kernel_time_ms - memcpy_time_ms)
        if runtime_for_breakdown_ms > 0
        else 0
    )

    lines.append(
        f"  Kernel Execution:  {kernel_time_ms:10,.2f} ms  ({kernel_pct:5.1f}% normalized)  {make_bar(kernel_pct)}"
    )
    lines.append(
        f"  Memory Copies:     {memcpy_time_ms:10,.2f} ms  ({memcpy_pct:5.1f}% normalized)  {make_bar(memcpy_pct)}"
    )
    lines.append(
        f"  API Overhead:      {overhead_time_ms:10,.2f} ms  ({overhead_pct:5.1f}% normalized)  {make_bar(overhead_pct)}"
    )
    # Per-API annotation (ROCM-21553 C2)
    if api_overhead and api_overhead.get("has_api_data"):
        top_calls = api_overhead["api_calls"][:3]
        parts = []
        for ac in top_calls:
            ms = ac["total_ns"] / 1e6
            parts.append(f"{ac['name']} x{ac['calls']} ({ms:.1f}ms)")
        if parts:
            lines.append(f"    -> top: {', '.join(parts)}")
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

        # Cross-reference hotspots with Tier-0 source locations so each row
        # can carry a one-line "Source: file.hip:42 (definition)" annotation.
        from ._source_correlation import (
            correlate_hotspots_with_source as _corr_hs,
            format_source_citation_inline as _cite_hs,
        )
        _text_detected = (
            getattr(tier0_result, "detected_kernels", None)
            if tier0_result is not None
            else None
        )
        _text_annotated = _corr_hs(hotspots, _text_detected)

        # Table rows
        for i, kernel in enumerate(_text_annotated, 1):
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
            _cite = _cite_hs(kernel.get("source_locations"))
            if _cite:
                lines.append(f"      Source: {_cite}")

        lines.append("")

    # Kernel Resources (ROCM-21553 I1)
    if kernel_resources and kernel_resources.get("kernels"):
        lines.append("\u2501" * width)
        lines.append("KERNEL RESOURCES".center(width))
        lines.append("\u2501" * width)
        lines.append("")
        lines.append(
            f" #  {'Kernel Name':<30}  {'Block':<10}  {'VGPR':>5}  {'SGPR':>5}  {'Scratch':>9}  {'LDS':>9}"
        )
        lines.append("\u2500" * width)
        for i, kr in enumerate(kernel_resources["kernels"], 1):
            kname = kr.get("name", "unknown")
            if len(kname) > 30:
                kname = kname[:27] + "..."
            block = kr.get("block", "?")
            vgpr = kr.get("vgpr", 0)
            sgpr = kr.get("sgpr", 0)
            scratch = kr.get("scratch_bytes", 0)
            lds = kr.get("lds_bytes", 0)
            scratch_s = f"{scratch} B" if scratch < 1024 else f"{scratch / 1024:.1f} KB"
            lds_s = f"{lds} B" if lds < 1024 else f"{lds / 1024:.1f} KB"
            lines.append(
                f"{i:2}  {kname:<30}  {block:<10}  {vgpr:5}  {sgpr:5}  {scratch_s:>9}  {lds_s:>9}"
            )
        lines.append("")
        # Occupancy
        arch = kernel_resources.get("arch")
        for kr in kernel_resources["kernels"]:
            occ = kr.get("occupancy")
            if occ:
                kname = kr["name"]
                if len(kname) > 40:
                    kname = kname[:37] + "..."
                w = occ["waves_per_simd"]
                mw = occ["max_waves_per_simd"]
                pct = occ["percent"]
                lim = occ["limiting_resource"]
                lim_str = f"limited by {lim}" if lim != "none" else "no resource bottleneck"
                lines.append(
                    f"  Occupancy ({arch}): {w}/{mw} waves/SIMD ({pct:.0f}%) \u2014 {lim_str}  [{kname}]"
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

    escalation = _normalize_hw_counter_escalation(hardware_counters or {})

    # Hardware Counters (Tier 2)
    if (hardware_counters and hardware_counters.get("has_counters")) or escalation:
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
        elif not hardware_counters.get("has_counters"):
            lines.append("No hardware counter samples were present in this report.")
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

        if escalation:
            lines.append("Counter Collection Escalation:")
            lines.append("")
            if escalation.get("reason"):
                lines.append(f"  {escalation['reason']}")
                lines.append("")
            lines.append(f"  Pass Count: {int(escalation.get('pass_count', 0))}")
            pmc_groups_path = escalation.get("pmc_groups_path")
            if pmc_groups_path:
                lines.append(f"  PMC Groups File: {pmc_groups_path}")
            lines.append("")
            if escalation.get("pmc_groups"):
                lines.append("  pmc_groups contents:")
                for line in escalation["pmc_groups"]:
                    lines.append(f"    {line}")
                lines.append("")
            for command in escalation.get("commands", []):
                tool = command.get("tool", "")
                desc = command.get("description", "")
                label = f"[{tool}] {desc}".strip()
                if label:
                    lines.append(f"  {label}")
                full_command = command.get("full_command", "")
                if full_command:
                    lines.append(f"    $ {full_command}")
                lines.append("")

    # Phase 10: Live Roofline text table
    if roofline and roofline.get("kernels"):
        lines.append("\u2501" * width)
        lines.append("LIVE ROOFLINE".center(width))
        lines.append("\u2501" * width)
        lines.append("")
        ridge_ai = float((roofline.get("ridge_point") or {}).get("ai", 0.0))
        arch = roofline.get("arch", "unknown")
        dtype = str(roofline.get("dtype", "fp32")).upper()
        lines.append(
            f"  Arch: {arch}   dominant dtype: {dtype}   "
            f"ridge @ {ridge_ai:.1f} FLOPs/Byte"
        )
        lines.append("")
        lines.append(
            f"  {'Kernel':<40}  {'AI':>8}  {'GFLOPs/s':>10}  {'Regime':<9}  {'dtype':<5}"
        )
        lines.append("\u2500" * width)
        for k in roofline["kernels"]:
            name = str(k.get("name", "unknown"))
            if len(name) > 40:
                name = name[:37] + "..."
            ai = float(k.get("ai", 0))
            gflops = float(k.get("achieved_flops_per_s", 0)) / 1e9
            regime = str(k.get("bottleneck_class", "-"))
            ftype = str(k.get("fp_type", "-"))
            lines.append(
                f"  {name:<40}  {ai:8.2f}  {gflops:10,.1f}  {regime:<9}  {ftype:<5}"
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

    # Communication (RCCL / NIC) — Phase 10
    if communication and communication.get("collectives"):
        lines.append("\u2501" * width)
        lines.append("COMMUNICATION (RCCL)".center(width))
        lines.append("\u2501" * width)
        lines.append("")
        _summary = communication.get("summary", {}) or {}
        _op_count = _summary.get("op_count", 0)
        _dominant = _summary.get("dominant_op") or "n/a"
        _avg_eff = _summary.get("avg_efficiency_pct", 0.0) or 0.0
        _overlap = _summary.get("overlap_pct", 0.0) or 0.0
        _peak = _summary.get("peak_bw_gbps")
        _peak_s = f"{_peak:.0f} GB/s" if _peak else "n/a"
        _ranks = _summary.get("ranks", 0)
        lines.append(
            f"  {_op_count} collective(s)  \u2014  dominant: {_dominant}  "
            f"\u2014  ranks: {_ranks}"
        )
        lines.append(
            f"  Peak busBW: {_peak_s}   Avg efficiency: {_avg_eff:.1f}%   "
            f"Comm/Compute overlap: {_overlap:.1f}%"
        )
        if _summary.get("capture_incomplete"):
            lines.append(
                "  [note] Capture incomplete \u2014 fell back to kernel-name regex."
            )
        lines.append("")
        # Box-drawn table.
        hdr = (
            f" {'Op':<16}  {'Bytes':>10}  {'Duration':>10}  "
            f"{'Bus BW':>12}  {'Peak':>10}  {'Eff%':>6}  {'Ovlp%':>6}"
        )
        lines.append(hdr)
        lines.append("\u2500" * width)
        for c in communication["collectives"]:
            mb = c.get("msg_bytes", 0) or 0
            if mb >= 1e9:
                mb_s = f"{mb / 1e9:.2f}GB"
            elif mb >= 1e6:
                mb_s = f"{mb / 1e6:.1f}MB"
            elif mb >= 1e3:
                mb_s = f"{mb / 1e3:.1f}KB"
            else:
                mb_s = f"{mb}B"
            dur_ns = c.get("duration_ns", 0) or 0
            dur_ms = dur_ns / 1e6
            bw = c.get("effective_bw_gbps", 0) or 0
            pk = c.get("peak_bw_gbps")
            pk_s = f"{pk:.0f}" if pk else "-"
            eff = c.get("efficiency_pct", 0) or 0
            ov = c.get("overlap_ratio", 0) or 0
            op_s = str(c.get("op_type", "?"))[:16]
            lines.append(
                f" {op_s:<16}  {mb_s:>10}  {dur_ms:>8.2f}ms  "
                f"{bw:>9.2f}GB/s  {pk_s:>10}  {eff:>5.1f}%  {ov:>5.1f}%"
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

        confidence = rec.get("confidence")
        conf_str = f"  (Confidence: {int(confidence * 100)}%)" if confidence is not None else ""
        # Phase 10: [advanced] badge next to priority for pragma recs.
        adv_badge = " [advanced]" if rec.get("subtype") == "pragma" else ""
        lines.append(f"[{priority}]{adv_badge} {category}{conf_str}")
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
        # Phase 10 — Change-Impact Prediction. Only rendered when the
        # specialist attached predicted_impact_range on this rec.
        pred_range = rec.get("predicted_impact_range")
        if pred_range and len(pred_range) == 2:
            lo, hi = pred_range
            pred_conf = rec.get("predicted_confidence")
            if pred_conf is None:
                pred_conf = rec.get("confidence")
            pconf_str = (
                f" (conf {int(float(pred_conf) * 100)}%)"
                if pred_conf is not None
                else ""
            )
            lines.append(
                f"  Predicted: {float(lo):.2f}-{float(hi):.2f}x{pconf_str}"
            )
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
    "_PERFXPERT_VERSION",
]
