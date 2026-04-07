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
Markdown formatting functions for ROCInsight analysis results.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional


def _format_as_markdown(
    time_breakdown: Dict[str, Any],
    hotspots: List[Dict[str, Any]],
    memory_analysis: Dict[str, Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
    hardware_counters: Optional[Dict[str, Any]] = None,
    database_path: str = "",
    interval_timeline=None,
    kernel_categories=None,
    short_kernels=None,
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
    lines.append("# ROCInsight AI Performance Analysis")
    lines.append("")
    if database_path:
        lines.append(f"**Database:** `{database_path}`")
    lines.append(f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    tier = 2 if has_counters else 1
    lines.append(
        f"**Analysis Tier:** {tier} ({'Hardware Counters' if has_counters else 'Trace Only'})"
    )
    lines.append("")

    lines.append("## Time Breakdown")
    lines.append("")
    lines.append("| Category | Time (ms) | Percentage |")
    lines.append("|----------|-----------|------------|")
    lines.append(f"| Kernel Execution | {kernel_ms:,.2f} | {kernel_pct:.1f}% |")
    lines.append(f"| Memory Copies | {memcpy_ms:,.2f} | {memcpy_pct:.1f}% |")
    overhead_ms = (
        max(0.0, total_runtime_ms - kernel_ms - memcpy_ms) if total_runtime_ms > 0 else 0
    )
    lines.append(f"| API Overhead | {overhead_ms:,.2f} | {overhead_pct:.1f}% |")
    lines.append(f"| **Total** | **{total_runtime_ms:,.2f}** | **100%** |")
    lines.append("")

    if hotspots:
        lines.append("## Top Kernel Hotspots")
        lines.append("")
        lines.append("| Rank | Kernel | Calls | Total (ms) | Avg (\u03bcs) | % Total |")
        lines.append("|------|--------|-------|------------|----------|---------|")
        for i, k in enumerate(hotspots, 1):
            name = k.get("name", "unknown")
            if len(name) > 40:
                name = name[:37] + "..."
            lines.append(
                f"| {i} | `{name}` | {k.get('calls', 0)} "
                f"| {k.get('total_duration', 0) / 1e6:,.2f} "
                f"| {k.get('avg_duration', 0) / 1e3:,.1f} "
                f"| {k.get('percent_of_total', 0):.1f}% |"
            )
        lines.append("")

    if memory_analysis:
        lines.append("## Memory Copy Analysis")
        lines.append("")
        lines.append(
            "| Direction | Count | Total Size | Duration (ms) | Bandwidth (GB/s) |"
        )
        lines.append(
            "|-----------|-------|------------|---------------|-----------------|"
        )
        for direction, s in memory_analysis.items():
            tb = s.get("total_bytes", 0)
            if tb >= 1e9:
                size_str = f"{tb / 1e9:.1f} GB"
            elif tb >= 1e6:
                size_str = f"{tb / 1e6:.1f} MB"
            elif tb >= 1e3:
                size_str = f"{tb / 1e3:.1f} KB"
            else:
                size_str = f"{tb:.0f} B"
            bw = s.get("bandwidth_bytes_per_sec", 0) / 1e9
            lines.append(
                f"| {direction} | {s.get('count', 0)} | {size_str} "
                f"| {s.get('total_duration', 0) / 1e6:,.2f} | {bw:.2f} |"
            )
        lines.append("")

    if has_counters:
        metrics = hw.get("metrics", {}) or {}
        lines.append("## Hardware Counters (Tier 2)")
        lines.append("")
        if "gpu_utilization_percent" in metrics:
            lines.append(
                f"- **GPU Utilization:** {metrics['gpu_utilization_percent']:.1f}%"
            )
        if "avg_waves" in metrics:
            lines.append(f"- **Avg Wave Occupancy:** {metrics['avg_waves']:.1f} waves")
            lines.append(
                f"- **Max Wave Occupancy:** {metrics.get('max_waves', 0):.1f} waves"
            )
        lines.append("")

    if recommendations:
        lines.append("## Recommendations")
        lines.append("")
        priority_emoji = {"HIGH": "\U0001f534", "MEDIUM": "\U0001f7e1", "LOW": "\U0001f7e2", "INFO": "\U0001f535"}
        for rec in recommendations:
            p = rec.get("priority", "INFO")
            emoji = priority_emoji.get(p, "\u2022")
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
                    lines.append(f"*{tool}* \u2014 {desc}")
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

    if kernel_categories:
        lines.append("## Kernel Category Breakdown")
        lines.append("")
        lines.append("| Category | Kernels | % of Kernel Time | Avg Duration |")
        lines.append("|----------|---------|-----------------|--------------|")
        for cat in kernel_categories:
            avg_us = cat["avg_duration_ns"] / 1_000
            lines.append(
                f"| {cat['category']} | {cat['count']} | "
                f"{cat['pct_of_kernel_time']:.1f}% | {avg_us:.1f}\u03bcs |"
            )
        lines.append("")

    if short_kernels and short_kernels.get("short_kernel_count", 0) > 0:
        lines.append("## Short Kernel Analysis")
        lines.append("")
        thresh = short_kernels.get("threshold_us", 10)
        count = short_kernels["short_kernel_count"]
        wasted = short_kernels["wasted_pct_of_kernel_time"]
        lines.append(
            f"**{count} kernels** below {thresh}\u03bcs threshold \u2014 "
            f"**{wasted:.1f}%** of kernel time wasted"
        )
        lines.append("")
        if short_kernels.get("histogram"):
            lines.append("| Bucket | Count |")
            lines.append("|--------|-------|")
            for b in short_kernels["histogram"]:
                lines.append(f"| {b['bucket_label']} | {b['count']} |")
            lines.append("")
        if short_kernels.get("top_offenders"):
            lines.append("**Top offenders by wasted time:**")
            lines.append("")
            for off in short_kernels["top_offenders"][:5]:
                lines.append(
                    f"- `{off['name']}` \u2014 \u00d7{off['count']} calls, avg {off['avg_us']:.1f}\u03bcs"
                )
            lines.append("")

    lines.append("---")
    lines.append(
        f"*Generated by ROCInsight analyze \u2022 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
    )
    return "\n".join(lines)


def _format_tier0_markdown(tier0_result: Any) -> str:
    """Format Tier 0 source-only analysis as Markdown."""
    lines = []
    lines.append("# ROCInsight AI Profiling Plan \u2014 Tier 0: Source Code Analysis")
    lines.append("")
    lines.append(f"**Source Directory:** `{tier0_result.source_dir}`")
    lines.append(f"**Analysis Date:** {tier0_result.analysis_timestamp}")
    lines.append(f"**Programming Model:** {tier0_result.programming_model}")
    lines.append("**Analysis Tier:** 0 (Source Code Analysis)")
    lines.append("")

    lines.append("## Detected Kernels")
    lines.append("")
    lines.append(f"**Total GPU kernels found:** {tier0_result.kernel_count}")
    lines.append("")
    if tier0_result.detected_kernels:
        lines.append("| Kernel | Launch Type | File | Line |")
        lines.append("|--------|-------------|------|------|")
        for k in tier0_result.detected_kernels[:20]:
            fname = k.get("file", "").split("/")[-1]
            lines.append(
                f"| `{k['name']}` | {k.get('launch_type', '')} | {fname} | {k.get('line', '')} |"
            )
        if len(tier0_result.detected_kernels) > 20:
            lines.append(
                f"\n*... and {len(tier0_result.detected_kernels) - 20} more kernels*"
            )
    else:
        lines.append("*No GPU kernels detected in source.*")
    lines.append("")

    lines.append("## Detected Patterns")
    lines.append("")
    if tier0_result.detected_patterns:
        lines.append("| Severity | Category | Description | Count |")
        lines.append("|----------|----------|-------------|-------|")
        for p in tier0_result.detected_patterns:
            sev = p.get("severity", "info")
            lines.append(
                f"| **{sev.upper()}** | {p.get('category', '')} | {p.get('description', '')} | {p.get('count', 0)} |"
            )
    else:
        lines.append("*No significant patterns detected.*")
    lines.append("")

    if tier0_result.risk_areas:
        lines.append("## Risk Areas")
        lines.append("")
        for risk in tier0_result.risk_areas:
            lines.append(f"- \u26a0 {risk}")
        lines.append("")

    if tier0_result.suggested_counters:
        lines.append("## Suggested Hardware Counters")
        lines.append("")
        lines.append("```")
        lines.append(" ".join(tier0_result.suggested_counters))
        lines.append("```")
        lines.append("")

    lines.append("## Profiling Recommendations")
    lines.append("")
    priority_emoji = {"HIGH": "\U0001f534", "MEDIUM": "\U0001f7e1", "LOW": "\U0001f7e2", "INFO": "\U0001f535"}
    for rec in tier0_result.recommendations:
        pri = rec.get("priority", "INFO")
        cat = rec.get("category", "")
        emoji = priority_emoji.get(pri, "\u2022")
        lines.append(f"### {emoji} [{pri}] {cat}")
        lines.append("")
        lines.append(f"**Issue:** {rec.get('issue', '')}")
        lines.append("")
        lines.append(f"**Suggestion:** {rec.get('suggestion', '')}")
        actions = rec.get("actions", [])
        if actions:
            lines.append("")
            for action in actions:
                lines.append(f"{action}")
        impact = rec.get("estimated_impact", "")
        if impact:
            lines.append("")
            lines.append(f"**Estimated Impact:** {impact}")
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
                lines.append(f"*{tool}* \u2014 {desc}")
                if flags:
                    lines.append(f"- Flags: `{' '.join(flags)}`")
                if args:
                    arg_strs = []
                    for a in args:
                        name = a.get("name", "")
                        value = a.get("value")
                        arg_strs.append(f"{name} {value}" if value is not None else name)
                    lines.append(f"- Args: `{' '.join(arg_strs)}`")
                if full_command:
                    lines.append(f"```bash\n{full_command}\n```")
                lines.append("")
        lines.append("")

    if tier0_result.suggested_first_command:
        lines.append("## Start Here \u2014 Suggested First Command")
        lines.append("")
        lines.append("```bash")
        lines.append(tier0_result.suggested_first_command)
        lines.append("```")
        lines.append("")

    if tier0_result.llm_explanation:
        lines.append("## AI-Enhanced Insights")
        lines.append("")
        lines.append(tier0_result.llm_explanation)
        lines.append("")

    lines.append("---")
    lines.append(
        f"*Generated by ROCInsight analyze (Tier 0) \u2022 {tier0_result.analysis_timestamp}*"
    )
    return "\n".join(lines)
