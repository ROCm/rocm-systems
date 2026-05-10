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
Markdown formatting functions for PerfXpert analysis results.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from .json_fmt import _normalize_hw_counter_escalation
from ._source_correlation import (
    correlate_hotspots_with_source,
    format_source_citation_inline,
)


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
    detected_kernels: Optional[List[Dict[str, Any]]] = None,
    communication: Optional[Dict[str, Any]] = None,
    # Phase 10 Live Roofline: accepted + ignored. The webview + JSON
    # renderers surface the live-roofline payload; markdown is kept
    # minimal to stay paste-friendly. Signature-compatible with
    # analyze._format_agentic_output so the unified call site works.
    roofline: Optional[Dict[str, Any]] = None,  # noqa: ARG001
) -> str:
    """Format analysis results as Markdown."""
    breakdown = time_breakdown or {}
    hw = hardware_counters or {}
    has_counters = bool(hw.get("has_counters", False))
    escalation = _normalize_hw_counter_escalation(hw)

    total_runtime_ms = breakdown.get("total_runtime", 0) / 1e6
    kernel_pct = breakdown.get("kernel_percent", 0)
    memcpy_pct = breakdown.get("memcpy_percent", 0)
    overhead_pct = breakdown.get("overhead_percent", 0)
    kernel_ms = breakdown.get("total_kernel_time", 0) / 1e6
    memcpy_ms = breakdown.get("total_memcpy_time", 0) / 1e6

    lines = []
    lines.append("# PerfXpert AI Performance Analysis")
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
    if "normalized_runtime" in breakdown:
        lines.append(f"Wall-clock total runtime: {total_runtime_ms:,.2f} ms")
        lines.append("")
        lines.append(
            f"Normalized runtime for percentage math: {breakdown['normalized_runtime'] / 1e6:,.2f} ms"
        )
        lines.append("")
    lines.append("| Category | Time (ms) | Percentage |")
    lines.append("|----------|-----------|------------|")
    lines.append(f"| Kernel Execution | {kernel_ms:,.2f} | {kernel_pct:.1f}% normalized |")
    lines.append(f"| Memory Copies | {memcpy_ms:,.2f} | {memcpy_pct:.1f}% normalized |")
    runtime_for_breakdown_ms = (
        breakdown.get("normalized_runtime", breakdown.get("total_runtime", 0)) / 1e6
    )
    overhead_ms = (
        max(0.0, runtime_for_breakdown_ms - kernel_ms - memcpy_ms)
        if runtime_for_breakdown_ms > 0
        else 0
    )
    lines.append(f"| API Overhead | {overhead_ms:,.2f} | {overhead_pct:.1f}% normalized |")
    if "normalized_runtime" in breakdown:
        lines.append(
            f"| **Normalized Total** | **{runtime_for_breakdown_ms:,.2f}** | **100% normalized** |"
        )
    else:
        lines.append(f"| **Total** | **{total_runtime_ms:,.2f}** | **100%** |")
    lines.append("")

    if hotspots:
        lines.append("## Top Kernel Hotspots")
        lines.append("")
        lines.append("| Rank | Kernel | Calls | Total (ms) | Avg (\u03bcs) | % Total |")
        lines.append("|------|--------|-------|------------|----------|---------|")
        annotated = correlate_hotspots_with_source(hotspots, detected_kernels)
        for i, k in enumerate(annotated, 1):
            name = k.get("name", "unknown")
            if len(name) > 40:
                name = name[:37] + "..."
            lines.append(
                f"| {i} | `{name}` | {k.get('calls', 0)} "
                f"| {k.get('total_duration', 0) / 1e6:,.2f} "
                f"| {k.get('avg_duration', 0) / 1e3:,.1f} "
                f"| {k.get('percent_of_total', 0):.1f}% |"
            )
            cite = format_source_citation_inline(k.get("source_locations"))
            if cite:
                lines.append(f"    - Source: {cite}")
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

    if has_counters or escalation:
        metrics = hw.get("metrics", {}) or {}
        lines.append("## Hardware Counters (Tier 2)")
        lines.append("")
        if has_counters:
            if "gpu_utilization_percent" in metrics:
                lines.append(
                    f"- **GPU Utilization:** {metrics['gpu_utilization_percent']:.1f}%"
                )
            if "avg_waves" in metrics:
                lines.append(f"- **Avg Wave Occupancy:** {metrics['avg_waves']:.1f} waves")
                lines.append(
                    f"- **Max Wave Occupancy:** {metrics.get('max_waves', 0):.1f} waves"
                )
        else:
            lines.append("- No hardware counter samples were present in this report.")
        if escalation:
            lines.append("")
            lines.append("### Counter Collection Escalation")
            lines.append("")
            if escalation.get("reason"):
                lines.append(escalation["reason"])
                lines.append("")
            lines.append(f"- **Pass count:** {int(escalation.get('pass_count', 0))}")
            pmc_groups_path = escalation.get("pmc_groups_path")
            if pmc_groups_path:
                lines.append(f"- **PMC groups file:** `{pmc_groups_path}`")
            lines.append("")
            if escalation.get("pmc_groups"):
                lines.append("**pmc_groups contents:**")
                lines.append("")
                lines.append("```text")
                lines.extend(str(line) for line in escalation["pmc_groups"])
                lines.append("```")
                lines.append("")
            commands = escalation.get("commands", [])
            if commands:
                lines.append("**Collection commands:**")
                lines.append("")
                for command in commands:
                    tool = command.get("tool", "")
                    desc = command.get("description", "")
                    if tool or desc:
                        label = f"{tool}: {desc}" if tool and desc else tool or desc
                        lines.append(f"- {label}")
                    full_command = command.get("full_command", "")
                    if full_command:
                        lines.append("")
                        lines.append("```bash")
                        lines.append(full_command)
                        lines.append("```")
                lines.append("")
        lines.append("")

    if recommendations:
        lines.append("## Recommendations")
        lines.append("")
        priority_emoji = {"HIGH": "\U0001f534", "MEDIUM": "\U0001f7e1", "LOW": "\U0001f7e2", "INFO": "\U0001f535"}
        for rec in recommendations:
            p = rec.get("priority", "INFO")
            emoji = priority_emoji.get(p, "\u2022")
            conf = rec.get("confidence")
            conf_str = f" — Confidence: {int(conf * 100)}%" if conf is not None else ""
            lines.append(f"### {emoji} [{p}] {rec.get('category', '')}{conf_str}")
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
            # Phase 10 — Change-Impact Prediction. Only rendered when the
            # specialist (or the analyze.py final-pass) attached
            # predicted_impact_range on the rec.
            pred_range = rec.get("predicted_impact_range")
            if pred_range and len(pred_range) == 2:
                lo, hi = pred_range
                pred_conf = rec.get("predicted_confidence")
                if pred_conf is None:
                    pred_conf = rec.get("confidence")
                conf_str = (
                    f" (confidence {int(float(pred_conf) * 100)}%)"
                    if pred_conf is not None
                    else ""
                )
                lines.append("")
                lines.append(
                    f"**Predicted:** {float(lo):.2f}-{float(hi):.2f}\u00d7{conf_str}"
                )
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

    if communication and communication.get("collectives"):
        lines.append("## Communication")
        lines.append("")
        summary = communication.get("summary", {}) or {}
        op_count = summary.get("op_count", 0)
        dominant = summary.get("dominant_op") or "n/a"
        avg_eff = summary.get("avg_efficiency_pct", 0.0) or 0.0
        overlap = summary.get("overlap_pct", 0.0) or 0.0
        peak = summary.get("peak_bw_gbps")
        peak_s = f"{peak:.0f} GB/s" if peak else "n/a"
        lines.append(
            f"**{op_count} collective(s)** - dominant: **{dominant}** - "
            f"avg efficiency: {avg_eff:.1f}% - peak: {peak_s} - "
            f"comm/compute overlap: {overlap:.1f}%"
        )
        if summary.get("capture_incomplete"):
            lines.append("")
            lines.append(
                "*Capture incomplete: fell back to kernel-name regex "
                "(no `category='RCCL'` spans in DB).*"
            )
        lines.append("")
        lines.append(
            "| Op | Bytes | Duration | Bus BW | Peak | Efficiency% | Overlap% |"
        )
        lines.append(
            "|----|-------|----------|--------|------|-------------|----------|"
        )
        for c in communication["collectives"]:
            mb = c.get("msg_bytes", 0) or 0
            if mb >= 1e9:
                mb_s = f"{mb / 1e9:.2f} GB"
            elif mb >= 1e6:
                mb_s = f"{mb / 1e6:.1f} MB"
            elif mb >= 1e3:
                mb_s = f"{mb / 1e3:.1f} KB"
            else:
                mb_s = f"{mb} B"
            dur_ns = c.get("duration_ns", 0) or 0
            dur_ms = dur_ns / 1e6
            bw = c.get("effective_bw_gbps", 0) or 0
            pk = c.get("peak_bw_gbps")
            pk_s = f"{pk:.0f}" if pk else "-"
            eff = c.get("efficiency_pct", 0) or 0
            ov = c.get("overlap_ratio", 0) or 0
            lines.append(
                f"| {c.get('op_type', '?')} | {mb_s} | {dur_ms:.3f} ms | "
                f"{bw:.2f} GB/s | {pk_s} | {eff:.1f}% | {ov:.1f}% |"
            )
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
        f"*Generated by PerfXpert analyze \u2022 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
    )
    return "\n".join(lines)


def _format_tier0_markdown(tier0_result: Any, has_profiling: bool = False) -> str:
    """Format Tier 0 source-only analysis as Markdown."""
    lines = []
    if not has_profiling:
        lines.append("# PerfXpert AI Profiling Plan \u2014 Tier 0: Source Code Analysis")
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

    if tier0_result.suggested_counters and not has_profiling:
        lines.append("## Suggested Hardware Counters")
        lines.append("")
        lines.append("```")
        lines.append(" ".join(tier0_result.suggested_counters))
        lines.append("```")
        lines.append("")

    # Bug 3 — Profiling Plan subsection (instrumentation advice).
    profiling_plan = getattr(tier0_result, "profiling_plan", None) or {}
    plan_actions = getattr(tier0_result, "profiling_plan_actions", None) or []
    if (profiling_plan or plan_actions or getattr(tier0_result, "suggested_first_command", "")) and not has_profiling:
        lines.append("### Profiling Plan")
        lines.append("")
        desc = profiling_plan.get("description") if isinstance(profiling_plan, dict) else None
        if desc:
            lines.append(desc)
            lines.append("")
        suggested_cmd = (
            profiling_plan.get("suggested_first_command")
            if isinstance(profiling_plan, dict)
            else None
        ) or tier0_result.suggested_first_command
        if suggested_cmd:
            lines.append("**Suggested first command:**")
            lines.append("")
            lines.append("```bash")
            lines.append(suggested_cmd)
            lines.append("```")
            lines.append("")
        actions_list = (
            profiling_plan.get("actions")
            if isinstance(profiling_plan, dict)
            else None
        ) or []
        extra_actions = [a for a in actions_list if a and a != suggested_cmd]
        if extra_actions:
            lines.append("**Additional actions:**")
            lines.append("")
            for a in extra_actions:
                lines.append(f"- `{a}`")
            lines.append("")

    lines.append("### Detected Code Patterns")
    lines.append("")
    priority_emoji = {"HIGH": "\U0001f534", "MEDIUM": "\U0001f7e1", "LOW": "\U0001f7e2", "INFO": "\U0001f535"}
    code_recs = (
        getattr(tier0_result, "code_patterns", None)
        or tier0_result.recommendations
        or []
    )
    if not code_recs:
        lines.append("*No code-level performance patterns detected.*")
        lines.append("")
    for rec in code_recs:
        pri = rec.get("priority", "INFO")
        cat = rec.get("category", "")
        emoji = priority_emoji.get(pri, "\u2022")
        lines.append(f"#### {emoji} [{pri}] {cat}")
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
        # Phase 10 — Change-Impact Prediction on Tier 0 rec cards.
        pred_range = rec.get("predicted_impact_range")
        if pred_range and len(pred_range) == 2:
            lo, hi = pred_range
            pred_conf = rec.get("predicted_confidence")
            if pred_conf is None:
                pred_conf = rec.get("confidence")
            conf_str = (
                f" (confidence {int(float(pred_conf) * 100)}%)"
                if pred_conf is not None
                else ""
            )
            lines.append("")
            lines.append(
                f"**Predicted:** {float(lo):.2f}-{float(hi):.2f}\u00d7{conf_str}"
            )
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

    if tier0_result.llm_explanation:
        lines.append("## AI-Enhanced Insights")
        lines.append("")
        lines.append(tier0_result.llm_explanation)
        lines.append("")

    lines.append("---")
    lines.append(
        f"*Generated by PerfXpert analyze (Tier 0) \u2022 {tier0_result.analysis_timestamp}*"
    )
    return "\n".join(lines)
