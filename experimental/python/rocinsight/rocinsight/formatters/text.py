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
Text formatting functions for ROCInsight analysis results.
"""

from typing import Any, Dict, List


def _tier0_recommendations_text(
    recommendations: List[Dict[str, Any]], width: int = 80
) -> List[str]:
    """Render Tier 0 recommendations as text lines (same format as Tier 1/2)."""
    lines = []
    for rec in recommendations:
        pri = rec.get("priority", "INFO")
        cat = rec.get("category", "")
        issue = rec.get("issue", "")
        suggestion = rec.get("suggestion", "")
        impact = rec.get("estimated_impact", "")
        actions = rec.get("actions", [])
        commands = rec.get("commands", [])

        lines.append(f"[{pri}] {cat}")
        lines.append("\u2500" * width)
        lines.append(f"  Issue: {issue}")
        lines.append("")
        if suggestion:
            lines.append(f"  Suggestion: {suggestion}")
            for action in actions:
                lines.append(f"    {action}")
            lines.append("")
        if impact:
            lines.append(f"  Estimated Impact: {impact}")
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
    return lines


def _format_tier0_text(tier0_result: Any) -> str:
    """Format Tier 0 source-only analysis as plain text."""
    width = 80
    lines = []
    lines.append("=" * width)
    lines.append("ROCPD AI PROFILING PLAN (TIER 0: SOURCE CODE ANALYSIS)".center(width))
    lines.append("=" * width)
    lines.append(f"Source Directory: {tier0_result.source_dir}")
    lines.append(f"Analysis Date:    {tier0_result.analysis_timestamp}")
    lines.append(f"Programming Model: {tier0_result.programming_model}")
    lines.append(
        f"Files Scanned:    {tier0_result.files_scanned}  "
        f"(skipped: {tier0_result.files_skipped})"
    )
    lines.append("")

    # Kernels
    lines.append("\u2501" * width)
    lines.append("DETECTED GPU KERNELS".center(width))
    lines.append("\u2501" * width)
    lines.append(f"  Total kernels found: {tier0_result.kernel_count}")
    if tier0_result.detected_kernels:
        for k in tier0_result.detected_kernels[:20]:
            lines.append(
                f"  \u2022 {k['name']}  ({k.get('launch_type', '')})  "
                f"{k.get('file', '').split('/')[-1]}:{k.get('line', '')}"
            )
        if len(tier0_result.detected_kernels) > 20:
            lines.append(f"  ... and {len(tier0_result.detected_kernels) - 20} more")
    else:
        lines.append("  No GPU kernels detected in source.")
    lines.append("")

    # Patterns by severity
    lines.append("\u2501" * width)
    lines.append("DETECTED PATTERNS".center(width))
    lines.append("\u2501" * width)
    if tier0_result.detected_patterns:
        for p in tier0_result.detected_patterns:
            sev = p.get("severity", "info").upper()
            cat = p.get("category", "")
            desc = p.get("description", "")
            count = p.get("count", 0)
            lines.append(f"  [{sev}] {cat} \u2014 {desc} (\u00d7{count})")
    else:
        lines.append("  No significant patterns detected.")
    lines.append("")

    # Risk areas
    if tier0_result.risk_areas:
        lines.append("\u2501" * width)
        lines.append("RISK AREAS".center(width))
        lines.append("\u2501" * width)
        for risk in tier0_result.risk_areas:
            lines.append(f"  \u26a0  {risk}")
        lines.append("")

    # ROCTx
    if tier0_result.already_instrumented:
        lines.append(
            f"  \u2713 ROCTx markers detected ({tier0_result.roctx_marker_count} markers)"
        )
        lines.append("")

    # Recommended counters
    if tier0_result.suggested_counters:
        lines.append("\u2501" * width)
        lines.append("SUGGESTED HARDWARE COUNTERS".center(width))
        lines.append("\u2501" * width)
        lines.append("  " + "  ".join(tier0_result.suggested_counters))
        lines.append("")

    # Recommendations
    lines.append("\u2501" * width)
    lines.append("PROFILING RECOMMENDATIONS".center(width))
    lines.append("\u2501" * width)
    lines.append("")
    lines.extend(_tier0_recommendations_text(tier0_result.recommendations, width))

    # Suggested first command
    if tier0_result.suggested_first_command:
        lines.append("\u2501" * width)
        lines.append("START HERE \u2014 SUGGESTED FIRST COMMAND".center(width))
        lines.append("\u2501" * width)
        lines.append("")
        lines.append(f"  $ {tier0_result.suggested_first_command}")
        lines.append("")

    # LLM explanation
    if tier0_result.llm_explanation:
        lines.append("\u2501" * width)
        lines.append("AI-ENHANCED INSIGHTS".center(width))
        lines.append("\u2501" * width)
        lines.append("")
        lines.append(tier0_result.llm_explanation)
        lines.append("")

    lines.append("=" * width)
    lines.append("Analysis complete.".center(width))
    lines.append("=" * width)

    return "\n".join(lines)
