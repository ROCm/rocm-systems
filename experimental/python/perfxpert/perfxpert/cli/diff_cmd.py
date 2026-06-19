###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################
"""perfxpert diff - compare two rocprofiler databases."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


__all__ = ["DIFF_FORMATS", "add_args", "render_diff", "run_diff"]


DIFF_FORMATS = ("webview", "text", "json", "markdown")


def add_args(parser: argparse.ArgumentParser) -> None:
    """Register flags for ``perfxpert diff``."""
    parser.add_argument(
        "baseline_db",
        type=str,
        metavar="BASELINE.DB",
        help="Path to the baseline database (before change).",
    )
    parser.add_argument(
        "new_db",
        type=str,
        metavar="NEW.DB",
        help="Path to the new database (after change).",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="text",
        choices=DIFF_FORMATS,
        help="Output format. Default: text.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        metavar="NAME",
        help=(
            "Output filename stem. Default: stdout for text/markdown/json, "
            "auto-named file for webview."
        ),
    )
    parser.add_argument(
        "-d",
        "--output-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Output directory (default: current working directory).",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default=None,
        metavar="PROV",
        choices=["anthropic", "openai", "ollama", "private", "opencode"],
        help="Optional provider for future narrative rewrite support.",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Reserved for future source correlation support in diff reports.",
    )
    parser.add_argument(
        "--top-kernels",
        type=int,
        default=20,
        metavar="N",
        help="How many kernels to include in the per-kernel table. Default: 20.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        default=False,
        help="Suppress progress output.",
    )


def _validate_dbs(baseline_db: str, new_db: str) -> Optional[str]:
    """Return an error string when either DB is missing."""
    missing = [path for path in (baseline_db, new_db) if not os.path.exists(path)]
    if missing:
        return "database not found: " + ", ".join(missing)
    return None


def render_diff(diff_result: Dict[str, Any], fmt: str) -> str:
    """Render a trace_diff dict in one of four formats."""
    if fmt == "json":
        return json.dumps(diff_result, indent=2)
    if fmt == "webview":
        from perfxpert.formatters.webview import _format_diff_webview

        return _format_diff_webview(diff_result)
    if fmt == "markdown":
        return _render_markdown(diff_result)
    return _render_text(diff_result)


def _render_text(diff_result: Dict[str, Any]) -> str:
    lines = []
    wall_pct = float(diff_result.get("wall_delta_pct", 0.0))
    wall_ns = int(diff_result.get("wall_delta_ns", 0))
    regressions = diff_result.get("primary_regressions", []) or []
    improvements = diff_result.get("primary_improvements", []) or []
    lines.append("=" * 72)
    lines.append(" PerfXpert trace diff")
    lines.append("=" * 72)
    lines.append(f"  baseline : {diff_result.get('baseline_db')}")
    lines.append(f"  new      : {diff_result.get('new_db')}")
    lines.append(
        f"  wall delta: {wall_pct:+.2f}% ({wall_ns:+,} ns) "
        f"regressions={len(regressions)}, improvements={len(improvements)}"
    )
    lines.append("-" * 72)
    lines.append(f" {'Kernel':<35} {'Base (ns)':>12} {'New (ns)':>12} {'Delta':>10}")
    lines.append("-" * 72)
    for kernel in diff_result.get("per_kernel") or []:
        name = str(kernel.get("name", "?"))
        if len(name) > 33:
            name = name[:32] + "."
        lines.append(
            f" {name:<35} "
            f"{int(kernel.get('baseline_ns', 0)):>12,} "
            f"{int(kernel.get('new_ns', 0)):>12,} "
            f"{float(kernel.get('delta_pct', 0.0)):>+9.2f}%"
        )
    lines.append("")
    lines.append("Summary:")
    for line in (diff_result.get("narrative", "") or "").splitlines():
        lines.append(f"  {line}")
    lines.append("")
    return "\n".join(lines)


def _render_markdown(diff_result: Dict[str, Any]) -> str:
    wall_pct = float(diff_result.get("wall_delta_pct", 0.0))
    wall_ns = int(diff_result.get("wall_delta_ns", 0))
    regressions = diff_result.get("primary_regressions", []) or []
    improvements = diff_result.get("primary_improvements", []) or []
    lines = [
        "# PerfXpert - Trace diff",
        "",
        f"- **Baseline:** `{diff_result.get('baseline_db')}`",
        f"- **New:** `{diff_result.get('new_db')}`",
        f"- **Wall-time delta:** `{wall_pct:+.2f}%` ({wall_ns:+,} ns)",
        f"- **Regressions (> +3%):** {len(regressions)}",
        f"- **Improvements (< -3%):** {len(improvements)}",
        "",
        "## Per-kernel deltas",
        "",
        "| Kernel | Baseline (ns) | New (ns) | Delta (ns) | Delta % | Hot? |",
        "|--------|---------------|----------|------------|---------|------|",
    ]
    for kernel in diff_result.get("per_kernel") or []:
        lines.append(
            f"| `{kernel.get('name')}` "
            f"| {int(kernel.get('baseline_ns', 0)):,} "
            f"| {int(kernel.get('new_ns', 0)):,} "
            f"| {int(kernel.get('delta_ns', 0)):+,} "
            f"| {float(kernel.get('delta_pct', 0.0)):+.2f}% "
            f"| {'yes' if kernel.get('was_hot') else 'no'} |"
        )
    lines.extend(["", "## Summary", "", "```", diff_result.get("narrative", "") or "", "```", ""])
    return "\n".join(lines)


def run_diff(args: argparse.Namespace) -> int:
    """Execute ``perfxpert diff`` and always return rc=0 on success."""
    error = _validate_dbs(args.baseline_db, args.new_db)
    if error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    from perfxpert.tools.trace_diff import diff_runs

    diff_result = diff_runs(
        args.baseline_db,
        args.new_db,
        top_kernels=getattr(args, "top_kernels", 20),
    )

    rendered = render_diff(diff_result, args.format)

    output_dir = Path(args.output_dir or ".").resolve()
    output_name = args.output
    ext_map = {"webview": ".html", "text": ".txt", "json": ".json", "markdown": ".md"}
    if args.format == "webview" and output_name is None:
        output_name = "diff_report"

    if output_name is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        ext = ext_map[args.format]
        stem = output_name[:-len(ext)] if output_name.endswith(ext) else output_name
        output_path = output_dir / (stem + ext)
        output_path.write_text(rendered, encoding="utf-8")
        print(f"Wrote {args.format} diff report to {output_path}")
    else:
        sys.stdout.write(rendered)
        if not rendered.endswith("\n"):
            sys.stdout.write("\n")
    return 0
