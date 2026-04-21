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

"""``perfxpert diff`` — compare two rocpd databases.

Confluence row #7. This is the *informational* variant (always rc=0);
``perfxpert ci`` wraps the same tool but returns rc=1 on regression above
threshold for CI systems.

Reads two databases, calls :func:`perfxpert.tools.trace_diff.diff_runs`,
and renders the result in one of four formats (webview / text / json /
markdown). The same ``trace_diff`` dict flows through every format so
there's only one diff engine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


__all__ = ["add_args", "run_diff", "render_diff", "DIFF_FORMATS"]


DIFF_FORMATS = ("webview", "text", "json", "markdown")


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------

def add_args(parser: argparse.ArgumentParser) -> None:
    """Register flags for ``perfxpert diff`` on ``parser``."""
    parser.add_argument(
        "baseline_db",
        type=str,
        metavar="BASELINE.DB",
        help="Path to the baseline rocpd database (before change).",
    )
    parser.add_argument(
        "new_db",
        type=str,
        metavar="NEW.DB",
        help="Path to the new rocpd database (after change).",
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
        help="Output filename stem (default: stdout when --format text/markdown/json; "
        "auto-named .html when --format webview).",
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
        help="Attach an LLM provider to rewrite the narrative (optional).",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Optional Tier-0 source directory (ignored for diff reports today; "
        "reserved for future per-kernel source correlation in diff output).",
    )
    parser.add_argument(
        "--top-kernels",
        type=int,
        default=20,
        metavar="N",
        help="How many kernels to include in the per-kernel table. Default: 20.",
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_dbs(baseline_db: str, new_db: str) -> Optional[str]:
    """Return an error string when either DB is missing; None otherwise."""
    missing = [p for p in (baseline_db, new_db) if not os.path.exists(p)]
    if missing:
        return "Database not found: " + ", ".join(missing)
    return None


# ---------------------------------------------------------------------------
# Format rendering — one result dict, four output formats
# ---------------------------------------------------------------------------

def render_diff(diff_result: Dict[str, Any], fmt: str) -> str:
    """Render a ``trace_diff`` dict in one of 4 formats.

    This is the single source of truth for diff-report rendering. The
    ``--baseline`` splice in ``perfxpert analyze`` and the ``perfxpert
    ci`` subcommand both call through to this function.
    """
    if fmt == "json":
        return json.dumps(diff_result, indent=2)
    if fmt == "webview":
        from perfxpert.formatters.webview import _format_diff_webview

        return _format_diff_webview(diff_result)
    if fmt == "markdown":
        return _render_markdown(diff_result)
    # default: text
    return _render_text(diff_result)


def _render_text(d: Dict[str, Any]) -> str:
    """Compact two-block text rendering — overview + per-kernel table."""
    lines = []
    wall_pct = float(d.get("wall_delta_pct", 0.0))
    wall_ns = int(d.get("wall_delta_ns", 0))
    regressions = d.get("primary_regressions", []) or []
    improvements = d.get("primary_improvements", []) or []
    lines.append("=" * 72)
    lines.append(" PerfXpert trace diff")
    lines.append("=" * 72)
    lines.append(f"  baseline : {d.get('baseline_db')}")
    lines.append(f"  new      : {d.get('new_db')}")
    lines.append(
        f"  wall delta: {wall_pct:+.2f}%  ({wall_ns:+,} ns)  "
        f"regressions={len(regressions)}, improvements={len(improvements)}"
    )
    lines.append("-" * 72)
    lines.append(
        f" {'Kernel':<35} {'Base (ns)':>12} {'New (ns)':>12} {'Delta':>10}"
    )
    lines.append("-" * 72)
    for k in (d.get("per_kernel") or []):
        name = str(k.get("name", "?"))
        if len(name) > 33:
            name = name[:32] + "…"
        lines.append(
            f" {name:<35} "
            f"{int(k.get('baseline_ns', 0)):>12,} "
            f"{int(k.get('new_ns', 0)):>12,} "
            f"{float(k.get('delta_pct', 0.0)):>+9.2f}%"
        )
    lines.append("")
    lines.append("Summary:")
    for ln in (d.get("narrative", "") or "").splitlines():
        lines.append(f"  {ln}")
    lines.append("")
    return "\n".join(lines)


def _render_markdown(d: Dict[str, Any]) -> str:
    """GitHub-flavored markdown rendering."""
    wall_pct = float(d.get("wall_delta_pct", 0.0))
    wall_ns = int(d.get("wall_delta_ns", 0))
    regressions = d.get("primary_regressions", []) or []
    improvements = d.get("primary_improvements", []) or []
    lines = [
        "# PerfXpert — Trace diff",
        "",
        f"- **Baseline:** `{d.get('baseline_db')}`",
        f"- **New:** `{d.get('new_db')}`",
        f"- **Wall-time delta:** `{wall_pct:+.2f}%` ({wall_ns:+,} ns)",
        f"- **Regressions (> +3%):** {len(regressions)}",
        f"- **Improvements (< -3%):** {len(improvements)}",
        "",
        "## Per-kernel deltas",
        "",
        "| Kernel | Baseline (ns) | New (ns) | Δ (ns) | Δ % | Hot? |",
        "|--------|---------------|----------|--------|-----|------|",
    ]
    for k in (d.get("per_kernel") or []):
        lines.append(
            f"| `{k.get('name')}` "
            f"| {int(k.get('baseline_ns', 0)):,} "
            f"| {int(k.get('new_ns', 0)):,} "
            f"| {int(k.get('delta_ns', 0)):+,} "
            f"| {float(k.get('delta_pct', 0.0)):+.2f}% "
            f"| {'yes' if k.get('was_hot') else 'no'} |"
        )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("```")
    lines.append(d.get("narrative", "") or "")
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_diff(args: argparse.Namespace) -> int:
    """Execute ``perfxpert diff`` and return rc=0 always.

    Informational only. CI systems should use ``perfxpert ci`` which wraps
    this and flips rc=1 on regression above threshold.
    """
    err = _validate_dbs(args.baseline_db, args.new_db)
    if err:
        print(f"\u26a0 {err}", file=sys.stderr)
        return 2

    from perfxpert.tools.trace_diff import diff_runs

    diff_result = diff_runs(
        args.baseline_db,
        args.new_db,
        top_kernels=getattr(args, "top_kernels", 20),
    )

    # Optional LLM narrative rewrite — plug point for future work.
    # Today we keep the deterministic narrative.
    if getattr(args, "llm", None):
        # Caller can overwrite diff_result["narrative"] after rendering.
        # We intentionally do NOT call the LLM here: the providers stack is
        # owned by analyze.py; diff stays lightweight + air-gap safe.
        pass

    rendered = render_diff(diff_result, args.format)

    # Write to disk if requested, otherwise stdout.
    out_dir = Path(args.output_dir or ".").resolve()
    out_name = args.output
    ext_map = {"webview": ".html", "text": ".txt", "json": ".json", "markdown": ".md"}
    # Webview always writes to disk so it's clickable.
    if args.format == "webview" and out_name is None:
        out_name = "diff_report"
    if out_name is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (out_name + ext_map[args.format])
        out_path.write_text(rendered, encoding="utf-8")
        print(f"Wrote {args.format} diff report to {out_path}")
    else:
        sys.stdout.write(rendered)
        if not rendered.endswith("\n"):
            sys.stdout.write("\n")
    return 0
