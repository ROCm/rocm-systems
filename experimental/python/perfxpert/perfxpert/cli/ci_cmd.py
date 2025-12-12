###############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc.
###############################################################################
"""perfxpert ci - CI gating wrapper over ``perfxpert diff``."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional


__all__ = ["DEFAULT_CI_THRESHOLD_PCT", "add_args", "resolve_ci_threshold", "run_ci"]


DEFAULT_CI_THRESHOLD_PCT = 5.0
_CI_THRESHOLD_ENV = "PERFXPERT_CI_REGRESSION_THRESHOLD"


def resolve_ci_threshold(cli_value: Optional[float]) -> float:
    """Return the effective CI threshold percent."""
    if cli_value is not None:
        return float(cli_value)
    env_value = os.environ.get(_CI_THRESHOLD_ENV)
    if env_value:
        try:
            return float(env_value)
        except ValueError:
            return DEFAULT_CI_THRESHOLD_PCT
    return DEFAULT_CI_THRESHOLD_PCT


def add_args(parser: argparse.ArgumentParser) -> None:
    """Register flags for ``perfxpert ci``."""
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
        "--threshold",
        type=float,
        default=None,
        metavar="PCT",
        help=(
            "Regression threshold in percent. rc=1 when wall_delta_pct is above "
            f"the threshold. Default: {DEFAULT_CI_THRESHOLD_PCT} "
            f"(env override: ${_CI_THRESHOLD_ENV})."
        ),
    )
    parser.add_argument(
        "--format",
        type=str,
        default="text",
        choices=("text", "json", "markdown"),
        help="Output format. Default: text.",
    )
    parser.add_argument(
        "--top-kernels",
        type=int,
        default=20,
        metavar="N",
        help="How many kernels to include in the per-kernel table. Default: 20.",
    )


def run_ci(args: argparse.Namespace) -> int:
    """Execute ``perfxpert ci``."""
    missing = [path for path in (args.baseline_db, args.new_db) if not os.path.exists(path)]
    if missing:
        for path in missing:
            print(f"error: database not found: {path}", file=sys.stderr)
        return 2

    from perfxpert.cli.diff_cmd import render_diff
    from perfxpert.tools.trace_diff import diff_runs

    threshold = resolve_ci_threshold(args.threshold)
    diff_result = diff_runs(
        args.baseline_db,
        args.new_db,
        top_kernels=getattr(args, "top_kernels", 20),
    )
    wall_pct = float(diff_result.get("wall_delta_pct", 0.0))

    sys.stdout.write(render_diff(diff_result, args.format))
    if not sys.stdout.isatty():
        sys.stdout.flush()

    if wall_pct > threshold:
        print(
            f"perfxpert ci: runtime regressed by {wall_pct:+.2f}% "
            f"(threshold: {threshold:.2f}%)",
            file=sys.stderr,
        )
        return 1

    print(
        f"perfxpert ci: runtime delta {wall_pct:+.2f}% within threshold {threshold:.2f}%",
        file=sys.stderr,
    )
    return 0
