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

"""``perfxpert ci`` — the CI-gating wrapper over ``perfxpert diff``.

Confluence row #8. Same engine as ``perfxpert diff`` (shared
:func:`perfxpert.tools.trace_diff.diff_runs`), but:

  * rc=1 when ``wall_delta_pct > --threshold`` (default 5.0%,
    overridable via ``PERFXPERT_CI_REGRESSION_THRESHOLD``);
  * rc=0 otherwise;
  * Prints a one-line summary to stderr + the full diff to stdout so
    humans and tooling can both consume it.

Default format is ``text`` (human-readable). ``--format json`` is
machine-friendly for GitHub-Actions / GitLab-CI consumption.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional


__all__ = ["add_args", "run_ci", "DEFAULT_CI_THRESHOLD_PCT", "resolve_ci_threshold"]


DEFAULT_CI_THRESHOLD_PCT = 5.0
_CI_THRESHOLD_ENV = "PERFXPERT_CI_REGRESSION_THRESHOLD"


def resolve_ci_threshold(cli_value: Optional[float]) -> float:
    """Return the effective CI threshold %.

    Precedence: ``--threshold`` > ``$PERFXPERT_CI_REGRESSION_THRESHOLD``
    > :data:`DEFAULT_CI_THRESHOLD_PCT` (5.0).
    """
    if cli_value is not None:
        return float(cli_value)
    env = os.environ.get(_CI_THRESHOLD_ENV)
    if env:
        try:
            return float(env)
        except ValueError:
            # Invalid env value — fall back to default rather than raise.
            return DEFAULT_CI_THRESHOLD_PCT
    return DEFAULT_CI_THRESHOLD_PCT


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------

def add_args(parser: argparse.ArgumentParser) -> None:
    """Register flags for ``perfxpert ci`` on ``parser``."""
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
        "--threshold",
        type=float,
        default=None,
        metavar="PCT",
        help=(
            "Regression threshold in percent. rc=1 when wall_delta_pct > "
            f"threshold. Default: {DEFAULT_CI_THRESHOLD_PCT} "
            f"(env override: ${_CI_THRESHOLD_ENV})."
        ),
    )
    parser.add_argument(
        "--format",
        type=str,
        default="text",
        choices=("text", "json", "markdown"),
        help="Output format (webview is not a CI-native format). Default: text.",
    )
    parser.add_argument(
        "--top-kernels",
        type=int,
        default=20,
        metavar="N",
        help="How many kernels to include in the per-kernel table. Default: 20.",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_ci(args: argparse.Namespace) -> int:
    """Execute ``perfxpert ci``. rc=0 on pass, rc=1 on regression, rc=2 on error."""
    # Validate inputs — same check as ``perfxpert diff``.
    missing = [p for p in (args.baseline_db, args.new_db) if not os.path.exists(p)]
    if missing:
        for p in missing:
            print(f"\u26a0 Database not found: {p}", file=sys.stderr)
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

    # Print full report to stdout (CI systems often capture stdout).
    sys.stdout.write(render_diff(diff_result, args.format))
    if not sys.stdout.isatty():
        sys.stdout.flush()

    # Gate verdict.
    if wall_pct > threshold:
        # One-line summary to stderr so CI log scanners pick it up.
        print(
            f"\u274c perfxpert ci: runtime regressed by {wall_pct:+.2f}% "
            f"(threshold: {threshold:.2f}%)",
            file=sys.stderr,
        )
        return 1

    print(
        f"\u2705 perfxpert ci: runtime delta {wall_pct:+.2f}% within threshold "
        f"{threshold:.2f}%",
        file=sys.stderr,
    )
    return 0
