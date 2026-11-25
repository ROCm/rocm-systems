#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
Run a traced command until its kernel trace has no bubbles.

The bubble check of the hip-graph-bubbles test fails when one scheduling stall on a
shared host opens a gap of a few hundred microseconds. Such a stall does not point to a
profiler regression, but a bubble that is there on every run does. This script runs the
command again, up to --attempts times, until the kernel trace passes the same check as
validate.py. The trace of the last run stays in place for the validate step, so a bubble
that is there on every run still fails there with the usual message.

Only the bubble check is retried. A command that exits with an error code ends the
loop, and so does a trace that cannot be read, so that the validate step reports them.

Usage: run_with_bubble_retry.py --csv <kernel trace CSV> [--attempts N] -- <command...>
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

# The typing aliases, not list[...] or X | None: the RHEL 8 and SLES 15 CI images run
# this script with Python 3.6.
from typing import List, Optional, Tuple

import bubble_gaps

DEFAULT_ATTEMPTS = 3


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a command until its kernel trace has no bubbles."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        required=True,
        help="Kernel trace CSV that the command writes.",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=DEFAULT_ATTEMPTS,
        help="Maximum number of runs of the command.",
    )
    parser.add_argument("command", nargs="+", help="Command to run, after --.")

    ret = parser.parse_args(argv)
    if ret.attempts < 1:
        parser.error("--attempts must be at least 1")
    return ret


def find_outlier_gap(csv_path: Path) -> Optional[Tuple[int, int, int]]:
    """Return the 99.9th percentile gap, the maximum gap and the number of gaps in ns of
    the trace. Return None when the trace cannot be read or has no gaps to judge."""
    ret = None
    try:
        with csv_path.open("r", newline="") as inp:
            rows = list(csv.DictReader(inp))
        kernels = bubble_gaps.select_graph_kernels(rows)
        gaps = bubble_gaps.launch_gaps(bubble_gaps.group_by_launch(kernels))
    except (OSError, KeyError, ValueError) as e:
        # Not the message of the error: it can contain a row of the trace, and the
        # validate step reports the details.
        print(
            f"bubble check: cannot judge {csv_path} ({type(e).__name__}), "
            f"the validate step reports it",
            flush=True,
        )
    else:
        if gaps:
            p99_9_gap, max_gap = bubble_gaps.outlier_gaps(gaps)
            ret = (p99_9_gap, max_gap, len(gaps))
    return ret


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    limit = bubble_gaps.OUTLIER_THRESHOLD_NS

    # The validate step skips the bubble check without KFD, so there is nothing to retry.
    attempts = args.attempts if bubble_gaps.KFD_AVAILABLE else 1

    ret = 0
    bubble_found = False
    for attempt in range(1, attempts + 1):
        bubble_found = False

        # Remove the previous trace, so that a run that writes none is not judged on it.
        try:
            args.csv.unlink()
        except FileNotFoundError:
            pass

        ret = subprocess.run(args.command).returncode
        if ret != 0 or attempts == 1:
            break

        outlier = find_outlier_gap(args.csv)
        if outlier is None:
            break

        p99_9_gap, max_gap, num_gaps = outlier
        bubble_found = p99_9_gap > limit
        print(
            f"bubble check: attempt {attempt} of {attempts}: 99.9th percentile gap "
            f"{p99_9_gap} ns, max gap {max_gap} ns, {num_gaps} gaps, limit {limit} ns",
            flush=True,
        )
        if not bubble_found:
            break

    if bubble_found:
        print(
            f"bubble check: the limit was exceeded in all {attempts} attempts, "
            f"the validate step reports it",
            flush=True,
        )
    return ret


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
