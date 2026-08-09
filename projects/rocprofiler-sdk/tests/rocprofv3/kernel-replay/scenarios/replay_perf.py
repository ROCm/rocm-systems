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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Kernel-replay performance harness. Operator-run, not part of CI.

Measures the claims the design rests on, rather than asserting a threshold:

  P1  relaunch (one application run per counter group) vs replay (one run, all groups)
  P2  scaling in pass count
  P3  scaling in tracked device footprint
  P4  peak host RSS against snapshot size
  P6  the cost imposed on NON-replayed dispatches while replay is merely active

P5 (drain cost as a fraction of a window) needs internal timers rather than wall clock,
so it is left to ROCP_INFO log analysis.

Writes one CSV row per measurement to --output. Every row records the full command so a
number can be traced back to what produced it.

  ./replay_perf.py --rocprofv3 /opt/rocm/bin/rocprofv3 --app ./kernel-replay \\
      --output perf.csv --repeats 5
"""

import argparse
import csv
import os
import resource
import shutil
import statistics
import subprocess
import sys
import time

# Each group shares the sanity counters and adds one unique counter, matching the CI test.
COMMON = ["SQ_WAVES", "SQ_INSTS_VALU"]
UNIQUE = [
    "GRBM_COUNT",
    "GRBM_GUI_ACTIVE",
    "SQ_INSTS_SALU",
    "SQ_INSTS_SMEM",
    "SQ_INSTS_LDS",
]


def _pmc_args(groups):
    args = []
    for unique in groups:
        args += ["--pmc"] + COMMON + [unique]
    return args


def _run(cmd, env=None):
    """Return (wall_seconds, peak_child_rss_kib, returncode)."""
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env={**os.environ, **(env or {})},
        check=False,
    )
    wall = time.perf_counter() - start
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if proc.returncode != 0:
        sys.stderr.write(
            f"  command failed rc={proc.returncode}: {' '.join(cmd)}\n"
            f"  {proc.stderr.decode(errors='replace')[-2000:]}\n"
        )
    return wall, max(after, before), proc.returncode


def _repeat(cmd, repeats, env=None):
    walls, rss, failures = [], 0, 0
    for _ in range(repeats):
        wall, peak, rc = _run(cmd, env)
        if rc != 0:
            failures += 1
            continue
        walls.append(wall)
        rss = max(rss, peak)
    if not walls:
        return None
    return {
        "median_s": statistics.median(walls),
        "min_s": min(walls),
        "max_s": max(walls),
        "peak_child_rss_kib": rss,
        "failures": failures,
        "samples": len(walls),
    }


class Recorder:
    def __init__(self, path):
        self.path = path
        self.rows = []

    def add(self, experiment, variant, detail, cmd, stats):
        if stats is None:
            print(f"  {experiment}/{variant}: FAILED (no successful run)")
            self.rows.append(
                {
                    "experiment": experiment,
                    "variant": variant,
                    "detail": detail,
                    "median_s": "",
                    "min_s": "",
                    "max_s": "",
                    "peak_child_rss_kib": "",
                    "samples": 0,
                    "failures": "all",
                    "command": " ".join(cmd),
                }
            )
            return
        print(
            f"  {experiment}/{variant}: median {stats['median_s']:.3f}s "
            f"(min {stats['min_s']:.3f} max {stats['max_s']:.3f}, n={stats['samples']})"
        )
        self.rows.append(
            {
                "experiment": experiment,
                "variant": variant,
                "detail": detail,
                **{
                    k: stats[k]
                    for k in (
                        "median_s",
                        "min_s",
                        "max_s",
                        "peak_child_rss_kib",
                        "samples",
                        "failures",
                    )
                },
                "command": " ".join(cmd),
            }
        )

    def write(self):
        if not self.rows:
            return
        with open(self.path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"\nwrote {len(self.rows)} rows to {self.path}")


def p1_relaunch_vs_replay(args, rec):
    """The headline claim: G groups in one run vs G application runs."""
    print("\nP1 relaunch vs replay")
    groups = UNIQUE[: args.groups]
    total = 0.0
    for idx, unique in enumerate(groups):
        cmd = (
            [args.rocprofv3]
            + _pmc_args([unique])
            + [
                "--output-format",
                "json",
                "-d",
                f"{args.workdir}/p1-base-{idx}",
                "-o",
                "out",
                "--",
            ]
            + args.app_cmd
        )
        stats = _repeat(cmd, args.repeats)
        rec.add("P1", f"relaunch-group-{idx}", unique, cmd, stats)
        if stats:
            total += stats["median_s"]
    print(f"  relaunch total (sum of {len(groups)} runs): {total:.3f}s")

    cmd = (
        [args.rocprofv3]
        + _pmc_args(groups)
        + [
            "--kernel-replay-beta-enabled",
            "--output-format",
            "json",
            "-d",
            f"{args.workdir}/p1-replay",
            "-o",
            "out",
            "--",
        ]
        + args.app_cmd
    )
    stats = _repeat(cmd, args.repeats)
    rec.add("P1", "replay-all-groups", f"{len(groups)} groups, one run", cmd, stats)
    if stats and total:
        print(f"  speedup vs relaunch: {total / stats['median_s']:.2f}x")


def p2_pass_count_scaling(args, rec):
    """Marginal cost per pass. Uses 1..G groups, so pass count tracks group count."""
    print("\nP2 pass-count scaling")
    for count in range(1, args.groups + 1):
        groups = UNIQUE[:count]
        cmd = (
            [args.rocprofv3]
            + _pmc_args(groups)
            + (["--kernel-replay-beta-enabled"] if count > 1 else [])
            + [
                "--output-format",
                "json",
                "-d",
                f"{args.workdir}/p2-{count}",
                "-o",
                "out",
                "--",
            ]
            + args.app_cmd
        )
        rec.add(
            "P2", f"passes-{count}", f"{count} group(s)", cmd, _repeat(cmd, args.repeats)
        )


def p3_p4_footprint_scaling(args, rec):
    """Snapshot cost and host RSS against tracked device footprint.

    Requires an app that takes a working-set size; the kernel-replay test binary takes an
    element count as its first argument.
    """
    print("\nP3/P4 footprint scaling")
    groups = UNIQUE[: args.groups]
    for elements in args.footprints:
        app = [args.app_cmd[0], str(elements)] + args.app_cmd[2:]
        cmd = (
            [args.rocprofv3]
            + _pmc_args(groups)
            + [
                "--kernel-replay-beta-enabled",
                "--output-format",
                "json",
                "-d",
                f"{args.workdir}/p3-{elements}",
                "-o",
                "out",
                "--",
            ]
            + app
        )
        rec.add(
            "P3",
            f"elements-{elements}",
            f"{elements} elements",
            cmd,
            _repeat(cmd, args.repeats),
        )


def p6_non_replayed_dispatch_tax(args, rec):
    """What replay costs dispatches that opt out.

    Replay active but every dispatch opting out is the realistic shape: one hot kernel
    replayed in a large application. The reader lock is taken per dispatch whenever a
    replay service is configured, so this isolates that overhead. Requires the app to be
    filtered so nothing matches; --kernel-include-regex with an unmatchable pattern gives
    replay-active-but-never-triggered.
    """
    print("\nP6 tax on non-replayed dispatches")
    groups = UNIQUE[: args.groups]
    baseline = (
        [args.rocprofv3]
        + _pmc_args([groups[0]])
        + [
            "--output-format",
            "json",
            "-d",
            f"{args.workdir}/p6-baseline",
            "-o",
            "out",
            "--",
        ]
        + args.app_cmd
    )
    rec.add(
        "P6",
        "no-replay",
        "single group, replay not configured",
        baseline,
        _repeat(baseline, args.repeats),
    )

    filtered = (
        [args.rocprofv3]
        + _pmc_args(groups)
        + [
            "--kernel-replay-beta-enabled",
            "--kernel-include-regex",
            "^__no_such_kernel__$",
            "--output-format",
            "json",
            "-d",
            f"{args.workdir}/p6-filtered",
            "-o",
            "out",
            "--",
        ]
        + args.app_cmd
    )
    rec.add(
        "P6",
        "replay-active-none-matched",
        "replay configured, no kernel matches",
        filtered,
        _repeat(filtered, args.repeats),
    )


EXPERIMENTS = {
    "p1": p1_relaunch_vs_replay,
    "p2": p2_pass_count_scaling,
    "p3": p3_p4_footprint_scaling,
    "p6": p6_non_replayed_dispatch_tax,
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rocprofv3", default=shutil.which("rocprofv3") or "rocprofv3")
    parser.add_argument(
        "--app",
        dest="app_cmd",
        nargs="+",
        required=True,
        help="application and arguments, e.g. --app ./kernel-replay 1048576 1. "
        "Arguments beginning with a dash are not supported here.",
    )
    parser.add_argument("--output", default="replay_perf.csv")
    parser.add_argument("--workdir", default="replay_perf_out")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--groups", type=int, default=len(UNIQUE), choices=range(1, len(UNIQUE) + 1)
    )
    parser.add_argument(
        "--footprints", type=int, nargs="+", default=[1 << 18, 1 << 20, 1 << 22, 1 << 24]
    )
    parser.add_argument(
        "--only", nargs="+", choices=sorted(EXPERIMENTS), default=sorted(EXPERIMENTS)
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = parser.parse_args()

    if not args.app_cmd:
        parser.error("--app requires a command")
    os.makedirs(args.workdir, exist_ok=True)

    if args.dry_run:
        print(f"rocprofv3: {args.rocprofv3}")
        print(f"app:       {' '.join(args.app_cmd)}")
        print(f"repeats:   {args.repeats}   groups: {args.groups}")
        print(f"experiments: {', '.join(args.only)}")
        print(f"footprints:  {args.footprints}")
        return 0

    if not shutil.which(args.rocprofv3) and not os.path.isfile(args.rocprofv3):
        sys.stderr.write(f"rocprofv3 not found: {args.rocprofv3}\n")
        return 2
    if not os.path.isfile(args.app_cmd[0]):
        sys.stderr.write(f"application not found: {args.app_cmd[0]}\n")
        return 2

    rec = Recorder(args.output)
    for name in args.only:
        EXPERIMENTS[name](args, rec)
    rec.write()
    return 0


if __name__ == "__main__":
    sys.exit(main())
