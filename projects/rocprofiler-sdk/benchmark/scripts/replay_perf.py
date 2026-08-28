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

"""Kernel replay performance harness. Operator-run, not part of CI.

Measures the claims the design rests on rather than asserting a threshold, and
compares the three ways of collecting several counter groups:

  application replay  one application run per group, costs the sum of them
  multiplexed         one run, groups rotated across dispatches, so no dispatch
                      carries every group
  kernel replay       one run, every dispatch replayed once per group

  P1  application replay vs multiplexed vs kernel replay, for the same groups
  P2  scaling in pass count
  P3  scaling in tracked device footprint, and the host RSS that comes with it
  P6  the cost imposed on NON-replayed dispatches while replay is merely active

P5 (drain cost as a fraction of a window) needs internal timers rather than wall
clock, so it is left to ROCP_INFO log analysis.

Every measurement is a median over --repeats runs taken after --warmup untimed
runs, which is the same convention the CI performance tests use: the first run
of an application pays for page cache misses and code object loading, and
folding that into the measurement makes a fast configuration look slow.

Writing the results is inside the measurement, and there are G times as many
counter records to write with G groups, so the output format is part of what is
being compared rather than a detail. It defaults to rocpd, which is what an
unqualified rocprofv3 invocation uses. Note that rocpd does not record which
replay pass a counter value came from; --output-format json does.

Writes one CSV row per measurement to --output. Every row records the full
command so a number can be traced back to what produced it.

  ./replay_perf.py --rocprofv3 /opt/rocm/bin/rocprofv3 --app ./kernel-replay \\
      --output perf.csv --repeats 5
"""

import argparse
import csv
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
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

REPLAY_FLAG = "--kernel-replay-beta-enabled"

# vocabulary shared with the counter_collection_mode column of the benchmark
# database, so a CSV row here and a row there describe a run the same way
MODE_SINGLE_PASS = "single-pass"
MODE_MULTIPLEXED = "multiplexed"
MODE_KERNEL_REPLAY = "kernel-replay"
MODE_APPLICATION_REPLAY = "application-replay"

CSV_FIELDS = [
    "experiment",
    "variant",
    "mode",
    "detail",
    "median_s",
    "min_s",
    "max_s",
    "peak_child_rss_kib",
    "samples",
    "failures",
    "command",
]


def _pmc_args(groups):
    args = []
    for unique in groups:
        args += ["--pmc"] + COMMON + [unique]
    return args


def _profile_cmd(args, groups, outdir, replay=False, extra=None):
    return (
        [args.rocprofv3]
        + _pmc_args(groups)
        + ([REPLAY_FLAG] if replay else [])
        + (extra or [])
        + [
            "--output-format",
            args.output_format,
            "-d",
            f"{args.workdir}/{outdir}",
            "-o",
            "out",
            "--",
        ]
        + args.app_cmd
    )


def _exit_code(status):
    if os.WIFSIGNALED(status):
        return -os.WTERMSIG(status)
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    return status


def _run(cmd, env=None):
    """Run one command and return (wall_seconds, peak_child_rss_kib, returncode).

    os.wait4 reports the resource usage of this child alone. Reading peak RSS
    from getrusage(RUSAGE_CHILDREN) instead would report the high water mark
    across every child reaped so far, which never decreases, so the first large
    run would set the figure reported for all of the smaller runs after it.
    """

    with tempfile.TemporaryFile() as errfile:
        start = time.perf_counter()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=errfile,
            env={**os.environ, **(env or {})},
        )
        _, status, usage = os.wait4(proc.pid, 0)
        wall = time.perf_counter() - start

        # the child is already reaped, so tell Popen not to wait for it again
        returncode = _exit_code(status)
        proc.returncode = returncode

        if returncode != 0:
            errfile.seek(0)
            tail = errfile.read().decode(errors="replace")[-2000:]
            sys.stderr.write(
                f"  command failed rc={returncode}: {' '.join(cmd)}\n  {tail}\n"
            )

    return wall, usage.ru_maxrss, returncode


def _repeat(cmd, repeats, warmup=1, env=None):
    """Median of `repeats` timed runs, after `warmup` untimed ones."""

    for _ in range(max(0, warmup)):
        _run(cmd, env)

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

    def add(self, experiment, variant, mode, detail, cmd, stats):
        row = {
            "experiment": experiment,
            "variant": variant,
            "mode": mode,
            "detail": detail,
            "command": " ".join(cmd),
        }

        if stats is None:
            print(f"  {experiment}/{variant}: FAILED (no successful run)")
            row.update(
                {
                    "median_s": "",
                    "min_s": "",
                    "max_s": "",
                    "peak_child_rss_kib": "",
                    "samples": 0,
                    "failures": "all",
                }
            )
        else:
            print(
                f"  {experiment}/{variant}: median {stats['median_s']:.3f}s "
                f"(min {stats['min_s']:.3f} max {stats['max_s']:.3f}, "
                f"n={stats['samples']})"
            )
            row.update(stats)

        self.rows.append(row)
        return row

    def write(self):
        if not self.rows:
            return

        with open(self.path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)

        print(f"\nwrote {len(self.rows)} rows to {self.path}")


def p1_collection_modes(args, rec):
    """The headline claim: G groups collected three different ways."""

    print("\nP1 application replay vs multiplexed vs kernel replay")
    groups = UNIQUE[: args.groups]

    total = 0.0
    complete = True
    for idx, unique in enumerate(groups):
        cmd = _profile_cmd(args, [unique], f"p1-base-{idx}")
        stats = _repeat(cmd, args.repeats, args.warmup)
        rec.add(
            "P1", f"application-replay-run-{idx}", MODE_SINGLE_PASS, unique, cmd, stats
        )
        if stats is None:
            complete = False
        else:
            total += stats["median_s"]

    if complete:
        print(f"  application replay total ({len(groups)} runs): {total:.3f}s")

    cmd = _profile_cmd(args, groups, "p1-multiplexed")
    rec.add(
        "P1",
        "multiplexed-all-groups",
        MODE_MULTIPLEXED,
        f"{len(groups)} groups rotated across dispatches, one run",
        cmd,
        _repeat(cmd, args.repeats, args.warmup),
    )

    cmd = _profile_cmd(args, groups, "p1-replay", replay=True)
    stats = _repeat(cmd, args.repeats, args.warmup)
    rec.add(
        "P1",
        "replay-all-groups",
        MODE_KERNEL_REPLAY,
        f"{len(groups)} groups, one run",
        cmd,
        stats,
    )

    if stats and complete and stats["median_s"] > 0:
        print(f"  speedup vs application replay: {total / stats['median_s']:.2f}x")


def p2_pass_count_scaling(args, rec):
    """Marginal cost per pass. Uses 1..G groups, so pass count tracks group count."""

    print("\nP2 pass-count scaling")
    for count in range(1, args.groups + 1):
        groups = UNIQUE[:count]
        replay = count > 1
        cmd = _profile_cmd(args, groups, f"p2-{count}", replay=replay)
        rec.add(
            "P2",
            f"passes-{count}",
            MODE_KERNEL_REPLAY if replay else MODE_SINGLE_PASS,
            f"{count} group(s)",
            cmd,
            _repeat(cmd, args.repeats, args.warmup),
        )


def p3_p4_footprint_scaling(args, rec):
    """Snapshot cost and host RSS against tracked device footprint.

    Requires an app that takes a working-set size; the kernel-replay test binary
    takes an element count as its first argument.
    """

    print("\nP3/P4 footprint scaling")
    groups = UNIQUE[: args.groups]
    for elements in args.footprints:
        app_cmd = args.app_cmd
        args.app_cmd = [app_cmd[0], str(elements)] + app_cmd[2:]
        try:
            cmd = _profile_cmd(args, groups, f"p3-{elements}", replay=True)
            rec.add(
                "P3",
                f"elements-{elements}",
                MODE_KERNEL_REPLAY,
                f"{elements} elements",
                cmd,
                _repeat(cmd, args.repeats, args.warmup),
            )
        finally:
            args.app_cmd = app_cmd


def p6_non_replayed_dispatch_tax(args, rec):
    """What replay costs dispatches that opt out.

    Replay active but every dispatch opting out is the realistic shape: one hot
    kernel replayed in a large application. The reader lock is taken per dispatch
    whenever a replay service is configured, so this isolates that overhead.
    An unmatchable --kernel-include-regex gives replay-active-but-never-triggered.
    """

    print("\nP6 tax on non-replayed dispatches")
    groups = UNIQUE[: args.groups]

    baseline = _profile_cmd(args, [groups[0]], "p6-baseline")
    rec.add(
        "P6",
        "no-replay",
        MODE_SINGLE_PASS,
        "single group, replay not configured",
        baseline,
        _repeat(baseline, args.repeats, args.warmup),
    )

    filtered = _profile_cmd(
        args,
        groups,
        "p6-filtered",
        replay=True,
        extra=["--kernel-include-regex", "^__no_such_kernel__$"],
    )
    rec.add(
        "P6",
        "replay-active-none-matched",
        MODE_KERNEL_REPLAY,
        "replay configured, no kernel matches",
        filtered,
        _repeat(filtered, args.repeats, args.warmup),
    )


EXPERIMENTS = {
    "p1": p1_collection_modes,
    "p2": p2_pass_count_scaling,
    "p3": p3_p4_footprint_scaling,
    "p6": p6_non_replayed_dispatch_tax,
}


def parse_args(argv=None):
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
    parser.add_argument(
        "--output-format",
        default="rocpd",
        choices=("rocpd", "json", "csv"),
        help="format rocprofv3 writes its results in (default: rocpd). Writing the results is "
        "part of what is being timed and the volume of them grows with the group count, so this "
        "defaults to what an unqualified rocprofv3 invocation uses. Pass json to inspect the "
        "replay_pass field afterwards; it is the only format that carries it.",
    )
    parser.add_argument("--workdir", default="replay_perf_out")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="untimed runs before the timed ones (default: 1)",
    )
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

    args = parser.parse_args(argv)

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")

    return args


def main(argv=None):
    args = parse_args(argv)

    if args.dry_run:
        print(f"rocprofv3: {args.rocprofv3}")
        print(f"app:       {' '.join(args.app_cmd)}")
        print(f"format:    {args.output_format}")
        print(f"repeats:   {args.repeats}   warmup: {args.warmup}")
        print(f"groups:    {args.groups}")
        print(f"experiments: {', '.join(args.only)}")
        print(f"footprints:  {args.footprints}")
        return 0

    if not shutil.which(args.rocprofv3) and not os.path.isfile(args.rocprofv3):
        sys.stderr.write(f"rocprofv3 not found: {args.rocprofv3}\n")
        return 2
    if not os.path.isfile(args.app_cmd[0]):
        sys.stderr.write(f"application not found: {args.app_cmd[0]}\n")
        return 2

    os.makedirs(args.workdir, exist_ok=True)

    rec = Recorder(args.output)
    for name in args.only:
        EXPERIMENTS[name](args, rec)
    rec.write()
    return 0


if __name__ == "__main__":
    sys.exit(main())
