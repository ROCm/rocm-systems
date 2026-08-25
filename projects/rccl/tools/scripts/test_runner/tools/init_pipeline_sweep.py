#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""
Init-pipeline scale sweep driver (Gate A5).

Runs test_runner.py in init-pipeline mode across --init-pool x --loader-policy
combinations, times each run's wall clock, locates the tests.jsonl it emitted,
and (if a serial baseline is given) reports the per-config correctness diff. Meant
to run on the cluster where the test binaries + GPUs exist.

Example:
  python tools/init_pipeline_sweep.py -c configs/mi300x.json \\
      --pools 1,2,4,6,8 --policies continuous,quiescent_exec \\
      --baseline /path/serial/tests.jsonl --results-root . \\
      --exclude '*_CuMem1' -- --no-build --emit-results

Everything after `--` is passed through to test_runner.py verbatim.
"""

import argparse
import glob
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lib.results_diff import diff_results  # noqa: E402


def _load_jsonl(path):
    import json
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _newest_tests_jsonl(root, since):
    """Newest tests.jsonl under root modified at/after `since` (run start)."""
    best, best_mtime = None, since - 1
    for p in glob.glob(os.path.join(root, "**", "tests.jsonl"), recursive=True):
        try:
            m = os.path.getmtime(p)
        except OSError:
            continue
        if m >= since and m > best_mtime:
            best, best_mtime = p, m
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description="Init-pipeline scale sweep (Gate A5).")
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--pools", default="1,2,4,6,8", help="comma-separated --init-pool values")
    ap.add_argument("--policies", default="continuous", help="comma-separated --loader-policy values")
    ap.add_argument("--baseline", default=None, help="serial tests.jsonl for the per-config diff")
    ap.add_argument("--results-root", default=".", help="dir to search for emitted tests.jsonl")
    ap.add_argument("--exclude", action="append", default=[], help="glob of test names to ignore in the diff")
    ap.add_argument("--runner", default=os.path.join(_ROOT, "test_runner.py"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("passthrough", nargs=argparse.REMAINDER,
                    help="args after -- are forwarded to test_runner.py")
    args = ap.parse_args(argv)

    extra = args.passthrough
    if extra and extra[0] == "--":
        extra = extra[1:]

    baseline_recs = _load_jsonl(args.baseline) if args.baseline else None
    pools = [p.strip() for p in args.pools.split(",") if p.strip()]
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]

    rows = []
    for pool in pools:
        for policy in policies:
            cmd = [args.python, args.runner, "-c", args.config,
                   "--exec-mode", "init-pipeline", "--init-pool", pool,
                   "--loader-policy", policy, "--emit-results", "--phase-timings", *extra]
            print(f"\n=== init_pool={pool} loader_policy={policy} ===\n{' '.join(cmd)}", flush=True)
            start = time.time()
            rc = subprocess.call(cmd)
            wall = time.time() - start

            diff_summary = ""
            if baseline_recs is not None:
                tj = _newest_tests_jsonl(args.results_root, start)
                if tj:
                    d = diff_results(baseline_recs, _load_jsonl(tj), exclude=args.exclude)
                    diff_summary = (f"gate={'PASS' if d['gate_ok'] else 'FAIL'} "
                                    f"reg={len(d['regressions'])} dropped={len(d['only_baseline'])}")
                else:
                    diff_summary = "no tests.jsonl found"
            rows.append((pool, policy, wall, rc, diff_summary))

    print("\n" + "=" * 72)
    print(f"{'init_pool':>9} {'policy':>15} {'wall(s)':>9} {'rc':>3}  diff")
    for pool, policy, wall, rc, ds in rows:
        print(f"{pool:>9} {policy:>15} {wall:>9.1f} {rc:>3}  {ds}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
