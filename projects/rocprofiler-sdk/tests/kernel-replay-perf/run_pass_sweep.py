#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Pass-count sweep regression: replay wall time must grow roughly linearly with the pass count,
# not super-linearly. Catches changes that add per-pass fixed overhead or break restore
# amortization.
#
# Each pass count is sampled several times after a warmup and compared on the median. Only the
# relative (shape) checks gate CI; the absolute cost-model ceiling is advisory unless
# ROCPROFILER_PERF_STRICT_CEILING is set.

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "perf-common"))

from perf_cost_model import max_pass_scaling_ratio, model_max_ms
from perf_stats import check_ceiling, parse_marker, repeat_measure, write_results


def run_case(
    testapp: Path,
    client: Path,
    passes: int,
    ballast_mb: int,
    launches: int,
    warmup: int,
) -> float:
    env = os.environ.copy()
    env["KR_PERF_PASSES"] = str(passes)
    preload = client.resolve()
    if env.get("LD_PRELOAD"):
        env["LD_PRELOAD"] = f"{preload}:{env['LD_PRELOAD']}"
    else:
        env["LD_PRELOAD"] = str(preload)

    proc = subprocess.run(
        [str(testapp.resolve()), str(ballast_mb), str(launches), str(warmup)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    out = proc.stdout + proc.stderr
    if proc.returncode != 0 or "[kr-perf] PASS" not in out:
        raise RuntimeError(f"P={passes} failed rc={proc.returncode}\n{out}")
    return float(parse_marker(out, "kr-perf")["wall_ms"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--testapp", type=Path, required=True)
    ap.add_argument("--client", type=Path, required=True)
    ap.add_argument("--ballast-mb", type=int, default=32)
    ap.add_argument("--launches", type=int, default=4)
    # Starts at 2, not 1: a pass count of 1 is not replayed (no snapshot, no restore), so including
    # it makes the first pairwise ratio measure the cost of enabling replay rather than per-pass
    # scaling. See the header comment in run_and_validate.py.
    ap.add_argument("--passes", type=int, nargs="+", default=[2, 3, 5, 8])
    ap.add_argument(
        "--slack",
        type=float,
        default=2.0,
        help="multiplier over ideal linear pass scaling",
    )
    ap.add_argument("--repeat", type=int, default=3, help="timed samples per pass count")
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    stats: dict = {}
    for p in args.passes:
        stats[p] = repeat_measure(
            lambda passes=p: run_case(
                args.testapp,
                args.client,
                passes,
                args.ballast_mb,
                args.launches,
                args.warmup,
            ),
            repeat=args.repeat,
            warmup=1,
            label=f"P={p}",
        )
        check_ceiling(
            stats[p]["median_ms"],
            model_max_ms(args.ballast_mb, args.launches, p),
            f"P={p}",
        )

    def median(p: int) -> float:
        return stats[p]["median_ms"]

    ordered = sorted(args.passes)
    for i in range(1, len(ordered)):
        p_lo, p_hi = ordered[i - 1], ordered[i]
        ratio = median(p_hi) / max(median(p_lo), 0.001)
        cap = max_pass_scaling_ratio(p_lo, p_hi, args.slack)
        assert (
            ratio <= cap
        ), f"pass sweep super-linear: P={p_lo}->{p_hi} ratio={ratio:.2f} > {cap:.2f}"
        print(f"[kr-perf-sweep] PASS P={p_lo}->{p_hi} ratio={ratio:.2f} <= {cap:.2f}")

    overall = median(ordered[-1]) / max(median(ordered[0]), 0.001)
    overall_cap = max_pass_scaling_ratio(ordered[0], ordered[-1], args.slack)
    assert overall <= overall_cap, (
        f"overall sweep ratio {overall:.2f} > {overall_cap:.2f} "
        f"(P={ordered[0]}..{ordered[-1]})"
    )
    print(f"[kr-perf-sweep] PASS overall ratio={overall:.2f} <= {overall_cap:.2f}")

    write_results(
        "KR_PERF_SWEEP_JSON",
        {
            "ballast_mb": args.ballast_mb,
            "launches": args.launches,
            "repeat": args.repeat,
            "per_pass": {str(p): stats[p] for p in ordered},
            "overall_ratio": overall,
            "overall_cap": overall_cap,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
