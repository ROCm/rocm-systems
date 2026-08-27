#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Runs the kernel-replay perf workload under LD_PRELOAD for a baseline pass count and P=N and
# checks that replay scales with the pass count rather than blowing up.
#
# The baseline defaults to P=2, not P=1, on purpose. A pass_count_cb returning 1 means the dispatch
# is NOT replayed: it takes the ordinary single-dispatch path with no snapshot and no restore (see
# experimental/kernel_replay.h). Timing P=1 therefore times a bare dispatch, and a P=N/P=1 ratio is
# dominated by the one-time cost of turning replay on at all rather than by the pass count -- orders
# of magnitude, which no linear-scaling cap can express. Comparing two replayed configurations keeps
# the snapshot/restore fixed cost on both sides, so the ratio measures per-pass scaling, which is
# what a regression would move.
#
# Alternatives considered and rejected: (b) keep the P=1 baseline and make the scaling check
# advisory behind ROCPROFILER_PERF_STRICT_CEILING, the way the absolute ceiling already works --
# rejected because it removes the only per-commit scaling signal; (c) keep P=1 and raise the caps
# to accommodate that fixed cost -- rejected because it bakes a machine- and build-type-specific
# constant into the gate and would not catch a real per-pass regression hiding under it.
#
# The caps here are deliberately loose rather than tuned to any one machine or build type; ratios
# differ between Debug and Release. See
# docs/conceptual/kernel_replay/kernel_replay_performance.md.
#
# Each configuration is sampled several times after a warmup and compared on the median, because
# a single timed run on a shared CI runner is not a measurement. The relative check (P=N against
# the baseline) is what gates CI; the absolute cost-model ceiling is advisory unless
# ROCPROFILER_PERF_STRICT_CEILING is set, since an absolute wall-time bound on a shared machine
# mostly measures the neighbours.

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
    """One timed run of the workload. Returns the in-app wall time in milliseconds."""
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
    if proc.returncode != 0:
        raise RuntimeError(f"P={passes} run failed rc={proc.returncode}\n{out}")
    if "[kr-perf] PASS" not in out:
        raise RuntimeError(f"P={passes} missing PASS marker\n{out}")
    return float(parse_marker(out, "kr-perf")["wall_ms"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--testapp", type=Path, required=True)
    ap.add_argument("--client", type=Path, required=True)
    ap.add_argument("--ballast-mb", type=int, default=64)
    ap.add_argument("--launches", type=int, default=8)
    ap.add_argument("--high-passes", type=int, default=5)
    ap.add_argument(
        "--base-passes",
        type=int,
        default=2,
        help="baseline pass count; must be >= 2 so the baseline is actually replayed",
    )
    ap.add_argument("--max-scaling-ratio", type=float, default=8.0)
    ap.add_argument(
        "--repeat", type=int, default=3, help="timed samples per configuration"
    )
    ap.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="in-app untimed dispatches before each timed loop",
    )
    args = ap.parse_args()

    if args.base_passes < 2:
        raise SystemExit(
            f"--base-passes must be >= 2 (got {args.base_passes}): a pass count of 1 is not "
            "replayed, so it cannot serve as a replay baseline"
        )
    if args.high_passes <= args.base_passes:
        raise SystemExit(
            f"--high-passes ({args.high_passes}) must exceed "
            f"--base-passes ({args.base_passes})"
        )

    stats = {}
    for passes in (args.base_passes, args.high_passes):
        stats[passes] = repeat_measure(
            lambda p=passes: run_case(
                args.testapp,
                args.client,
                p,
                args.ballast_mb,
                args.launches,
                args.warmup,
            ),
            repeat=args.repeat,
            # The in-app warmup already covers first-dispatch effects, so only one extra
            # process is spent on warming the host side (file cache, code-object load).
            warmup=1,
            label=f"P={passes}",
        )
        ceiling = model_max_ms(args.ballast_mb, args.launches, passes)
        check_ceiling(stats[passes]["median_ms"], ceiling, f"P={passes}")

    base_ms = stats[args.base_passes]["median_ms"]
    high_ms = stats[args.high_passes]["median_ms"]
    ratio = high_ms / max(base_ms, 0.001)
    cap = min(
        args.max_scaling_ratio,
        max_pass_scaling_ratio(args.base_passes, args.high_passes, 2.0),
    )
    assert ratio <= cap, (
        f"scaling ratio {ratio:.2f} > {cap:.2f} "
        f"(P={args.base_passes} {base_ms:.1f} ms vs "
        f"P={args.high_passes} {high_ms:.1f} ms, medians)"
    )
    print(
        f"[kr-perf-run] PASS scaling ratio={ratio:.2f} <= {cap:.2f} "
        f"(P={args.base_passes} {base_ms:.1f} ms -> "
        f"P={args.high_passes} {high_ms:.1f} ms, medians)"
    )

    write_results(
        "KR_PERF_RESULTS_JSON",
        {
            "ballast_mb": args.ballast_mb,
            "launches": args.launches,
            "passes_base": args.base_passes,
            "passes_high": args.high_passes,
            "repeat": args.repeat,
            "p_base": stats[args.base_passes],
            "p_high": stats[args.high_passes],
            "scaling_ratio": ratio,
            "scaling_cap": cap,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
