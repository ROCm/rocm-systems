#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Runs the kernel-replay perf workload under LD_PRELOAD for P=1 and P=N and checks that replay
# scales with the pass count rather than blowing up.
#
# Each configuration is sampled several times after a warmup and compared on the median, because
# a single timed run on a shared CI runner is not a measurement. The relative check (P=N against
# P=1) is what gates CI; the absolute cost-model ceiling is advisory unless
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

    stats = {}
    for label, passes in (("P=1", 1), (f"P={args.high_passes}", args.high_passes)):
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
            label=label,
        )
        ceiling = model_max_ms(args.ballast_mb, args.launches, passes)
        check_ceiling(stats[passes]["median_ms"], ceiling, label)

    base_ms = stats[1]["median_ms"]
    high_ms = stats[args.high_passes]["median_ms"]
    ratio = high_ms / max(base_ms, 0.001)
    cap = min(args.max_scaling_ratio, max_pass_scaling_ratio(1, args.high_passes, 2.0))
    assert ratio <= cap, (
        f"scaling ratio {ratio:.2f} > {cap:.2f} "
        f"(P=1 {base_ms:.1f} ms vs P={args.high_passes} {high_ms:.1f} ms, medians)"
    )
    print(
        f"[kr-perf-run] PASS scaling ratio={ratio:.2f} <= {cap:.2f} "
        f"(P=1 {base_ms:.1f} ms -> P={args.high_passes} {high_ms:.1f} ms, medians)"
    )

    write_results(
        "KR_PERF_RESULTS_JSON",
        {
            "ballast_mb": args.ballast_mb,
            "launches": args.launches,
            "passes_high": args.high_passes,
            "repeat": args.repeat,
            "p1": stats[1],
            "pN": stats[args.high_passes],
            "scaling_ratio": ratio,
            "scaling_cap": cap,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
