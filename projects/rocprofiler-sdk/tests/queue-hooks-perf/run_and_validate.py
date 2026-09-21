#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Runs the dispatch-counter perf workload under LD_PRELOAD at two launch counts and checks that
# per-dispatch cost stays flat as the count grows.
#
# Each launch count is sampled several times after a warmup and compared on the median. The
# relative check gates CI; the absolute cost-model ceiling is advisory unless
# ROCPROFILER_PERF_STRICT_CEILING is set.

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "perf-common"))

from perf_cost_model import max_launch_scaling_ratio, model_max_ms
from perf_stats import check_ceiling, parse_marker, repeat_measure, write_results


def run_case(
    testapp: Path, preload: Path, ballast_mb: int, launches: int, warmup: int
) -> float:
    env = os.environ.copy()
    preload = preload.resolve()
    if env.get("LD_PRELOAD"):
        env["LD_PRELOAD"] = f"{preload}:{env['LD_PRELOAD']}"
    else:
        env["LD_PRELOAD"] = str(preload)
    env.setdefault("ROCPROFILER_TOOL_CONTEXTS", "COUNTER_COLLECTION")
    env.setdefault("ROCPROF_COUNTERS", "SQ_WAVES_sum")
    env["ROCPROFILER_TOOL_OUTPUT_FILE"] = os.environ.get(
        "QH_PERF_JSON", f"/tmp/qh_perf_{launches}.json"
    )

    proc = subprocess.run(
        [str(testapp.resolve()), str(ballast_mb), str(launches), str(warmup)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    out = proc.stdout + proc.stderr
    if proc.returncode != 0 or "[qh-perf] PASS" not in out:
        raise RuntimeError(f"launches={launches} failed rc={proc.returncode}\n{out}")
    return float(parse_marker(out, "qh-perf")["wall_ms"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--testapp", type=Path, required=True)
    ap.add_argument(
        "--preload", type=Path, required=True, help="rocprofiler-sdk-json-tool .so"
    )
    ap.add_argument("--ballast-mb", type=int, default=8)
    ap.add_argument("--launches-low", type=int, default=8)
    ap.add_argument("--launches-high", type=int, default=32)
    ap.add_argument("--max-scaling-ratio", type=float, default=6.0)
    ap.add_argument(
        "--repeat", type=int, default=3, help="timed samples per configuration"
    )
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    stats = {}
    for launches in (args.launches_low, args.launches_high):
        stats[launches] = repeat_measure(
            lambda n=launches: run_case(
                args.testapp, args.preload, args.ballast_mb, n, args.warmup
            ),
            repeat=args.repeat,
            warmup=1,
            label=f"L={launches}",
        )
        check_ceiling(
            stats[launches]["median_ms"], model_max_ms(launches), f"L={launches}"
        )

    low_ms = stats[args.launches_low]["median_ms"]
    high_ms = stats[args.launches_high]["median_ms"]
    ratio = high_ms / max(low_ms, 0.001)
    cap = min(
        args.max_scaling_ratio,
        max_launch_scaling_ratio(args.launches_low, args.launches_high, 2.0),
    )
    assert ratio <= cap, (
        f"scaling ratio {ratio:.2f} > {cap:.2f} "
        f"(L={args.launches_low} {low_ms:.1f} ms vs "
        f"L={args.launches_high} {high_ms:.1f} ms, medians)"
    )
    print(f"[qh-perf-run] PASS scaling ratio={ratio:.2f} <= {cap:.2f}")

    write_results(
        "QH_PERF_RESULTS_JSON",
        {
            # What was actually preloaded, rather than a label nothing sets. The earlier
            # QH_PERF_VARIANT knob was never assigned by the build, so every result recorded
            # itself as "unknown"; the preload path is the thing that distinguishes runs.
            "preload": str(args.preload.resolve()),
            "ballast_mb": args.ballast_mb,
            "launches_low": args.launches_low,
            "launches_high": args.launches_high,
            "repeat": args.repeat,
            "low": stats[args.launches_low],
            "high": stats[args.launches_high],
            "scaling_ratio": ratio,
            "scaling_cap": cap,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
