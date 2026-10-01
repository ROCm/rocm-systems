#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Pass-count scaling check for range replay: run a fixed range at a baseline pass count and at a
# higher one, and require that wall time grows no faster than the pass count does, with slack.
#
# The baseline is P=2 rather than P=1 for the same reason it is in the kernel replay suite: a pass
# count of 1 means the range is recorded and closed without any re-execution, so a P=1 sample times
# the application rather than the replay, and the ratio against it reports the cost of turning
# replay on at all instead of per-pass scaling. Comparing two replayed configurations keeps the
# once-per-range fixed cost (drain, snapshot, restore) on both sides of the ratio, which is what
# makes the ratio a per-pass measurement.
#
# Because that fixed cost is on both sides and does not scale with P, the expected ratio is well
# below P_high/P_base. The cap is still set from the pass counts with generous slack rather than
# tuned to a machine, because these run on shared runners where the neighbours move the numbers
# more than a modest regression would.

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "perf-common"))

from perf_stats import repeat_measure, write_results  # noqa: E402
from rr_perf_common import run_case  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--testapp", type=Path, required=True)
    ap.add_argument("--client", type=Path, required=True)
    ap.add_argument("--ballast-mb", type=int, default=32)
    ap.add_argument("--dispatches", type=int, default=4)
    ap.add_argument("--ranges", type=int, default=2)
    ap.add_argument(
        "--base-passes",
        type=int,
        default=2,
        help="baseline pass count; must be >= 2 so the baseline is actually replayed",
    )
    ap.add_argument("--high-passes", type=int, default=5)
    ap.add_argument("--max-scaling-ratio", type=float, default=8.0)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    if args.base_passes < 2:
        raise SystemExit(
            f"--base-passes must be >= 2 (got {args.base_passes}): a pass count below 2 is not "
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
                args.dispatches,
                args.ranges,
                args.warmup,
            ),
            repeat=args.repeat,
            warmup=1,
            label=f"P={passes}",
        )

    base_ms = stats[args.base_passes]["median_ms"]
    high_ms = stats[args.high_passes]["median_ms"]
    ratio = high_ms / max(base_ms, 0.001)
    # Slack of 2x over the ideal pass ratio, and never above the caller's absolute cap.
    cap = min(args.max_scaling_ratio, 2.0 * args.high_passes / args.base_passes)

    assert ratio <= cap, (
        f"pass scaling ratio {ratio:.2f} > {cap:.2f} "
        f"(P={args.base_passes} {base_ms:.1f} ms vs "
        f"P={args.high_passes} {high_ms:.1f} ms, medians)"
    )
    print(
        f"[rr-perf-run] PASS scaling ratio={ratio:.2f} <= {cap:.2f} "
        f"(P={args.base_passes} {base_ms:.1f} ms -> "
        f"P={args.high_passes} {high_ms:.1f} ms, medians)"
    )

    write_results(
        "RR_PERF_RESULTS_JSON",
        {
            "ballast_mb": args.ballast_mb,
            "dispatches": args.dispatches,
            "ranges": args.ranges,
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
