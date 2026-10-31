#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Amortization check: the one property that distinguishes range replay from kernel replay.
#
# Kernel replay opens a replay window per dispatch, so a K-dispatch phase pays the fixed window
# cost -- agent drain, snapshot of the tracked inventory, restore between passes -- K times. Range
# replay opens one window for the whole range and pays that cost once. So as the dispatch count per
# range grows against a fixed memory footprint, wall time should grow far more slowly than the
# dispatch count does.
#
# This sweep varies exactly that axis and nothing else: same ballast, same pass count, same range
# count, only the number of dispatches inside each range changes. The workload's dispatches are
# deliberately cheap relative to the snapshot (see kWorkElems in main.cpp), so if the measured wall
# time tracked the dispatch count linearly it would mean the fixed cost is being paid per dispatch
# again -- which is the regression this test exists to catch, and which a pass-count scaling test
# cannot see because it holds the dispatch count constant.
#
# The bound is a ratio between the longest and shortest range in the sweep. Perfect amortization
# puts it near 1.0; per-dispatch windows put it near the dispatch ratio. The default cap sits well
# between the two so that ordinary runner noise does not trip it but a structural regression does.

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
    ap.add_argument("--ballast-mb", type=int, default=64)
    ap.add_argument(
        "--dispatches",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="dispatch counts per range to sweep, ascending",
    )
    ap.add_argument("--ranges", type=int, default=2)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument(
        "--max-amortization-ratio",
        type=float,
        default=3.0,
        help=(
            "cap on wall(max dispatches) / wall(min dispatches). Must be well below the "
            "dispatch ratio itself, or the test cannot distinguish amortized from per-dispatch"
        ),
    )
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    counts = sorted(set(args.dispatches))
    if len(counts) < 2:
        raise SystemExit(f"--dispatches needs at least two distinct values, got {counts}")
    if args.passes < 2:
        raise SystemExit(
            f"--passes must be >= 2 (got {args.passes}): a pass count below 2 is not replayed, "
            "so the sweep would measure the application rather than the replay"
        )

    dispatch_ratio = counts[-1] / counts[0]
    if args.max_amortization_ratio >= dispatch_ratio:
        raise SystemExit(
            f"--max-amortization-ratio ({args.max_amortization_ratio}) must be below the "
            f"dispatch ratio ({dispatch_ratio:.1f}); a cap at or above it would pass even if "
            "the fixed window cost were paid per dispatch, which is the whole point of the test"
        )

    stats = {}
    for k in counts:
        stats[k] = repeat_measure(
            lambda kk=k: run_case(
                args.testapp,
                args.client,
                args.passes,
                args.ballast_mb,
                kk,
                args.ranges,
                args.warmup,
            ),
            repeat=args.repeat,
            warmup=1,
            label=f"K={k}",
        )

    low_ms = stats[counts[0]]["median_ms"]
    high_ms = stats[counts[-1]]["median_ms"]
    ratio = high_ms / max(low_ms, 0.001)

    # Reported for the trend, not asserted on individually: per-dispatch cost should fall as the
    # fixed window cost spreads over more dispatches, and seeing that curve in the log is what
    # makes a borderline ratio interpretable.
    for k in counts:
        per_dispatch = stats[k]["median_ms"] / (k * args.ranges)
        print(
            f"[rr-perf-amort] K={k} median={stats[k]['median_ms']:.1f} ms "
            f"per_dispatch={per_dispatch:.2f} ms"
        )

    assert ratio <= args.max_amortization_ratio, (
        f"amortization ratio {ratio:.2f} > {args.max_amortization_ratio:.2f} "
        f"(K={counts[0]} {low_ms:.1f} ms -> K={counts[-1]} {high_ms:.1f} ms, medians; "
        f"{dispatch_ratio:.1f}x the dispatches). Wall time is tracking the dispatch count, "
        "which means the per-range fixed cost is being paid per dispatch"
    )
    print(
        f"[rr-perf-amort] PASS amortization ratio={ratio:.2f} <= "
        f"{args.max_amortization_ratio:.2f} for {dispatch_ratio:.1f}x the dispatches "
        f"(K={counts[0]} {low_ms:.1f} ms -> K={counts[-1]} {high_ms:.1f} ms, medians)"
    )

    write_results(
        "RR_PERF_AMORT_JSON",
        {
            "ballast_mb": args.ballast_mb,
            "dispatch_counts": counts,
            "ranges": args.ranges,
            "passes": args.passes,
            "repeat": args.repeat,
            "by_dispatch_count": {str(k): stats[k] for k in counts},
            "amortization_ratio": ratio,
            "amortization_cap": args.max_amortization_ratio,
            "dispatch_ratio": dispatch_ratio,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
