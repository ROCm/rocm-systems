#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Launch-count sweep for dispatch-counter interposition overhead. Per-dispatch cost should stay
# flat as the launch count grows; a super-linear shape means the interceptor is doing work that
# scales with the number of dispatches already seen.
#
# Each point is sampled several times after a warmup and compared on the median.

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "perf-common"))

from perf_cost_model import max_launch_scaling_ratio, model_max_ms
from perf_stats import check_ceiling, repeat_measure, write_results
from run_and_validate import run_case


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--testapp", type=Path, required=True)
    ap.add_argument("--preload", type=Path, required=True)
    ap.add_argument("--ballast-mb", type=int, default=8)
    ap.add_argument("--launches", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--slack", type=float, default=2.0)
    ap.add_argument(
        "--repeat", type=int, default=3, help="timed samples per launch count"
    )
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    stats: dict = {}
    for n in args.launches:
        stats[n] = repeat_measure(
            lambda launches=n: run_case(
                args.testapp, args.preload, args.ballast_mb, launches, args.warmup
            ),
            repeat=args.repeat,
            warmup=1,
            label=f"L={n}",
        )
        check_ceiling(stats[n]["median_ms"], model_max_ms(n), f"L={n}")

    def median(n: int) -> float:
        return stats[n]["median_ms"]

    ordered = sorted(args.launches)
    for i in range(1, len(ordered)):
        lo, hi = ordered[i - 1], ordered[i]
        ratio = median(hi) / max(median(lo), 0.001)
        cap = max_launch_scaling_ratio(lo, hi, args.slack)
        assert ratio <= cap, f"L={lo}->{hi} ratio={ratio:.2f} > {cap:.2f}"
        print(f"[qh-perf-sweep] PASS L={lo}->{hi} ratio={ratio:.2f} <= {cap:.2f}")

    write_results(
        "QH_PERF_SWEEP_JSON",
        {
            "preload": str(args.preload.resolve()),
            "ballast_mb": args.ballast_mb,
            "repeat": args.repeat,
            "per_launch": {str(n): stats[n] for n in ordered},
        },
    )
    print("[qh-perf-sweep] PASS overall")
    return 0


if __name__ == "__main__":
    sys.exit(main())
