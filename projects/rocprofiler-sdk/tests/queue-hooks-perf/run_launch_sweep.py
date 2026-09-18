#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Launch-count sweep for dispatch-counter interposition overhead.

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from perf_cost_model import max_launch_scaling_ratio, model_max_ms
from run_and_validate import parse_marker, run_case


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--testapp", type=Path, required=True)
    ap.add_argument("--preload", type=Path, required=True)
    ap.add_argument("--ballast-mb", type=int, default=8)
    ap.add_argument("--launches", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--slack", type=float, default=2.0)
    args = ap.parse_args()

    wall: dict[int, float] = {}
    for n in args.launches:
        out = run_case(args.testapp, args.preload, args.ballast_mb, n)
        wall[n] = parse_marker(out)["wall_ms"]
        ceiling = model_max_ms(n)
        assert wall[n] <= ceiling, f"L={n} wall_ms={wall[n]:.1f} > {ceiling:.1f} ms"
        print(f"[qh-perf-sweep] L={n} wall_ms={wall[n]:.1f} <= {ceiling:.1f} ms")

    ordered = sorted(args.launches)
    for i in range(1, len(ordered)):
        lo, hi = ordered[i - 1], ordered[i]
        ratio = wall[hi] / max(wall[lo], 0.001)
        cap = max_launch_scaling_ratio(lo, hi, args.slack)
        assert ratio <= cap, f"L={lo}->{hi} ratio={ratio:.2f} > {cap:.2f}"
        print(f"[qh-perf-sweep] PASS L={lo}->{hi} ratio={ratio:.2f} <= {cap:.2f}")

    out_path = Path(os.environ.get("QH_PERF_SWEEP_JSON", ""))
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "variant": os.environ.get("QH_PERF_VARIANT", "unknown"),
                    "ballast_mb": args.ballast_mb,
                    "wall_ms": wall,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print("[qh-perf-sweep] PASS overall")
    return 0


if __name__ == "__main__":
    sys.exit(main())
