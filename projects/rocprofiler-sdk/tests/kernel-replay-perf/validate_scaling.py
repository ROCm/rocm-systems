#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Validates kernel-replay end-to-end performance from test stdout markers:
#   [kr-perf] ballast_mb=... launches=... wall_ms=... counter=...
#
# Checks:
#   1. Replay with P passes completes within the bandwidth cost-model ceiling.
#   2. P-pass wall time does not blow up relative to a 1-pass baseline (linear scaling).

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "perf-common"))

from perf_cost_model import model_max_ms
from perf_stats import check_ceiling
from perf_stats import parse_marker as _parse_marker


def parse_marker(text: str) -> dict:
    return _parse_marker(text, "kr-perf")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, help="ctest stdout capture")
    ap.add_argument("--passes", type=int, required=True)
    ap.add_argument(
        "--baseline-log", help="optional 1-pass baseline log for scaling check"
    )
    ap.add_argument("--max-scaling-ratio", type=float, default=8.0)
    args = ap.parse_args()

    log_text = open(args.log, encoding="utf-8").read()
    m = parse_marker(log_text)
    assert (
        m["counter"] == m["launches"]
    ), f"counter {m['counter']} != launches {m['launches']} (restore failure)"

    max_ms = model_max_ms(int(m["ballast_mb"]), int(m["launches"]), args.passes)
    check_ceiling(
        m["wall_ms"],
        max_ms,
        f"[kr-perf-validate] P={args.passes} "
        f"(ballast_mb={m['ballast_mb']} launches={m['launches']})",
    )

    if args.baseline_log:
        base = parse_marker(open(args.baseline_log, encoding="utf-8").read())
        assert base["ballast_mb"] == m["ballast_mb"]
        assert base["launches"] == m["launches"]
        if base["wall_ms"] > 0:
            ratio = m["wall_ms"] / base["wall_ms"]
            assert ratio <= args.max_scaling_ratio, (
                f"replay scaling blew up: P={args.passes} wall_ms={m['wall_ms']:.1f} vs "
                f"P=1 wall_ms={base['wall_ms']:.1f} ratio={ratio:.2f} > {args.max_scaling_ratio}"
            )
            print(
                f"[kr-perf-validate] PASS scaling ratio={ratio:.2f} <= "
                f"{args.max_scaling_ratio} (P={args.passes} vs P=1)"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
