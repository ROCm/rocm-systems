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
import re
import sys

from perf_cost_model import model_max_ms

_MARKER = re.compile(
    r"\[kr-perf\]\s+ballast_mb=(?P<ballast_mb>\d+)\s+launches=(?P<launches>\d+)\s+"
    r"wall_ms=(?P<wall_ms>[\d.]+)\s+counter=(?P<counter>\d+)"
)


def parse_marker(text: str) -> dict:
    for line in text.splitlines():
        m = _MARKER.search(line)
        if m:
            return {k: float(v) if k == "wall_ms" else int(v) for k, v in m.groupdict().items()}
    raise AssertionError("missing [kr-perf] wall_ms marker in test output")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, help="ctest stdout capture")
    ap.add_argument("--passes", type=int, required=True)
    ap.add_argument("--baseline-log", help="optional 1-pass baseline log for scaling check")
    ap.add_argument("--max-scaling-ratio", type=float, default=8.0)
    args = ap.parse_args()

    log_text = open(args.log, encoding="utf-8").read()
    m = parse_marker(log_text)
    assert m["counter"] == m["launches"], (
        f"counter {m['counter']} != launches {m['launches']} (restore failure)"
    )

    max_ms = model_max_ms(int(m["ballast_mb"]), int(m["launches"]), args.passes)
    assert m["wall_ms"] <= max_ms, (
        f"P={args.passes} wall_ms={m['wall_ms']:.1f} exceeds cost-model ceiling "
        f"{max_ms:.1f} ms (ballast_mb={m['ballast_mb']} launches={m['launches']})"
    )
    print(
        f"[kr-perf-validate] PASS wall_ms={m['wall_ms']:.1f} <= ceiling={max_ms:.1f} ms "
        f"(P={args.passes})"
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
