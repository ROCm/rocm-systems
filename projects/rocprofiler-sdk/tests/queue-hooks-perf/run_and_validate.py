#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Runs dispatch-counter perf workload under LD_PRELOAD and validates wall-time bounds.

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from perf_cost_model import max_launch_scaling_ratio, model_max_ms

_MARKER = __import__("re").compile(
    r"\[qh-perf\]\s+ballast_mb=(?P<ballast_mb>\d+)\s+launches=(?P<launches>\d+)\s+"
    r"wall_ms=(?P<wall_ms>[\d.]+)\s+counter=(?P<counter>\d+)"
)


def parse_marker(text: str) -> dict:
    for line in text.splitlines():
        m = _MARKER.search(line)
        if m:
            return {
                k: float(v) if k == "wall_ms" else int(v)
                for k, v in m.groupdict().items()
            }
    raise AssertionError("missing [qh-perf] wall_ms marker")


def run_case(testapp: Path, preload: Path, ballast_mb: int, launches: int) -> str:
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
        [str(testapp.resolve()), str(ballast_mb), str(launches)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    out = proc.stdout + proc.stderr
    if proc.returncode != 0 or "[qh-perf] PASS" not in out:
        raise RuntimeError(f"launches={launches} failed rc={proc.returncode}\n{out}")
    return out


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
    args = ap.parse_args()

    low_out = run_case(args.testapp, args.preload, args.ballast_mb, args.launches_low)
    high_out = run_case(args.testapp, args.preload, args.ballast_mb, args.launches_high)
    low_m = parse_marker(low_out)
    high_m = parse_marker(high_out)

    for label, m, launches in (
        (f"L={args.launches_low}", low_m, args.launches_low),
        (f"L={args.launches_high}", high_m, args.launches_high),
    ):
        ceiling = model_max_ms(launches)
        assert (
            m["wall_ms"] <= ceiling
        ), f"{label} wall_ms={m['wall_ms']:.1f} > ceiling {ceiling:.1f} ms"
        print(f"[qh-perf-run] {label} wall_ms={m['wall_ms']:.1f} <= {ceiling:.1f} ms")

    ratio = high_m["wall_ms"] / max(low_m["wall_ms"], 0.001)
    cap = min(
        args.max_scaling_ratio,
        max_launch_scaling_ratio(args.launches_low, args.launches_high, 2.0),
    )
    assert ratio <= cap, (
        f"scaling ratio {ratio:.2f} > {cap:.2f} "
        f"(L={args.launches_low} {low_m['wall_ms']:.1f} ms vs "
        f"L={args.launches_high} {high_m['wall_ms']:.1f} ms)"
    )
    print(f"[qh-perf-run] PASS scaling ratio={ratio:.2f} <= {cap:.2f}")

    out_json = os.environ.get("QH_PERF_RESULTS_JSON", "")
    if out_json:
        out_path = Path(out_json)
        variant = os.environ.get("QH_PERF_VARIANT", "unknown")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "variant": variant,
                    "ballast_mb": args.ballast_mb,
                    "launches_low": args.launches_low,
                    "launches_high": args.launches_high,
                    "wall_ms_low": low_m["wall_ms"],
                    "wall_ms_high": high_m["wall_ms"],
                    "scaling_ratio": ratio,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
