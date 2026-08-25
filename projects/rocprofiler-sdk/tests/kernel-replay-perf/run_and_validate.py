#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Runs the kernel-replay perf workload under LD_PRELOAD for P=1 and P=5, then validates
# wall-time bounds and pass-count scaling.

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Re-use marker parsing and cost-model checks from validate_scaling.py in this directory.
from validate_scaling import model_max_ms, parse_marker


def run_case(testapp: Path, client: Path, passes: int, ballast_mb: int, launches: int) -> str:
    env = os.environ.copy()
    env["KR_PERF_PASSES"] = str(passes)
    preload = client.resolve()
    if env.get("LD_PRELOAD"):
        env["LD_PRELOAD"] = f"{preload}:{env['LD_PRELOAD']}"
    else:
        env["LD_PRELOAD"] = str(preload)
    proc = subprocess.run(
        [str(testapp.resolve()), str(ballast_mb), str(launches)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(
            f"P={passes} run failed rc={proc.returncode}\n{out}"
        )
    if "[kr-perf] PASS" not in out:
        raise RuntimeError(f"P={passes} missing PASS marker\n{out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--testapp", type=Path, required=True)
    ap.add_argument("--client", type=Path, required=True)
    ap.add_argument("--ballast-mb", type=int, default=64)
    ap.add_argument("--launches", type=int, default=8)
    ap.add_argument("--high-passes", type=int, default=5)
    ap.add_argument("--max-scaling-ratio", type=float, default=8.0)
    args = ap.parse_args()

    base_out = run_case(args.testapp, args.client, 1, args.ballast_mb, args.launches)
    high_out = run_case(args.testapp, args.client, args.high_passes, args.ballast_mb, args.launches)

    base_m = parse_marker(base_out)
    high_m = parse_marker(high_out)

    for label, m, p in (("P=1", base_m, 1), (f"P={args.high_passes}", high_m, args.high_passes)):
        ceiling = model_max_ms(int(m["ballast_mb"]), int(m["launches"]), p)
        assert m["wall_ms"] <= ceiling, (
            f"{label} wall_ms={m['wall_ms']:.1f} > ceiling {ceiling:.1f} ms"
        )
        print(f"[kr-perf-run] {label} wall_ms={m['wall_ms']:.1f} <= {ceiling:.1f} ms")

    ratio = high_m["wall_ms"] / max(base_m["wall_ms"], 0.001)
    assert ratio <= args.max_scaling_ratio, (
        f"scaling ratio {ratio:.2f} > {args.max_scaling_ratio} "
        f"(P=1 {base_m['wall_ms']:.1f} ms vs P={args.high_passes} {high_m['wall_ms']:.1f} ms)"
    )
    print(
        f"[kr-perf-run] PASS scaling ratio={ratio:.2f} "
        f"(P=1 {base_m['wall_ms']:.1f} ms -> P={args.high_passes} {high_m['wall_ms']:.1f} ms)"
    )

    results = {
        "ballast_mb": args.ballast_mb,
        "launches": args.launches,
        "p1_wall_ms": base_m["wall_ms"],
        "pN_wall_ms": high_m["wall_ms"],
        "passes_high": args.high_passes,
        "scaling_ratio": ratio,
    }
    out_path = Path(os.environ.get("KR_PERF_RESULTS_JSON", ""))
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
