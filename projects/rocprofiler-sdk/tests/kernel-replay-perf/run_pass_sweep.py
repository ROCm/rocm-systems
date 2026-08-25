#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Pass-count sweep regression: replay wall time must grow roughly linearly with P,
# not super-linearly. Catches version regressions that add per-pass fixed overhead
# or break restore amortization.

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from perf_cost_model import max_pass_scaling_ratio, model_max_ms


def run_case(testapp: Path, client: Path, passes: int, ballast_mb: int, launches: int) -> float:
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
    if proc.returncode != 0 or "[kr-perf] PASS" not in out:
        raise RuntimeError(f"P={passes} failed rc={proc.returncode}\n{out}")
    for line in out.splitlines():
        if line.startswith("[kr-perf] ballast_mb="):
            parts = dict(
                kv.split("=", 1)
                for kv in line.replace("[kr-perf] ", "").split()
                if "=" in kv
            )
            return float(parts["wall_ms"])
    raise RuntimeError(f"P={passes} missing wall_ms marker\n{out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--testapp", type=Path, required=True)
    ap.add_argument("--client", type=Path, required=True)
    ap.add_argument("--ballast-mb", type=int, default=32)
    ap.add_argument("--launches", type=int, default=4)
    ap.add_argument("--passes", type=int, nargs="+", default=[1, 3, 5, 8])
    ap.add_argument("--slack", type=float, default=2.0,
                    help="multiplier over ideal linear pass scaling")
    args = ap.parse_args()

    wall: dict[int, float] = {}
    for p in args.passes:
        wall[p] = run_case(args.testapp, args.client, p, args.ballast_mb, args.launches)
        ceiling = model_max_ms(args.ballast_mb, args.launches, p)
        assert wall[p] <= ceiling, (
            f"P={p} wall_ms={wall[p]:.1f} > ceiling {ceiling:.1f} ms"
        )
        print(f"[kr-perf-sweep] P={p} wall_ms={wall[p]:.1f} <= {ceiling:.1f} ms")

    ordered = sorted(args.passes)
    for i in range(1, len(ordered)):
        p_lo, p_hi = ordered[i - 1], ordered[i]
        ratio = wall[p_hi] / max(wall[p_lo], 0.001)
        cap = max_pass_scaling_ratio(p_lo, p_hi, args.slack)
        assert ratio <= cap, (
            f"pass sweep super-linear: P={p_lo}->{p_hi} ratio={ratio:.2f} > {cap:.2f}"
        )
        print(f"[kr-perf-sweep] PASS P={p_lo}->{p_hi} ratio={ratio:.2f} <= {cap:.2f}")

    overall = wall[ordered[-1]] / max(wall[ordered[0]], 0.001)
    overall_cap = max_pass_scaling_ratio(ordered[0], ordered[-1], args.slack)
    assert overall <= overall_cap, (
        f"overall sweep ratio {overall:.2f} > {overall_cap:.2f} "
        f"(P={ordered[0]}..{ordered[-1]})"
    )
    print(f"[kr-perf-sweep] PASS overall ratio={overall:.2f} <= {overall_cap:.2f}")

    out_path = Path(os.environ.get("KR_PERF_SWEEP_JSON", ""))
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"ballast_mb": args.ballast_mb, "launches": args.launches,
                        "wall_ms": wall}, indent=2) + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
