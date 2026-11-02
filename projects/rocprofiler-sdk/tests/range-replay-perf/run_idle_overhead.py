#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Idle-gate overhead (P1): range replay is configured, but the application never opens a
# range. The interception gate is still process-wide, so every dispatch pays the hook that
# checks whether a range is open. This compares per-dispatch wall time with that gate against
# the same workload with no tool at all.
#
# The tool half must confirm that zero ranges closed. Without that check a harness that
# accidentally opened a range would time the replay path and read it as gate overhead.

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "perf-common"))

from perf_stats import parse_marker, repeat_measure, write_results  # noqa: E402
from rr_perf_common import parse_client_marker  # noqa: E402


def run_idle(testapp: Path, client: Path | None, dispatches: int, warmup: int) -> float:
    """One timed idle-gate run. Returns in-app wall time in milliseconds."""
    env = os.environ.copy()
    env["RR_PERF_IDLE_GATE"] = "1"
    if client is not None:
        env["RR_PERF_EXPECT_IDLE"] = "1"
        preload = client.resolve()
        if env.get("LD_PRELOAD"):
            env["LD_PRELOAD"] = f"{preload}:{env['LD_PRELOAD']}"
        else:
            env["LD_PRELOAD"] = str(preload)
    else:
        env.pop("RR_PERF_EXPECT_IDLE", None)
        # A leftover preload would install some other tool and poison the no-tool baseline.
        env.pop("LD_PRELOAD", None)

    # ballast_mb is unused in idle mode but kept so the argv layout matches the other drivers.
    proc = subprocess.run(
        [str(testapp.resolve()), "1", str(dispatches), "1", str(warmup)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    out = proc.stdout + proc.stderr
    label = "idle+tool" if client is not None else "idle+no-tool"
    # Propagate the application's no-GPU skip so ctest's SKIP_REGULAR_EXPRESSION can match.
    if "[rr-perf] SKIP" in out:
        print(out, end="")
        raise SystemExit(0)
    if proc.returncode != 0:
        raise RuntimeError(f"{label} run failed rc={proc.returncode}\n{out}")
    if "[rr-perf] PASS" not in out:
        raise RuntimeError(f"{label} missing PASS marker\n{out}")

    if client is not None:
        tool = parse_client_marker(out)
        if tool.get("ranges", -1) != 0 or tool.get("replayed", -1) != 0:
            raise RuntimeError(
                f"{label} expected zero ranges under RR_PERF_EXPECT_IDLE, "
                f"tool reported ranges={tool.get('ranges')} replayed={tool.get('replayed')}\n{out}"
            )

    return float(parse_marker(out, "rr-perf")["wall_ms"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--testapp", type=Path, required=True)
    ap.add_argument("--client", type=Path, required=True)
    ap.add_argument("--dispatches", type=int, default=256)
    ap.add_argument(
        "--max-overhead-ratio",
        type=float,
        default=2.0,
        help=(
            "cap on wall(with range-replay tool) / wall(no tool). The idle gate should be "
            "near free; the cap is generous for shared runners"
        ),
    )
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    if args.dispatches < 1:
        raise SystemExit(f"--dispatches must be >= 1 (got {args.dispatches})")
    if args.max_overhead_ratio < 1.0:
        raise SystemExit(
            f"--max-overhead-ratio must be >= 1.0 (got {args.max_overhead_ratio})"
        )

    baseline = repeat_measure(
        lambda: run_idle(args.testapp, None, args.dispatches, args.warmup),
        repeat=args.repeat,
        warmup=1,
        label="no-tool",
    )
    with_tool = repeat_measure(
        lambda: run_idle(args.testapp, args.client, args.dispatches, args.warmup),
        repeat=args.repeat,
        warmup=1,
        label="range-replay-configured",
    )

    base_ms = baseline["median_ms"]
    tool_ms = with_tool["median_ms"]
    ratio = tool_ms / max(base_ms, 0.001)

    assert ratio <= args.max_overhead_ratio, (
        f"idle-gate overhead ratio {ratio:.2f} > {args.max_overhead_ratio:.2f} "
        f"(no-tool {base_ms:.1f} ms vs configured {tool_ms:.1f} ms, medians)"
    )
    print(
        f"[rr-perf-idle] PASS overhead ratio={ratio:.2f} <= {args.max_overhead_ratio:.2f} "
        f"(no-tool {base_ms:.1f} ms -> configured {tool_ms:.1f} ms, medians)"
    )

    write_results(
        "RR_PERF_IDLE_JSON",
        {
            "dispatches": args.dispatches,
            "repeat": args.repeat,
            "no_tool": baseline,
            "configured": with_tool,
            "overhead_ratio": ratio,
            "overhead_cap": args.max_overhead_ratio,
        },
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
