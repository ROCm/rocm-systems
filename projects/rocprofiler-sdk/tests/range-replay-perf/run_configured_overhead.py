#!/usr/bin/env python3
# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Overhead of hipMemcpyAsync (P2) and hipGraphLaunch (P3) while range replay is configured
# and no range is opened.
#
# P2: the async-copy wrappers stay installed for range replay, and every copy pays pointer
# classification before it is forwarded, whether or not a range is open.
# P3: a graph launch does not take the fast path that writes the original packets through
# while a range replay context is active, because that path would hide the launch from the
# service.
#
# The tool half must report zero closed ranges before the ratio is accepted.

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "perf-common"))

from perf_stats import parse_marker, repeat_measure, write_results  # noqa: E402
from rr_perf_common import parse_client_marker  # noqa: E402

KINDS = {
    "copies": ("RR_PERF_COPIES", "RR_PERF_COPIES_JSON", "rr-perf-copies"),
    "graphs": ("RR_PERF_GRAPHS", "RR_PERF_GRAPHS_JSON", "rr-perf-graphs"),
}


def run_case(
    testapp: Path, client: Path | None, kind: str, count: int, warmup: int
) -> float:
    """One timed run. Returns in-app wall time in milliseconds."""
    flag, _, _ = KINDS[kind]
    env = os.environ.copy()
    env[flag] = "1"
    if client is not None:
        env["RR_PERF_EXPECT_IDLE"] = "1"
        preload = client.resolve()
        if env.get("LD_PRELOAD"):
            env["LD_PRELOAD"] = f"{preload}:{env['LD_PRELOAD']}"
        else:
            env["LD_PRELOAD"] = str(preload)
    else:
        env.pop("RR_PERF_EXPECT_IDLE", None)
        env.pop("LD_PRELOAD", None)

    proc = subprocess.run(
        [str(testapp.resolve()), "1", str(count), "1", str(warmup)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    out = proc.stdout + proc.stderr
    label = f"{kind}+tool" if client is not None else f"{kind}+no-tool"
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
                f"{label} expected zero ranges, tool reported "
                f"ranges={tool.get('ranges')} replayed={tool.get('replayed')}\n{out}"
            )

    return float(parse_marker(out, "rr-perf")["wall_ms"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kind", choices=sorted(KINDS), required=True)
    ap.add_argument("--testapp", type=Path, required=True)
    ap.add_argument("--client", type=Path, required=True)
    ap.add_argument("--count", type=int, default=64)
    ap.add_argument("--max-overhead-ratio", type=float, required=True)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    args = ap.parse_args()

    if args.count < 1:
        raise SystemExit(f"--count must be >= 1 (got {args.count})")
    if args.max_overhead_ratio < 1.0:
        raise SystemExit(
            f"--max-overhead-ratio must be >= 1.0 (got {args.max_overhead_ratio})"
        )

    _, results_env, marker = KINDS[args.kind]
    baseline = repeat_measure(
        lambda: run_case(args.testapp, None, args.kind, args.count, args.warmup),
        repeat=args.repeat,
        warmup=1,
        label="no-tool",
    )
    with_tool = repeat_measure(
        lambda: run_case(args.testapp, args.client, args.kind, args.count, args.warmup),
        repeat=args.repeat,
        warmup=1,
        label="range-replay-configured",
    )

    base_ms = baseline["median_ms"]
    tool_ms = with_tool["median_ms"]
    ratio = tool_ms / max(base_ms, 0.001)

    assert ratio <= args.max_overhead_ratio, (
        f"{args.kind} overhead ratio {ratio:.2f} > {args.max_overhead_ratio:.2f} "
        f"(no-tool {base_ms:.1f} ms vs configured {tool_ms:.1f} ms, medians)"
    )
    print(
        f"[{marker}] PASS overhead ratio={ratio:.2f} <= {args.max_overhead_ratio:.2f} "
        f"(no-tool {base_ms:.1f} ms -> configured {tool_ms:.1f} ms, medians)"
    )

    write_results(
        results_env,
        {
            "kind": args.kind,
            "count": args.count,
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
