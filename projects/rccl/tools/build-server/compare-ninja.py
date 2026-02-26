#!/usr/bin/env python3
"""Compare build-server trace timing against ninja log timing.

Parses .ninja_log for per-target start/end times and build-trace.json for
build-server task durations. Matches targets by TU name and produces a
side-by-side comparison showing where the build server is slower or faster
than ninja subprocesses.

Usage:
  python3 compare-ninja.py <build_dir> [trace.json]

Example:
  python3 compare-ninja.py /path/to/rccl/build build-trace.json
"""

import json
import os
import re
import sys
from collections import defaultdict


def parse_ninja_log(build_dir):
    """Parse .ninja_log, return dict of output_path -> (start_ms, end_ms, dur_ms)."""
    log_path = os.path.join(build_dir, ".ninja_log")
    targets = {}
    with open(log_path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            start_ms = int(parts[0])
            end_ms = int(parts[1])
            output = parts[3]
            targets[output] = (start_ms, end_ms, end_ms - start_ms)
    return targets


def extract_tu_name(path):
    """Extract the TU name from a ninja output path.

    split_device/bc/all_reduce_sum_f32.gfx950.bc -> all_reduce_sum_f32
    split_device/asm/all_reduce_sum_f32.gfx950.s -> all_reduce_sum_f32
    split_device/dev_obj/all_reduce_sum_f32.gfx950.o -> all_reduce_sum_f32
    CMakeFiles/rccl.dir/hipify/src/allocator.cc.o -> allocator
    """
    base = os.path.basename(path)
    # Device: name.gfx*.{bc,s,o}
    m = re.match(r"(.+)\.gfx\w+\.(bc|s|o)$", base)
    if m:
        name = m.group(1)
        if name.endswith(".patched"):
            name = name[: -len(".patched")]
        return name
    # Host: name.cc.o
    m = re.match(r"(.+)\.cc\.o$", base)
    if m:
        return m.group(1)
    return None


def load_trace(path):
    with open(path) as f:
        return json.load(f)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <build_dir> [trace.json]")
        sys.exit(1)

    build_dir = sys.argv[1]
    trace_path = sys.argv[2] if len(sys.argv) > 2 else "build-trace.json"

    ninja = parse_ninja_log(build_dir)
    events = load_trace(trace_path)

    # Classify ninja targets by stage
    ninja_fe = {}  # TU name -> dur_ms  (split_device/bc/*.bc = frontend)
    ninja_be = {}  # TU name -> dur_ms  (split_device/asm/*.s = backend)
    ninja_asm = {}  # TU name -> dur_ms (split_device/dev_obj/*.o = assembly)
    ninja_host = {}  # TU name -> dur_ms (CMakeFiles/rccl.dir/*.cc.o)
    ninja_other = {}  # special targets

    for path, (start, end, dur) in ninja.items():
        name = extract_tu_name(path)
        if path.startswith("split_device/bc/") and path.endswith(".bc"):
            if name:
                ninja_fe[name] = dur
        elif path.startswith("split_device/asm/") and path.endswith(".s"):
            if name:
                ninja_be[name] = dur
        elif path.startswith("split_device/dev_obj/") and path.endswith(".o"):
            if name and name != "combined":
                ninja_asm[name] = dur
        elif "CMakeFiles/rccl.dir/" in path and path.endswith(".o"):
            if name:
                ninja_host[name] = dur
        elif "combined" in path or "hipfb" in path or "librccl" in path:
            ninja_other[os.path.basename(path)] = dur

    # Build server events by (cat, name)
    bs = defaultdict(dict)
    for e in events:
        bs[e["cat"]][e.get("name", "")] = e["dur"] / 1000.0  # us -> ms

    # --- Compare device TUs ---
    print("=" * 100)
    print(f"{'TU Name':<45s} {'Stage':<8s} {'Ninja(ms)':>10s} {'BS(ms)':>10s} {'Delta(ms)':>10s} {'Ratio':>8s}")
    print("=" * 100)

    total_ninja = 0.0
    total_bs = 0.0
    comparisons = []

    for name in sorted(set(ninja_fe.keys()) | set(bs.get("callee_fe", {}).keys()) | set(bs.get("kernel_fe", {}).keys())):
        for stage, ninja_dict, bs_cat in [
            ("fe", ninja_fe, "callee_fe"),
            ("be", ninja_be, "callee_be"),
            ("asm", ninja_asm, "callee_asm"),
        ]:
            n_ms = ninja_dict.get(name)
            b_ms = bs.get(bs_cat, {}).get(name)
            # Also check kernel categories
            if b_ms is None:
                b_ms = bs.get(f"kernel_{stage}", {}).get(name)
                if b_ms is None and stage == "asm":
                    b_ms = bs.get("kernel_asm", {}).get(name)

            if n_ms is not None and b_ms is not None:
                delta = b_ms - n_ms
                ratio = b_ms / n_ms if n_ms > 0 else float("inf")
                comparisons.append((name, stage, n_ms, b_ms, delta, ratio))
                total_ninja += n_ms
                total_bs += b_ms

    # Sort by delta (biggest slowdown first)
    comparisons.sort(key=lambda x: -x[4])

    for name, stage, n_ms, b_ms, delta, ratio in comparisons[:40]:
        print(f"{name:<45s} {stage:<8s} {n_ms:>10.0f} {b_ms:>10.0f} {delta:>+10.0f} {ratio:>7.2f}x")

    if len(comparisons) > 40:
        print(f"  ... ({len(comparisons) - 40} more rows)")

    # --- Compare host TUs ---
    host_comparisons = []
    for name in sorted(set(ninja_host.keys()) | set(bs.get("host_compile", {}).keys())):
        n_ms = ninja_host.get(name)
        b_ms = bs.get("host_compile", {}).get(name)
        if n_ms is not None and b_ms is not None:
            delta = b_ms - n_ms
            ratio = b_ms / n_ms if n_ms > 0 else float("inf")
            host_comparisons.append((name, n_ms, b_ms, delta, ratio))
            total_ninja += n_ms
            total_bs += b_ms

    if host_comparisons:
        print()
        print("=" * 100)
        print(f"{'Host TU':<45s} {'Stage':<8s} {'Ninja(ms)':>10s} {'BS(ms)':>10s} {'Delta(ms)':>10s} {'Ratio':>8s}")
        print("=" * 100)
        host_comparisons.sort(key=lambda x: -x[3])
        for name, n_ms, b_ms, delta, ratio in host_comparisons[:20]:
            print(f"{name:<45s} {'host':<8s} {n_ms:>10.0f} {b_ms:>10.0f} {delta:>+10.0f} {ratio:>7.2f}x")
        if len(host_comparisons) > 20:
            print(f"  ... ({len(host_comparisons) - 20} more rows)")

    # --- Special targets ---
    print()
    print("=" * 100)
    print("Special targets:")
    print("=" * 100)
    for label, ninja_key, bs_cat in [
        ("lld -r (combined.o)", "combined.gfx950.o", "lld_r"),
        ("lld -shared (combined.so)", "combined.gfx950.so", "split_cobj"),
        ("offload-bundler (hipfb)", "combined.hipfb", "split_hipfb"),
        ("host stub (common.host.o)", "common.host.o", "split_host"),
        ("final link (librccl.so)", "librccl.so.1.0", "final_link"),
    ]:
        n_ms = ninja_other.get(ninja_key)
        b_ms_dict = bs.get(bs_cat, {})
        b_ms = next(iter(b_ms_dict.values()), None) if b_ms_dict else None
        if b_ms is not None:
            b_ms_val = b_ms
        else:
            b_ms_val = None
        if n_ms is not None and b_ms_val is not None:
            delta = b_ms_val - n_ms
            print(f"  {label:<40s}  ninja={n_ms:>8.0f}ms  bs={b_ms_val:>8.0f}ms  delta={delta:>+8.0f}ms")
        elif n_ms is not None:
            print(f"  {label:<40s}  ninja={n_ms:>8.0f}ms  bs=    N/A")
        elif b_ms_val is not None:
            print(f"  {label:<40s}  ninja=    N/A  bs={b_ms_val:>8.0f}ms")

    # --- Summary ---
    print()
    print("=" * 100)
    print("Summary (matched tasks only):")
    print(f"  Total ninja task time:  {total_ninja/1000:>10.1f}s")
    print(f"  Total BS task time:     {total_bs/1000:>10.1f}s")
    print(f"  Aggregate delta:        {(total_bs - total_ninja)/1000:>+10.1f}s")
    print(f"  Aggregate ratio:        {total_bs/total_ninja:>10.2f}x" if total_ninja > 0 else "")

    # Ninja wall time
    all_starts = [s for s, e, d in ninja.values()]
    all_ends = [e for s, e, d in ninja.values()]
    if all_starts and all_ends:
        ninja_wall = (max(all_ends) - min(all_starts)) / 1000.0
        print(f"  Ninja wall time:        {ninja_wall:>10.1f}s")

    # BS wall time
    if events:
        bs_wall = (max(e["ts"] + e["dur"] for e in events) - min(e["ts"] for e in events)) / 1e6
        print(f"  BS wall time:           {bs_wall:>10.1f}s")


if __name__ == "__main__":
    main()
