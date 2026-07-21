#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Parse and analyze SQTT address trace data.

Reads shaderdata JSON files containing address trace records (emitted by
SQTT_TRACE_ADDRESSES=memory|lds|memory,lds) and outputs per-lane addresses
as JSON for downstream analysis (cache line utilization, stride detection,
memory coalescing analysis).

Usage:
    # Dump all address traces as JSON
    python3 sqtt_memory_trace.py build/test/ -o addresses.json

    # Filter to specific CU/SIMD
    python3 sqtt_memory_trace.py build/test/ --cu 0 --simd 0 -o addresses.json

    # With demangling (for kernel context in output)
    python3 sqtt_memory_trace.py build/test/ --demangle -o addresses.json

    # Print summary stats to stderr
    python3 sqtt_memory_trace.py build/test/ --summary
"""

import argparse
import json
import os
import sys
from typing import Optional

from sqtt_data import (
    AddressTrace,
    build_kernel_to_co,
    discover_base_dir,
    load_funcmaps,
    load_occupancy,
    load_shaderdata,
    make_funcmap_resolver,
    merge_funcmaps,
    preprocess_records,
)


def format_address_trace(trace: AddressTrace) -> dict:
    """Convert an AddressTrace to a JSON-serializable dict."""
    is_64bit = trace.kind not in ("lds_load", "lds_store", "lds_atomic",
                                      "ds_permute", "ds_bpermute")
    fmt = "0x{:016x}" if is_64bit else "0x{:08x}"
    result = {
        "kind": trace.kind,
        "marker_id": trace.marker_id,
        "time": trace.time,
        "shader_engine": trace.shader_engine,
        "cu": trace.cu,
        "simd": trace.simd,
        "wave": trace.wave_id,
        "exec_mask": f"0x{trace.exec_mask:016x}",
        "active_lanes": trace.active_lane_count(),
        "addresses": [fmt.format(a) for a in trace.addresses],
    }
    if trace.source_loc:
        result["source"] = trace.source_loc
    return result


def compute_deltas(addresses: list[int]) -> list[int]:
    """Compute inter-lane address deltas (stride pattern)."""
    if len(addresses) < 2:
        return []
    return [addresses[i+1] - addresses[i] for i in range(len(addresses) - 1)]


def format_addr(addr: int, is_64bit: bool) -> str:
    return f"0x{addr:016x}" if is_64bit else f"0x{addr:08x}"


def print_summary(traces: list[AddressTrace], file=sys.stderr):
    """Print hierarchical summary: source → marker_id → wave → addresses + deltas."""
    if not traces:
        print("No address traces found.", file=file)
        return

    # Global stats
    total_addrs = sum(len(t.addresses) for t in traces)
    shader_engines = set()
    cus = set()
    simds = set()
    by_kind: dict[str, int] = {}
    for t in traces:
        by_kind[t.kind] = by_kind.get(t.kind, 0) + 1
        shader_engines.add(t.shader_engine)
        cus.add(t.cu)
        simds.add(t.simd)

    print(f"\nAddress trace summary:", file=file)
    print(f"  Total operations: {len(traces)}", file=file)
    print(f"  Total addresses:  {total_addrs}", file=file)
    print(f"  Shader engines:  {sorted(shader_engines)}", file=file)
    print(f"  CUs seen:         {sorted(cus)}", file=file)
    print(f"  SIMDs seen:       {sorted(simds)}", file=file)
    print(f"  By kind:", file=file)
    for kind in sorted(by_kind):
        print(f"    {kind}: {by_kind[kind]}", file=file)

    # Group by source location (or marker_id if no source)
    from collections import defaultdict
    by_source: dict[str, list[AddressTrace]] = defaultdict(list)
    for t in traces:
        key = t.source_loc if t.source_loc else f"marker#{t.marker_id}"
        by_source[key].append(t)

    print(f"\n  Hierarchical breakdown:", file=file)
    for src_key in sorted(by_source.keys()):
        src_traces = by_source[src_key]
        kinds = set(t.kind for t in src_traces)
        kind_str = ",".join(sorted(kinds))
        print(f"\n  {src_key}  ({kind_str}, {len(src_traces)} ops)", file=file)

        # Group by marker_id within source
        by_mid: dict[int, list[AddressTrace]] = defaultdict(list)
        for t in src_traces:
            by_mid[t.marker_id].append(t)

        for mid in sorted(by_mid.keys()):
            mid_traces = by_mid[mid]
            print(f"    marker_id={mid}  ({len(mid_traces)} samples)", file=file)

            # Group by wave (SE, CU, SIMD, wave_id)
            by_wave: dict[tuple[int, int, int, int], list[AddressTrace]] = defaultdict(list)
            for t in mid_traces:
                by_wave[(t.shader_engine, t.cu, t.simd, t.wave_id)].append(t)

            for (se, cu, simd, wid) in sorted(by_wave.keys()):
                wave_traces = by_wave[(se, cu, simd, wid)]
                is_64bit = wave_traces[0].kind in ("load", "store")
                print(f"      wave SE={se} CU={cu} SIMD={simd} wave={wid}  "
                      f"({len(wave_traces)} samples)", file=file)

                for t in wave_traces:
                    n_active = t.active_lane_count()
                    addrs = t.addresses
                    deltas = compute_deltas(addrs)
                    unique_deltas = sorted(set(deltas)) if deltas else []

                    # Compact delta display
                    if unique_deltas:
                        delta_strs = [str(d) for d in unique_deltas]
                        delta_display = ",".join(delta_strs)
                    else:
                        delta_display = "-"

                    if addrs:
                        first = format_addr(addrs[0], is_64bit)
                        last = format_addr(addrs[-1], is_64bit)
                        print(f"        t={t.time} lanes={n_active} "
                              f"range=[{first}..{last}] "
                              f"deltas=[{delta_display}]", file=file)
                    else:
                        print(f"        t={t.time} lanes={n_active} "
                              f"(no addresses)", file=file)


def main():
    parser = argparse.ArgumentParser(
        description="Parse and analyze SQTT address trace data")
    parser.add_argument("base_dir",
        help="Base directory containing trace output.")
    parser.add_argument("--code-objects", "-c", nargs="+",
        help="Code object files (.out) with .sqtt_funcmap sections.")
    parser.add_argument("--demangle", "-d", action="store_true",
        help="Demangle C++ function names")
    parser.add_argument("--cu", type=int, default=None,
        help="Filter to specific CU/WGP ID")
    parser.add_argument("--simd", type=int, default=None,
        help="Filter to specific SIMD ID")
    parser.add_argument("--wave", type=int, default=None,
        help="Filter to specific wave ID")
    parser.add_argument("--output", "-o", default=None,
        help="Output JSON file (default: stdout)")
    parser.add_argument("--summary", "-s", action="store_true",
        help="Print summary statistics to stderr")
    args = parser.parse_args()

    base_dir = args.base_dir
    if not os.path.isdir(base_dir):
        print(f"Error: {base_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    trace_dirs, discovered_cos = discover_base_dir(base_dir)
    code_objects = args.code_objects or discovered_cos

    if not trace_dirs:
        print(f"Error: no shaderdata_*.json files found under {base_dir}",
              file=sys.stderr)
        sys.exit(1)

    if not code_objects:
        print("Error: no code objects found. Use --code-objects.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(trace_dirs)} trace dir(s), "
          f"{len(code_objects)} code object(s)", file=sys.stderr)

    per_co_raw = load_funcmaps(code_objects, do_demangle=False)
    kernel_to_co = build_kernel_to_co(per_co_raw)
    per_co = (load_funcmaps(code_objects, do_demangle=True)
              if args.demangle else per_co_raw)

    # Build merged funcmap for preprocessing.
    merged = merge_funcmaps(per_co.values())

    # Collect address traces from all directories
    all_traces: list[AddressTrace] = []
    total_records = 0

    for td in trace_dirs:
        records = load_shaderdata(td)
        if not records:
            continue
        total_records += len(records)

        dispatches, wave_spans = load_occupancy(td)
        resolver = make_funcmap_resolver(
            per_co, dispatches, kernel_to_co, wave_spans, merged)
        _, addr_traces = preprocess_records(records, merged, resolver)

        # Apply filters
        for t in addr_traces:
            if args.cu is not None and t.cu != args.cu:
                continue
            if args.simd is not None and t.simd != args.simd:
                continue
            if args.wave is not None and t.wave_id != args.wave:
                continue
            all_traces.append(t)

        print(f"  {os.path.basename(td)}: {len(records)} records, "
              f"{len(addr_traces)} address traces", file=sys.stderr)

    print(f"Total: {total_records} records, "
          f"{len(all_traces)} address traces (after filters)",
          file=sys.stderr)

    if args.summary:
        print_summary(all_traces)

    # Output JSON
    output = [format_address_trace(t) for t in all_traces]

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
            f.write("\n")
        print(f"Wrote {len(output)} traces to {args.output}", file=sys.stderr)
    else:
        json.dump(output, sys.stdout, indent=2)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
