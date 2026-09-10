#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from common import add_common_args, decode_traces, load_inputs
from rocprof_trace_decoder import HiddenLatency, Pc, analyze_hidden_latency


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print ISA hotspots ranked by non-hidden latency."
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=30,
        help="Number of instructions to print (default: 30).",
    )
    add_common_args(parser, output_dir=False)
    args = parser.parse_args()

    inputs, artifacts = load_inputs(args)
    code_index = artifacts.code_index
    traces = decode_traces(inputs.att_paths, code_index=code_index)

    hidden_by_pc: dict[Pc, HiddenLatency] = {}
    for index, trace in enumerate(traces):
        # Scope keys only have to be distinct per capture.
        result = analyze_hidden_latency(
            {index: trace.records},
            code_index=code_index,
        )
        for pc, hidden in result.by_pc.items():
            hidden_by_pc.setdefault(pc, HiddenLatency())
            hidden_by_pc[pc] += hidden
        for wave in trace.records.waves:
            code_index.accumulate_wave(wave)

    rows = []
    overcount_count = 0
    maximum_excess = 0
    for entry in code_index.entries.values():
        total = entry.latency + entry.idle
        if not total:
            continue
        hidden = hidden_by_pc.get(entry.pc, HiddenLatency()).total()
        if hidden > total:
            overcount_count += 1
            maximum_excess = max(maximum_excess, hidden - total)
        non_hidden = max(total - hidden, 0)
        rows.append((non_hidden, total, hidden, entry))
    rows.sort(key=lambda row: (row[0], row[1], row[3].hitcount), reverse=True)

    print(
        f"{'Rank':>4} {'NonHidden':>12} {'Total':>12} {'Hidden':>12} "
        f"{'Hits':>8} {'CodeObj':>7} {'Vaddr':>12}  Instruction / Source"
    )
    for rank, (non_hidden, total, hidden, entry) in enumerate(rows[: args.limit], 1):
        detail = entry.inst
        if entry.source:
            detail += f"  [{entry.source}]"
        print(
            f"{rank:4d} {non_hidden:12d} {total:12d} {hidden:12d} "
            f"{entry.hitcount:8d} {entry.pc.code_object_id:7d} "
            f"0x{entry.pc.address:010x}  {detail}"
        )
    if overcount_count:
        print(
            f"Warning: hidden latency exceeded total latency for {overcount_count} PCs "
            f"(maximum excess: {maximum_excess} cycles); displayed non-hidden values "
            "were clamped to zero.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
