#!/usr/bin/env python3
from __future__ import annotations

import argparse

from common import add_common_args, decode_traces, load_inputs
from rocprof_trace_decoder import AnalysisFlags, Pc


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
    traces = decode_traces(
        inputs.att_paths,
        code_index=code_index,
        analysis=AnalysisFlags.HIDDEN_LATENCY,
    )

    hidden_by_pc: dict[Pc, list[int]] = {}
    for trace in traces:
        # The decoder reports one record per (SIMD, program counter) and one parse covers one
        # capture, so summing every record aggregates the run by program counter.
        for record in trace.records.hidden_latency:
            totals = hidden_by_pc.setdefault(record.pc, [0, 0, 0])
            totals[0] += record.idle
            totals[1] += record.stall
            totals[2] += record.issue
        for wave in trace.records.waves:
            code_index.accumulate_wave(wave)

    rows = []
    for entry in code_index.entries.values():
        total = entry.latency + entry.idle
        if not total:
            continue
        idle, stall, issue = hidden_by_pc.get(entry.pc, (0, 0, 0))
        hidden_total = stall + issue + min(idle, entry.idle)
        rows.append((total - hidden_total, total, hidden_total, entry))
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
