#!/usr/bin/env python3
from __future__ import annotations

import argparse

from common import (
    add_common_args,
    decode_traces,
    load_inputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print the hottest ISA instructions by latency + idle cycles."
    )
    add_common_args(parser, output_dir=False)
    args = parser.parse_args()

    inputs, artifacts = load_inputs(args)
    code_index = artifacts.code_index

    traces = decode_traces(
        inputs.att_paths,
        code_index=code_index,
    )
    for trace in traces:
        for wave in trace.records.waves:
            code_index.accumulate_wave(wave)

    rows = [
        entry
        for entry in code_index.entries.values()
        if entry.hitcount or entry.latency or entry.idle
    ]
    rows.sort(key=lambda entry: (entry.latency + entry.idle, entry.latency), reverse=True)
    top = rows[:30]

    print(
        f"{'Rank':>4} {'Score':>12} {'Latency':>12} {'Idle':>12} {'Hits':>8} "
        f"{'CodeObj':>7} {'Vaddr':>12}  Instruction"
    )
    for idx, entry in enumerate(top, 1):
        score = entry.latency + entry.idle
        print(
            f"{idx:4d} {score:12d} {entry.latency:12d} {entry.idle:12d} "
            f"{entry.hitcount:8d} {entry.pc.code_object_id:7d} "
            f"0x{entry.pc.address:010x}  {entry.inst}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
