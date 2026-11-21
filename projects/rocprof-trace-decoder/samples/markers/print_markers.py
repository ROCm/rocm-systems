#!/usr/bin/env python3
from __future__ import annotations

import argparse

from marker_common import (
    add_marker_args,
    load_marker_traces,
    marker_name,
    marker_selected,
    marker_time,
)
from rocprof_trace_decoder import MarkerFlags, MarkerKind


def main() -> int:
    parser = argparse.ArgumentParser(description="Print decoded SQTT marker records.")
    add_marker_args(parser)
    args = parser.parse_args()

    for trace in load_marker_traces(args):
        print(f"== {trace.path} ==")
        for marker in trace.records.markers:
            if not marker_selected(marker, args):
                continue
            time = marker_time(marker)
            location = (
                f"cu={marker.shaderdata.cu} simd={marker.shaderdata.simd} "
                f"wave={marker.shaderdata.wave_id}"
            )
            flags = []
            if marker.marker_flags & MarkerFlags.NEW_WAVE:
                flags.append("new-wave")
            if marker.marker_flags & MarkerFlags.EXIT_PREVIOUS:
                flags.append("exit")
            if marker.marker_flags & MarkerFlags.ENTER:
                flags.append("enter")
            if marker.marker_flags & MarkerFlags.INVALID:
                flags.append("invalid")
            if marker.payload_dropped:
                flags.append(f"dropped={marker.payload_dropped}")
            kind = (
                marker.marker_kind.name.lower()
                if isinstance(marker.marker_kind, MarkerKind)
                else f"unknown({marker.marker_kind})"
            )
            print(
                f"{time:12d} {location} "
                f"id={marker.marker_id} kind={kind} "
                f"flags={','.join(flags) or '-'} {marker_name(marker, args)}"
            )
            for index, payload in enumerate(marker.payload):
                print(
                    "    "
                    f"payload[{index}/{len(marker.payload) + marker.payload_dropped}]="
                    f"0x{payload:08x}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
