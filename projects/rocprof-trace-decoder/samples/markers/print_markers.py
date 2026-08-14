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
from rocprof_trace_decoder import MarkerFlags, MarkerRecordKind


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
            if marker.record_kind == MarkerRecordKind.PAYLOAD:
                print(
                    f"{time:12d} {location} "
                    f"payload[{marker.payload_index}/{marker.payload_count}]="
                    f"0x{marker.shaderdata.value:08x} {marker_name(marker, args)}"
                )
                continue

            flags = []
            if marker.marker_flags & MarkerFlags.NEW_WAVE:
                flags.append("new-wave")
            if marker.marker_flags & MarkerFlags.EXIT_PREVIOUS:
                flags.append("exit")
            if marker.marker_flags & MarkerFlags.ENTER:
                flags.append("enter")
            print(
                f"{time:12d} {location} "
                f"id={marker.marker_id} kind={marker.marker_kind.name.lower()} "
                f"flags={','.join(flags) or '-'} {marker_name(marker, args)}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
