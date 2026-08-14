#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from marker_common import (
    add_marker_args,
    load_marker_traces,
    marker_name,
    marker_selected,
    marker_time,
    prepare_output_dir,
)
from rocprof_trace_decoder import MarkerFlags, MarkerKind, MarkerRecordKind


def trace_events(markers, args=None, clock_rate_ghz: float | None = None) -> list[dict]:
    events = []
    depths: dict[tuple[int, int], int] = {}
    last_base = {}
    for marker in sorted(markers, key=marker_time):
        if marker.record_kind != MarkerRecordKind.HEADER:
            continue
        if not marker_selected(marker, args):
            continue
        time = marker_time(marker)
        if clock_rate_ghz:
            time /= clock_rate_ghz * 1000.0
        base = {
            "pid": marker.code_object_id,
            "tid": (
                (marker.shaderdata.cu << 16)
                | (marker.shaderdata.simd << 8)
                | marker.shaderdata.wave_id
            ),
            "ts": time,
        }
        key = (base["pid"], base["tid"])
        last_base[key] = base
        depth = depths.get(key, 0)
        if marker.marker_flags & MarkerFlags.NEW_WAVE:
            for _ in range(depth):
                events.append({**base, "ph": "E"})
            depth = 0
        if marker.marker_flags & MarkerFlags.EXIT_PREVIOUS:
            if depth:
                events.append({**base, "ph": "E"})
                depth -= 1
        if marker.marker_flags & MarkerFlags.ENTER:
            events.append({**base, "ph": "B", "name": marker_name(marker, args)})
            depth += 1
        elif marker.marker_kind == MarkerKind.POINT:
            events.append(
                {
                    **base,
                    "ph": "i",
                    "s": "t",
                    "name": marker_name(marker, args),
                }
            )
        depths[key] = depth
    for key, depth in depths.items():
        for _ in range(depth):
            events.append({**last_base[key], "ph": "E"})
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description="Export decoded SQTT markers to Perfetto JSON.")
    add_marker_args(parser, output_dir=True)
    parser.add_argument(
        "--clock-rate-ghz", type=float, help="Convert shader cycles to microseconds."
    )
    args = parser.parse_args()

    output_dir = prepare_output_dir(args.dir)
    for trace in load_marker_traces(args):
        output = output_dir / f"{trace.path.stem}_markers.json"
        output.write_text(
            json.dumps(
                {"traceEvents": trace_events(trace.records.markers, args, args.clock_rate_ghz)},
                indent=2,
            )
        )
        print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
