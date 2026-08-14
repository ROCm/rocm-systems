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
)
from rocprof_trace_decoder import MarkerRecordKind


def main() -> int:
    parser = argparse.ArgumentParser(description="Print decoded marker payload records as JSON.")
    add_marker_args(parser)
    args = parser.parse_args()

    rows = []
    for trace in load_marker_traces(args):
        for marker in trace.records.markers:
            if marker.record_kind != MarkerRecordKind.PAYLOAD:
                continue
            if not marker_selected(marker, args):
                continue
            rows.append(
                {
                    "trace": str(trace.path),
                    "time": marker_time(marker),
                    "code_object_id": marker.code_object_id,
                    "marker_id": marker.marker_id,
                    "name": marker_name(marker, args),
                    "source_location": marker.source_location,
                    "cu": marker.shaderdata.cu,
                    "simd": marker.shaderdata.simd,
                    "wave_id": marker.shaderdata.wave_id,
                    "payload_index": marker.payload_index,
                    "payload_count": marker.payload_count,
                    "value": marker.shaderdata.value,
                }
            )
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
