#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from marker_common import add_marker_args, load_marker_traces, marker_selected, marker_time
from rocprof_trace_decoder import MarkerRecordKind


@dataclass
class AddressTrace:
    kind: str
    time: int
    code_object_id: int
    marker_id: int
    source_location: str | None
    cu: int
    simd: int
    wave_id: int
    exec_mask: int
    addresses: list[int]


PREFIX = "addr_trace_"
MEMORY_KINDS = {"load", "store", "atomic"}
LDS_KINDS = {"lds_load", "lds_store", "lds_atomic", "ds_permute", "ds_bpermute"}
BUFFER_KINDS = {
    "buffer_load",
    "buffer_store",
    "buffer_atomic",
    "struct_buffer_load",
    "struct_buffer_store",
    "struct_buffer_atomic",
}


def match_kind(name: str | None):
    if not name or not name.startswith(PREFIX):
        return None
    kind = name[len(PREFIX) :]
    if kind in MEMORY_KINDS:
        return kind, True
    if kind in LDS_KINDS:
        return kind, False
    if kind in BUFFER_KINDS:
        return kind, None
    return None


def infer_wave_size(kind: str, is_64bit: bool | None, payload_count: int) -> int:
    if "buffer" in kind:
        fixed_count = 5
        words_per_lane = 2 if kind.startswith("struct_buffer") else 1
    else:
        fixed_count = 2
        words_per_lane = 2 if is_64bit else 1
    data_count = payload_count - fixed_count
    wave_size = (
        data_count // words_per_lane
        if data_count >= 0 and data_count % words_per_lane == 0
        else 0
    )
    return wave_size if wave_size in (32, 64) else 0


def decode_block(header, payloads) -> AddressTrace | None:
    matched = match_kind(header.name)
    if matched is None:
        return None
    kind, is_64bit = matched
    wave_size = infer_wave_size(kind, is_64bit, header.payload_count)
    if not wave_size or len(payloads) != header.payload_count:
        return None

    values = [payload.shaderdata.value & 0xFFFFFFFF for payload in payloads]
    exec_mask = values[0] | (values[1] << 32)
    values = values[2:]
    addresses: list[int] = []

    if kind.startswith("struct_buffer"):
        return None
    if "buffer" in kind:
        rsrc_lo, rsrc_hi, soffset = values[:3]
        voffsets = values[3 : 3 + wave_size]
        base = (rsrc_lo | (rsrc_hi << 32)) & 0xFFFFFFFFFFFF
        addresses = [
            base + soffset + voffsets[lane]
            for lane in range(wave_size)
            if (exec_mask >> lane) & 1
        ]
    elif is_64bit:
        addresses = [
            values[2 * lane] | (values[2 * lane + 1] << 32)
            for lane in range(wave_size)
            if (exec_mask >> lane) & 1
        ]
    else:
        addresses = [values[lane] for lane in range(wave_size) if (exec_mask >> lane) & 1]

    return AddressTrace(
        kind=kind,
        time=marker_time(header),
        code_object_id=header.code_object_id,
        marker_id=header.marker_id,
        source_location=header.source_location,
        cu=header.shaderdata.cu,
        simd=header.shaderdata.simd,
        wave_id=header.shaderdata.wave_id,
        exec_mask=exec_mask,
        addresses=addresses,
    )


def decode_address_traces(markers, args) -> list[AddressTrace]:
    active = {}
    traces = []
    for marker in markers:
        if not marker_selected(marker, args):
            continue
        key = (
            marker.code_object_id,
            marker.shaderdata.cu,
            marker.shaderdata.simd,
            marker.shaderdata.wave_id,
            marker.marker_id,
        )
        if marker.record_kind == MarkerRecordKind.HEADER:
            if match_kind(marker.name) is not None and marker.payload_count:
                active[key] = (marker, [None] * marker.payload_count)
            continue
        block = active.get(key)
        if block is None or marker.payload_index >= len(block[1]):
            continue
        block[1][marker.payload_index] = marker
        if all(payload is not None for payload in block[1]):
            trace = decode_block(block[0], block[1])
            if trace is not None:
                traces.append(trace)
            del active[key]
    return traces


def trace_json(trace: AddressTrace) -> dict:
    width = 8 if trace.kind.startswith(("lds_", "ds_")) else 16
    return {
        "kind": trace.kind,
        "time": trace.time,
        "code_object_id": trace.code_object_id,
        "marker_id": trace.marker_id,
        "source_location": trace.source_location,
        "cu": trace.cu,
        "simd": trace.simd,
        "wave_id": trace.wave_id,
        "exec_mask": f"0x{trace.exec_mask:016x}",
        "active_lanes": len(trace.addresses),
        "addresses": [f"0x{address:0{width}x}" for address in trace.addresses],
        "deltas": [b - a for a, b in zip(trace.addresses, trace.addresses[1:])],
    }


def print_summary(traces: list[AddressTrace]) -> None:
    by_kind: dict[str, int] = {}
    for trace in traces:
        by_kind[trace.kind] = by_kind.get(trace.kind, 0) + 1
    print(f"Address operations: {len(traces)}", file=sys.stderr)
    active_lanes = sum(len(trace.addresses) for trace in traces)
    print(f"Active-lane addresses: {active_lanes}", file=sys.stderr)
    for kind, count in sorted(by_kind.items()):
        print(f"  {kind}: {count}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Decode SQTT marker address-trace payloads.")
    add_marker_args(parser)
    parser.add_argument("-o", "--output", type=Path, help="Write JSON to this file.")
    parser.add_argument("--summary", action="store_true", help="Print summary statistics.")
    args = parser.parse_args()

    traces = []
    for decoded in load_marker_traces(args):
        traces.extend(decode_address_traces(decoded.records.markers, args))
    if args.summary:
        print_summary(traces)
    output = json.dumps([trace_json(trace) for trace in traces], indent=2)
    if args.output:
        args.output.write_text(output)
        print(f"Wrote {args.output}")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
