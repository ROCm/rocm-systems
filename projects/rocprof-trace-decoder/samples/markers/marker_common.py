from __future__ import annotations

import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from rocprof_trace_decoder import Decoder, RecordType

SAMPLES_DIR = Path(__file__).resolve().parents[1]
if str(SAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(SAMPLES_DIR))

from common import DecodedTrace, add_common_args, load_inputs, prepare_output_dir


def add_marker_args(parser, *, output_dir: bool = False) -> None:
    add_common_args(parser, output_dir=output_dir)
    parser.add_argument("--cu", type=int, help="Only include this CU/WGP.")
    parser.add_argument("--simd", type=int, help="Only include this SIMD.")
    parser.add_argument("--wave", type=int, help="Only include this wave slot.")
    parser.add_argument(
        "--demangle", action="store_true", help="Demangle marker names with c++filt."
    )


def load_marker_traces(args):
    expanded = []
    for value in args.files:
        path = Path(value).expanduser()
        if not path.is_dir():
            expanded.append(value)
            continue
        expanded.extend(str(item) for item in sorted(path.rglob("*.att")))
        for suffix in ("*.out", "*.co", "*.hsaco"):
            expanded.extend(
                str(item)
                for item in sorted(path.rglob(suffix))
                if "code_object_id" in item.name
            )
    args.files = expanded

    inputs, artifacts = load_inputs(args)
    unnamed = [
        str(code_object.path)
        for code_object in inputs.code_objects
        if code_object.code_object_id == 0
    ]
    if unnamed:
        raise SystemExit(
            "Marker decoding requires nonzero code-object IDs in capture filenames: "
            + ", ".join(unnamed)
        )
    traces = []
    with Decoder() as decoder:
        decoder.set_record_filter([RecordType.INFO, RecordType.MARKER])
        for code_object in inputs.code_objects:
            path = Path(code_object.path)
            decoder.load_code_object_file(
                path,
                load_id=code_object.code_object_id,
                load_addr=0,
                load_size=max(path.stat().st_size, 1),
            )
        for path in inputs.att_paths:
            records = decoder.parse(path.read_bytes(), isa=artifacts.code_index)
            for info in records.info:
                print(f"Warning: {path}: {decoder.info_string(info)}", file=sys.stderr)
            traces.append(DecodedTrace(path=path, records=records))
    return traces


def marker_time(marker) -> int:
    return marker.shaderdata.time - marker.delay


def marker_selected(marker, args=None) -> bool:
    return args is None or (
        (args.cu is None or marker.shaderdata.cu == args.cu)
        and (args.simd is None or marker.shaderdata.simd == args.simd)
        and (args.wave is None or marker.shaderdata.wave_id == args.wave)
    )


@lru_cache(maxsize=None)
def demangle(name: str) -> str:
    try:
        result = subprocess.run(["c++filt", name], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return name


def marker_name(marker, args=None) -> str:
    name = marker.name or f"marker#{marker.marker_id}"
    return demangle(name) if args is not None and args.demangle else name


__all__ = [
    "add_marker_args",
    "load_marker_traces",
    "marker_name",
    "marker_selected",
    "marker_time",
    "prepare_output_dir",
]
