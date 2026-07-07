from __future__ import annotations

import argparse
import glob
import sys
from dataclasses import dataclass
from pathlib import Path

from _code_object_paths import code_objects_from_paths
from rocprof_trace_decoder import (
    CodeArtifacts,
    CodeObject,
    Decoder,
    TraceRecords,
    generate_code_artifacts,
)

CODE_SUFFIXES = {".out", ".co", ".hsaco"}


@dataclass(frozen=True)
class DecodedTrace:
    path: Path
    records: TraceRecords


@dataclass(frozen=True)
class SampleInputs:
    att_paths: list[Path]
    code_objects: list[CodeObject]


def add_common_args(parser: argparse.ArgumentParser, *, output_dir: bool = True) -> None:
    if output_dir:
        parser.add_argument(
            "-d",
            "--dir",
            type=Path,
            default=Path("."),
            help="Directory for generated files.",
        )
    parser.add_argument(
        "files",
        nargs="+",
        help="Mixed .att and code object files (.out/.co/.hsaco), with optional globs.",
    )


def load_inputs(args: argparse.Namespace) -> tuple[SampleInputs, CodeArtifacts]:
    inputs = expand_input_paths(args.files)
    if not inputs.att_paths:
        raise SystemExit("At least one .att file is required.")
    if not inputs.code_objects:
        raise SystemExit("At least one code object (.out, .co, .hsaco) is required.")
    return inputs, generate_code_artifacts(inputs.code_objects)


def expand_input_paths(patterns: list[str]) -> SampleInputs:
    att_paths: list[Path] = []
    code_paths: list[Path] = []
    for pattern in patterns:
        expanded = str(Path(pattern).expanduser())
        matches = (
            sorted(glob.glob(expanded)) if any(ch in expanded for ch in "*?[]") else [expanded]
        )
        if not matches:
            raise SystemExit(f"No files matched: {pattern}")
        for item in matches:
            path = Path(item).expanduser().resolve()
            if not path.is_file():
                raise SystemExit(f"Input file does not exist: {path}")
            if path.suffix.lower() == ".att":
                att_paths.append(path)
            elif path.suffix.lower() in CODE_SUFFIXES:
                code_paths.append(path)
            else:
                raise SystemExit(f"Unsupported input type: {path}")

    try:
        code_objects = code_objects_from_paths(code_paths)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    return SampleInputs(att_paths=att_paths, code_objects=code_objects)


def decode_traces(
    att_paths: list[Path],
    *,
    code_index,
) -> list[DecodedTrace]:
    decoded: list[DecodedTrace] = []
    with Decoder() as decoder:
        for path in att_paths:
            records = decoder.parse(path.read_bytes(), isa=code_index)
            for info in records.info:
                print(f"Warning: {path}: {decoder.info_string(info)}", file=sys.stderr)
            decoded.append(DecodedTrace(path=path, records=records))
    return decoded


def prepare_output_dir(path: Path) -> Path:
    out_dir = path.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def instruction_text(artifacts: CodeArtifacts, pc) -> str:
    entry = artifacts.code_index.entries.get(pc)
    return entry.inst if entry is not None else ""


def wave_idle_time(wave) -> int:
    idle = 0
    prev_time = wave.begin_time
    for inst in wave.instructions:
        idle += max(inst.time - prev_time, 0)
        prev_time = max(prev_time, inst.time + inst.duration)
    return idle
