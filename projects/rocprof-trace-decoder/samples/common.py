from __future__ import annotations

import argparse
import glob
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from rocprof_trace_decoder import (
    CodeArtifacts,
    CodeObject,
    Decoder,
    TraceRecords,
    generate_code_artifacts,
)

CODE_SUFFIXES = {".out", ".co", ".hsaco"}
CODE_OBJECT_ID_RE = re.compile(r"(?:^|_)code_object_id_(\d+)(?:_|$)", re.IGNORECASE)


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
        matches = sorted(glob.glob(pattern)) if any(ch in pattern for ch in "*?[]") else [pattern]
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

    code_objects = _code_objects_from_paths(code_paths)
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


def _code_objects_from_paths(paths: list[Path]) -> list[CodeObject]:
    parsed = [(path, _code_object_id_from_path(path)) for path in paths]
    untagged = [str(path) for path, code_object_id in parsed if code_object_id is None]
    parsed_ids = {code_object_id for _path, code_object_id in parsed if code_object_id is not None}
    if len(untagged) > 1:
        raise SystemExit(
            "Cannot infer code object ids for multiple unnamed inputs: "
            + ", ".join(untagged)
        )
    if untagged and 0 in parsed_ids:
        raise SystemExit(
            f"Cannot assign code object id 0 to {untagged[0]}: another input already uses id 0."
        )
    return [
        CodeObject(path=path, code_object_id=code_object_id if code_object_id is not None else 0)
        for path, code_object_id in parsed
    ]


def _code_object_id_from_path(path: Path) -> int | None:
    match = CODE_OBJECT_ID_RE.search(path.stem)
    if match:
        return int(match.group(1))

    pos = path.stem.rfind("_")
    if pos == -1:
        return None
    try:
        return int(path.stem[pos + 1:])
    except ValueError:
        return None
