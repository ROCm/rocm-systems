#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _code_object_paths import code_objects_from_paths
from rocprof_trace_decoder.att import AttTrace, generate_att_outputs
from rocprof_trace_decoder.code_index import CodeIndex
from rocprof_trace_decoder.codegen import generate_code_artifacts
from rocprof_trace_decoder.rcv import (
    copy_source_snapshots,
    write_code_json,
    write_source_snapshots,
)

ATT_RE = re.compile(r"_shader_engine_(\d+)_(\d+)\.att$", re.IGNORECASE)
CODE_SUFFIXES = {".out", ".co", ".hsaco"}


@dataclass
class _Inputs:
    att_files: list[Path] = field(default_factory=list)
    code_objects: list[Path] = field(default_factory=list)
    code_json: Path | None = None


def generate_outputs_from_files(
    files: Iterable[str | Path],
    *,
    output_dir: str | Path | None = None,
    lib_path: str | Path | None = None,
    formats: str = "json,csv",
    base_name: str | None = None,
) -> list[Path]:
    inputs = _discover_inputs(files)
    if not inputs.att_files and not inputs.code_objects and inputs.code_json is None:
        raise SystemExit(
            "No .att files, code.json, or code object files were provided or discovered."
        )

    base_input = _base_input(inputs)
    output_base_name = base_name or base_input.parent.name or "att"
    decode_output_dir = Path(output_dir).expanduser().resolve() if output_dir else None

    code_json = inputs.code_json
    code_index = None
    source_paths: tuple[Path, ...] = ()
    snapshot_source_dir = code_json.parent if code_json else None
    code_dir = Path(output_dir).expanduser().resolve() if output_dir else _code_dir(inputs)
    generated_code = False
    if inputs.code_objects:
        try:
            code_objects = code_objects_from_paths(inputs.code_objects)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        artifacts = generate_code_artifacts(code_objects)
        code_index = artifacts.code_index
        source_paths = artifacts.source_paths
        code_json = None
        snapshot_source_dir = None
        generated_code = True
    elif code_json is not None:
        code_index = CodeIndex.from_code_json(code_json, load_counts=False)

    if not inputs.att_files:
        if code_index is not None:
            if generated_code or output_dir:
                code_dir.mkdir(parents=True, exist_ok=True)
                write_code_json(code_dir, code_index)
                write_source_snapshots(source_paths, code_dir)
                return [code_dir]
            return [code_json.parent] if code_json else []
        return [code_json.parent] if code_json else []

    traces = _traces_from_att_paths(inputs.att_files)
    outputs = generate_att_outputs(
        traces,
        code_index=code_index,
        source_paths=source_paths,
        output_dir=decode_output_dir,
        lib_path=lib_path,
        formats=formats,
        base_name=output_base_name,
        on_warning=_print_warning,
    )
    if snapshot_source_dir:
        for output in outputs:
            copy_source_snapshots(snapshot_source_dir, output)
    return outputs


def _print_warning(path: Path, message: str) -> None:
    print(f"Warning: {path}: {message}", file=sys.stderr)


def _discover_inputs(files: Iterable[str | Path]) -> _Inputs:
    out = _Inputs()
    for raw in files:
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            for child in sorted(path.iterdir()):
                _add_input(out, child, ignore_unknown=True)
        else:
            _add_input(out, path)
    return out


def _base_input(inputs: _Inputs) -> Path:
    if inputs.att_files:
        return inputs.att_files[0]
    if inputs.code_objects:
        return inputs.code_objects[0]
    if inputs.code_json:
        return inputs.code_json
    raise SystemExit("No input files were provided or discovered.")


def _code_dir(inputs: _Inputs) -> Path:
    if inputs.code_objects:
        return inputs.code_objects[0].parent
    return _base_input(inputs).parent


def _add_input(out: _Inputs, path: Path, *, ignore_unknown: bool = False) -> None:
    if not path.exists():
        raise SystemExit(f"Input does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".att":
        out.att_files.append(path)
    elif path.name.lower() == "code.json":
        out.code_json = path
    elif suffix in CODE_SUFFIXES:
        out.code_objects.append(path)
    elif not ignore_unknown:
        raise SystemExit(
            f"Unsupported input type: {path}. Expected .att, code.json, or a code object "
            "(.out, .co, .hsaco)."
        )


def _traces_from_att_paths(paths: Iterable[Path]) -> list[AttTrace]:
    parsed: list[tuple[Path, tuple[int, int] | None]] = []
    used_shader_engines: dict[int, set[int]] = defaultdict(set)

    for path in paths:
        metadata = _trace_metadata_from_name(path)
        parsed.append((path, metadata))
        if metadata is not None:
            shader_engine, run = metadata
            used_shader_engines[run].add(shader_engine)

    next_shader_engine: dict[int, int] = defaultdict(int)
    traces: list[AttTrace] = []
    for path, metadata in parsed:
        if metadata is None:
            run = 1
            shader_engine = _next_available_shader_engine(
                used_shader_engines[run],
                next_shader_engine[run],
            )
            used_shader_engines[run].add(shader_engine)
            next_shader_engine[run] = shader_engine + 1
        else:
            shader_engine, run = metadata
        traces.append(AttTrace(path=path, shader_engine=shader_engine, run=run))
    return traces


def _trace_metadata_from_name(path: Path) -> tuple[int, int] | None:
    match = ATT_RE.search(path.name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _next_available_shader_engine(used: set[int], start: int) -> int:
    shader_engine = start
    while shader_engine in used:
        shader_engine += 1
    return shader_engine


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decode ATT traces and/or generate ROCprof Compute Viewer code metadata."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Mixed list of .att files, code objects (.out/.co/.hsaco), code.json, or directories.",
    )
    parser.add_argument(
        "-d",
        "--dir",
        dest="output_dir",
        help="Output directory. For multiple runs, subdirectories are created here.",
    )
    parser.add_argument("--lib", help="Path to librocprof-trace-decoder.so")
    parser.add_argument(
        "--formats",
        default="json,csv",
        help="Comma-separated outputs. Default: json,csv",
    )
    parser.add_argument(
        "--base-name",
        help="Base name for default ui_output_<name><run> and stats_<...>.csv output names.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    outputs = generate_outputs_from_files(
        args.files,
        output_dir=args.output_dir,
        lib_path=args.lib,
        formats=args.formats,
        base_name=args.base_name,
    )
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
