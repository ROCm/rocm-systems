#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from rocprof_trace_decoder.att import AttTrace, generate_att_outputs

ATT_RE = re.compile(r"_shader_engine_(\d+)_(\d+)\.att$", re.IGNORECASE)
CODE_SUFFIXES = {".out", ".co", ".hsaco"}


class _Inputs:
    def __init__(self) -> None:
        self.att_files: list[Path] = []
        self.code_objects: list[Path] = []
        self.code_json: Path | None = None


def generate_outputs_from_files(
    files: Iterable[str | Path],
    *,
    output_dir: str | Path | None = None,
    lib_path: str | Path | None = None,
    formats: str = "json,csv",
    force_codegen: bool = False,
    base_name: str | None = None,
) -> list[Path]:
    inputs = _discover_inputs(files)
    if not inputs.att_files and not inputs.code_objects and inputs.code_json is None:
        raise SystemExit("No .att files, code.json, or code object files were provided or discovered.")

    base_input = _base_input(inputs)
    output_base_name = base_name or base_input.parent.name or "att"
    base_dir = Path(output_dir).expanduser().resolve() if output_dir else base_input.parent

    code_json = inputs.code_json
    code_dir = base_dir if output_dir else _code_dir(inputs)
    generated_code = False
    if inputs.code_objects and (code_json is None or force_codegen):
        code_dir.mkdir(parents=True, exist_ok=True)
        code_json = _run_generate_code(inputs.code_objects, code_dir)
        generated_code = True

    if not inputs.att_files:
        if generated_code:
            return [code_dir]
        return [code_json.parent] if code_json else []

    traces = _traces_from_att_paths(inputs.att_files)
    return generate_att_outputs(
        traces,
        code_json=code_json,
        output_dir=base_dir,
        lib_path=lib_path,
        formats=formats,
        base_name=output_base_name,
        on_warning=_print_warning,
    )


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


def _run_generate_code(code_objects: list[Path], output_dir: Path) -> Path:
    script = Path(__file__).resolve().parent / "generate_code.py"
    cmd = [sys.executable, str(script), *[str(p) for p in code_objects]]
    subprocess.run(cmd, cwd=output_dir, check=True)
    code_json = output_dir / "code.json"
    if not code_json.exists():
        raise RuntimeError(f"generate_code.py did not produce {code_json}")
    return code_json


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
        "--force-codegen",
        action="store_true",
        help="Regenerate code.json even if one is provided.",
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
        force_codegen=args.force_codegen,
        base_name=args.base_name,
    )
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
