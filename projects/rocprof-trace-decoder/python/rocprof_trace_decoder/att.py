from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .bindings import Decoder
from .code_index import CodeIndex, copy_snapshots
from .rcv import RcvOutputWriter

ATT_RE = re.compile(r"_shader_engine_(\d+)_(\d+)\.att$")
CODE_SUFFIXES = {".out", ".co", ".hsaco"}


def generate_att_outputs(
    files: Iterable[str | Path],
    *,
    output_dir: str | Path | None = None,
    lib_path: str | Path | None = None,
    formats: str = "json,csv",
    force_codegen: bool = False,
    base_name: str | None = None,
) -> list[Path]:
    inputs = _discover_inputs(files)
    if not inputs.att_files:
        raise SystemExit("No .att files were provided or discovered.")

    runs = _group_att_files(inputs.att_files)
    base_dir = Path(output_dir) if output_dir else inputs.att_files[0].parent
    output_base_name = base_name or inputs.att_files[0].parent.name or "att"

    code_json = inputs.code_json
    code_dir = base_dir if output_dir else inputs.att_files[0].parent
    if code_json is None or force_codegen:
        if not inputs.code_objects:
            raise SystemExit("No code.json or code object files (.out, .co, .hsaco) were provided.")
        code_dir.mkdir(parents=True, exist_ok=True)
        code_json = _run_generate_code(inputs.code_objects, code_dir)

    output_dirs: list[Path] = []
    for run, att_files in runs.items():
        run_dir = base_dir if output_dir and len(runs) == 1 else base_dir / f"ui_output_{output_base_name}{run}"
        run_dir.mkdir(parents=True, exist_ok=True)

        code_index = CodeIndex.from_code_json(code_json)
        writer = RcvOutputWriter(run_dir, code_index, formats=formats)

        with Decoder(lib_path) as decoder:
            for att_file in att_files:
                records = decoder.parse_file(att_file, isa=code_index)
                writer.add_shader_records(shader_engine_id(att_file), records)

        writer.finish()
        code_index.write_code_json(run_dir / "code.json")
        if "csv" in formats.lower():
            code_index.write_stats_csv(run_dir.parent / f"stats_{run_dir.name}.csv")
        copy_snapshots(Path(code_json).parent, run_dir)
        output_dirs.append(run_dir)

    return output_dirs


class _Inputs:
    def __init__(self) -> None:
        self.att_files: list[Path] = []
        self.code_objects: list[Path] = []
        self.code_json: Path | None = None


def _discover_inputs(files: Iterable[str | Path]) -> _Inputs:
    out = _Inputs()
    for raw in files:
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            for child in sorted(path.iterdir()):
                _add_input(out, child)
        else:
            _add_input(out, path)
    return out


def _add_input(out: _Inputs, path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Input does not exist: {path}")
    if path.suffix == ".att":
        out.att_files.append(path)
    elif path.name == "code.json":
        out.code_json = path
    elif path.suffix in CODE_SUFFIXES:
        out.code_objects.append(path)


def _group_att_files(att_files: Iterable[Path]) -> dict[int, list[Path]]:
    grouped: dict[int, list[Path]] = defaultdict(list)
    for path in sorted(att_files):
        match = ATT_RE.search(path.name)
        grouped[int(match.group(2)) if match else 1].append(path)
    return dict(sorted(grouped.items()))


def shader_engine_id(path: str | Path) -> int:
    match = ATT_RE.search(Path(path).name)
    if not match:
        raise ValueError(f"Could not parse shader engine id from {path}")
    return int(match.group(1))


def _run_generate_code(code_objects: list[Path], output_dir: Path) -> Path:
    script = Path(__file__).resolve().parents[1] / "generate_code.py"
    cmd = [sys.executable, str(script), *[str(p) for p in code_objects]]
    subprocess.run(cmd, cwd=output_dir, check=True)
    code_json = output_dir / "code.json"
    if not code_json.exists():
        raise RuntimeError(f"generate_code.py did not produce {code_json}")
    return code_json


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decode ATT traces and generate ROCprof Compute Viewer CSV/JSON files."
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Mixed list of .att files, code objects (.out/.co/.hsaco), code.json, or directories.",
    )
    parser.add_argument("--output-dir", help="Output directory. For multiple runs, subdirectories are created here.")
    parser.add_argument("--lib", help="Path to librocprof-trace-decoder.so")
    parser.add_argument("--formats", default="json,csv", help="Comma-separated outputs. Default: json,csv")
    parser.add_argument("--force-codegen", action="store_true", help="Regenerate code.json even if one is provided.")
    parser.add_argument(
        "--base-name",
        help="Base name for default ui_output_<name><run> and stats_<...>.csv output names.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    outputs = generate_att_outputs(
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
