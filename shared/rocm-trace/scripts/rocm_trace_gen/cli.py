# Copyright Advanced Micro Devices, Inc. SPDX-License-Identifier: MIT

"""Command line: generate the probes, or check the checked-in ones."""

from __future__ import annotations

import argparse
import difflib
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from . import __version__
from .config import ConfigError, load
from .emit import GenerateError, generate
from .parse import ParseError, parse_tables


def _arguments(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rocm_trace_gen.py",
        description=(
            "Generate TraceLogging probes for every entry of a runtime's dispatch "
            "tables, as described by a configuration file."
        ),
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="the runtime's YAML configuration"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="directory of the generated .h and .cpp",
    )
    parser.add_argument(
        "-I",
        dest="include_dirs",
        action="append",
        default=[],
        metavar="DIR",
        help="include directory for the runtime headers (repeatable)",
    )
    parser.add_argument(
        "-D",
        dest="defines",
        action="append",
        default=[],
        metavar="NAME[=VALUE]",
        help="preprocessor definition (repeatable)",
    )
    parser.add_argument(
        "--resource-dir",
        type=Path,
        help="clang's resource directory, for its builtin headers "
        "(the output of 'clang -print-resource-dir')",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; fail if the files in --output-dir differ from what would be generated",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser.parse_args(argv)


def _clang_args(args: argparse.Namespace) -> List[str]:
    out: List[str] = []
    if args.resource_dir is not None:
        out += ["-resource-dir", str(args.resource_dir)]
    out += [f"-I{d}" for d in args.include_dirs]
    out += [f"-D{d}" for d in args.defines]
    return out


def _check(files: dict, output_dir: Path) -> int:
    stale = 0
    for name, text in sorted(files.items()):
        path = output_dir / name
        current = path.read_text(encoding="utf-8") if path.exists() else ""
        if current == text:
            continue
        stale += 1
        diff = difflib.unified_diff(
            current.splitlines(keepends=True),
            text.splitlines(keepends=True),
            fromfile=f"{path} (checked in)",
            tofile=f"{path} (generated)",
            n=2,
        )
        sys.stdout.writelines(list(diff)[:200])
    if stale:
        print(
            f"error: {stale} generated file(s) are out of date; regenerate them",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _arguments(argv)
    try:
        config = load(args.config)
        tables = parse_tables(
            config.includes,
            [(t.name, t.size_member) for t in config.tables],
            _clang_args(args),
        )
        files = generate(config, tables)
    except (ConfigError, ParseError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    except GenerateError as err:
        for e in err.errors:
            print(f"error: {e}", file=sys.stderr)
        print(f"{len(err.errors)} error(s)", file=sys.stderr)
        return 1

    if args.check:
        return _check(files, args.output_dir)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, text in sorted(files.items()):
        path = args.output_dir / name
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            with path.open("w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            print(f"wrote {path}")
    return 0
