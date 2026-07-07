#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _code_object_paths import code_objects_from_paths
from rocprof_trace_decoder.codegen import generate_code_artifacts, llvm_objdump
from rocprof_trace_decoder.rcv import write_source_snapshots

CODE_SUFFIXES = {".co", ".hsaco", ".o", ".out"}


def _default_inputs() -> list[Path]:
    return sorted(
        path
        for path in Path.cwd().iterdir()
        if path.is_file() and path.suffix.lower() in CODE_SUFFIXES
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate code.json and source snapshots from GPU code objects."
    )
    parser.add_argument("code_objects", nargs="*", help="ELF code objects (.hsaco, .out, .co, .o)")
    args = parser.parse_args(argv)

    paths = [Path(path) for path in args.code_objects] or _default_inputs()
    if not paths:
        print(
            "No code object given and no .hsaco/.out/.co/.o found in current directory.",
            file=sys.stderr,
        )
        return 1

    try:
        artifacts = generate_code_artifacts(code_objects_from_paths(paths))
        artifacts.code_index.write_code_json(Path.cwd() / "code.json")
        num_snapshots = write_source_snapshots(artifacts.source_paths, Path.cwd())
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Using llvm-objdump: {llvm_objdump()}")
    print(f"wrote {Path.cwd() / 'code.json'}")
    if num_snapshots:
        print(f"wrote {Path.cwd() / 'snapshots.json'}")
    else:
        print("no source files snapshotted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
