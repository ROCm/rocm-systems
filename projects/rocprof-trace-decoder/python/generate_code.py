#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rocprof_trace_decoder.codegen import CodeObject, generate_code_artifacts, llvm_objdump
from rocprof_trace_decoder.rcv import write_source_snapshots

CODE_SUFFIXES = {".co", ".hsaco", ".o", ".out"}


def _default_inputs() -> list[Path]:
    return sorted(
        path
        for path in Path.cwd().iterdir()
        if path.is_file() and path.suffix.lower() in CODE_SUFFIXES
    )


def _parse_codeobj_id_from_path(path: Path) -> int | None:
    stem = path.stem
    pos = stem.rfind("_")
    if pos == -1:
        return None
    try:
        return int(stem[pos + 1:])
    except ValueError:
        return None


def _code_objects_from_paths(paths: list[Path]) -> list[CodeObject]:
    parsed = [(path.expanduser().resolve(), _parse_codeobj_id_from_path(path)) for path in paths]
    untagged = [str(path) for path, codeobj_id in parsed if codeobj_id is None]
    parsed_ids = {codeobj_id for _path, codeobj_id in parsed if codeobj_id is not None}

    if len(untagged) > 1:
        raise ValueError(
            "Cannot infer code object IDs for multiple unnamed inputs: " + ", ".join(untagged)
        )
    if untagged and 0 in parsed_ids:
        raise ValueError(
            f"Cannot assign code object id 0 to {untagged[0]}: another input already uses id 0."
        )

    return [
        CodeObject(path=path, code_object_id=codeobj_id if codeobj_id is not None else 0)
        for path, codeobj_id in parsed
    ]


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
        artifacts = generate_code_artifacts(_code_objects_from_paths(paths))
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
