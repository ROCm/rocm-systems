#!/usr/bin/env python3
"""Generate a ConSan kernel allowlist from rocprofv3 kernel-trace CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys


class AllowlistError(RuntimeError):
    """The profiler output cannot produce a valid allowlist."""


def _trace_files(inputs: list[Path]) -> tuple[Path, ...]:
    files: set[Path] = set()
    for input_path in inputs:
        if input_path.is_dir():
            files.update(input_path.rglob("*kernel_trace.csv"))
        elif input_path.is_file():
            files.add(input_path)
        else:
            raise AllowlistError(f"input does not exist: {input_path}")
    if not files:
        raise AllowlistError("no rocprofv3 kernel-trace CSV files found")
    return tuple(sorted(files))


def collect_kernel_names(inputs: list[Path]) -> tuple[str, ...]:
    names: set[str] = set()
    for trace_file in _trace_files(inputs):
        with trace_file.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or "Kernel_Name" not in reader.fieldnames:
                raise AllowlistError(
                    f"{trace_file}: missing rocprofv3 Kernel_Name column"
                )
            for line_number, record in enumerate(reader, 2):
                if record.get("Kind", "KERNEL_DISPATCH") != "KERNEL_DISPATCH":
                    continue
                name = record.get("Kernel_Name", "").strip()
                if not name:
                    raise AllowlistError(
                        f"{trace_file}:{line_number}: empty kernel name"
                    )
                if "\n" in name or "\r" in name:
                    raise AllowlistError(
                        f"{trace_file}:{line_number}: kernel name contains a newline"
                    )
                if name.endswith(".kd"):
                    name = name[:-3]
                names.add(name)
    if not names:
        raise AllowlistError("kernel trace contains no GPU dispatches")
    return tuple(sorted(names))


def write_allowlist(output: Path, names: tuple[str, ...]) -> None:
    output.write_text("".join(f"{name}\n" for name in names), encoding="utf-8")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        type=Path,
        nargs="+",
        help="rocprofv3 kernel-trace CSV file or output directory",
    )
    parser.add_argument("-o", "--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    try:
        names = collect_kernel_names(args.inputs)
        write_allowlist(args.output, names)
    except (AllowlistError, OSError, csv.Error) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"wrote {len(names)} kernel names to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
