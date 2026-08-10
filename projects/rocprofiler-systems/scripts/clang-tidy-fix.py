#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Apply clang-tidy's automatic fixes to files or whole directory trees.

Not every check has a fixit (e.g. cppcoreguidelines-avoid-magic-numbers
does not), so this reports what's left after fixing what it can.
"""

import argparse
import concurrent.futures
import os
import re
import subprocess
import sys
from dataclasses import dataclass

SOURCE_EXTENSION_RE = re.compile(
    r".*\.(cpp|cc|c\+\+|cxx|c|cl|h|hpp|m|mm|inc)$", re.IGNORECASE
)
EXCLUDED_DIR_NAMES = {"external", "build"}

DIAGNOSTIC_RE = re.compile(r"^(.*):(\d+):(\d+): (warning|error): (.*) \[([\w,.\-]+)\]$")

_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "yellow": "\033[33m",
    "green": "\033[32m",
}


def _color(text: str, *codes: str) -> str:
    """Wrap `text` in ANSI codes when the output is not a dumb terminal."""
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return text
    prefix = "".join(_ANSI[c] for c in codes if c in _ANSI)
    return f"{prefix}{text}{_ANSI['reset']}"


@dataclass
class Diagnostic:
    """A single clang-tidy diagnostic, with its location and message."""

    file: str
    line: int
    col: int
    checks: list[str]
    message: str


@dataclass
class FileResult:
    path: str
    before_count: int
    fixed_count: int
    remaining: list[Diagnostic]


def expand_paths(paths: list[str]) -> list[str]:
    """Expand directories into the source files they contain, recursively."""
    files: list[str] = []
    for path in paths:
        if not os.path.isdir(path):
            files.append(path)
            continue
        for root, dirs, filenames in os.walk(path):
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIR_NAMES]
            for filename in filenames:
                if SOURCE_EXTENSION_RE.match(filename):
                    files.append(os.path.join(root, filename))
    return files


def run_clang_tidy(
    args: argparse.Namespace, file_path: str, fix: bool
) -> list[Diagnostic]:
    command = [args.clang_tidy_binary]
    if fix:
        command.append("-fix")
    if args.checks:
        command.append(f"-checks={args.checks}")
    command.append(f"-p={args.build_path}")
    command.append(file_path)

    result = subprocess.run(command, capture_output=True, text=True)
    diagnostics = []
    for line in (result.stdout + result.stderr).splitlines():
        match = DIAGNOSTIC_RE.match(line)
        if not match:
            continue
        file_name, line_no, col_no, _severity, message, checks = match.groups()
        diagnostics.append(
            Diagnostic(
                file=file_name,
                line=int(line_no),
                col=int(col_no),
                checks=checks.split(","),
                message=message,
            )
        )
    return diagnostics


def fix_file(args: argparse.Namespace, file_path: str) -> FileResult:
    before = run_clang_tidy(args, file_path, fix=False)
    run_clang_tidy(args, file_path, fix=True)
    after = run_clang_tidy(args, file_path, fix=False)
    return FileResult(
        path=file_path,
        before_count=len(before),
        fixed_count=max(len(before) - len(after), 0),
        remaining=after,
    )


def print_result(result: FileResult) -> None:
    print(f"{result.path}: {result.before_count} found, " f"fixed {result.fixed_count}")
    if not result.remaining:
        return
    for diagnostic in result.remaining:
        checks = ",".join(diagnostic.checks)
        location = f"{diagnostic.file}:{diagnostic.line}:{diagnostic.col}"
        print(f"  {_color(location, 'yellow')}: {diagnostic.message} [{checks}]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Source files and/or directories to fix")
    parser.add_argument(
        "--build-path",
        required=True,
        help="Directory containing compile_commands.json",
    )
    parser.add_argument(
        "--clang-tidy-binary",
        default="clang-tidy",
        help="Path to the clang-tidy binary (default: clang-tidy)",
    )
    parser.add_argument(
        "--checks",
        default="",
        help="Checks filter override (default: use .clang-tidy)",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of files to fix in parallel",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = expand_paths(args.paths)
    if not files:
        print("No source files found.")
        return 0

    results: list[FileResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(fix_file, args, file_path) for file_path in files]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print_result(result)

    total_fixed = sum(r.fixed_count for r in results)
    total_remaining = sum(len(r.remaining) for r in results)
    print()
    print(
        f"{len(results)} file(s) processed, {total_fixed} issue(s) fixed, "
        f"{total_remaining} remaining without an auto-fix"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
