#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check that banned headers are not included directly.

Reads the banned header list from banned-includes.json and scans the project
sources for a direct `#include` of any listed header, in either the angled or
the quoted form. Indirect (transitive) inclusion is allowed.

To ban another header, add an entry to banned-includes.json.
"""

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LIST = Path(__file__).with_name("banned-includes.json")
DEFAULT_ROOTS = ("source", "examples", "tests")
SOURCE_EXTENSION_RE = re.compile(r".*\.(h|hpp|c|cpp)(\.in)?$")
EXCLUDED_DIRS = frozenset({"external", "tpls"})

_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "yellow": "\033[33m",
}


def _color(text: str, *codes: str) -> str:
    """Color `text` with ANSI codes.

    Skips coloring when NO_COLOR is set or the terminal can't render them
    (TERM=dumb).
    """
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return text
    prefix = "".join(_ANSI[c] for c in codes if c in _ANSI)
    return f"{prefix}{text}{_ANSI['reset']}"


@dataclass
class BannedHeader:
    """A header that must not be included directly."""

    header: str
    hint: str
    pattern: re.Pattern[str]


@dataclass
class Violation:
    """A direct include of a banned header."""

    path: str
    line: int
    banned: BannedHeader


def load_banned_headers(list_path: Path) -> list[BannedHeader]:
    """Parse the banned header list; exit with a clear error if it is invalid."""
    try:
        raw = json.loads(list_path.read_text(encoding="utf-8"))
    except OSError as exc:
        sys.exit(f"error: cannot read {list_path}: {exc.strerror}")
    except json.JSONDecodeError as exc:
        sys.exit(f"error: {list_path} is not valid JSON: {exc}")

    if not isinstance(raw, list):
        sys.exit(f"error: {list_path} must contain a list of objects")

    banned = []
    for index, entry in enumerate(raw):
        where = f"{list_path} entry {index}"
        if not isinstance(entry, dict):
            sys.exit(f"error: {where} must be an object")
        header = entry.get("header")
        if not isinstance(header, str) or not header:
            sys.exit(f"error: {where} is missing a non-empty 'header'")
        hint = entry.get("hint", "")
        if not isinstance(hint, str):
            sys.exit(f"error: {where} has a non-string 'hint'")
        pattern = re.compile(rf'#\s*include\s*[<"]{re.escape(header)}[>"]')
        banned.append(BannedHeader(header=header, hint=hint, pattern=pattern))

    return banned


def iter_source_files(roots: tuple[str, ...]) -> list[str]:
    """Return the source files under `roots`, skipping vendored directories."""
    files = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
            for name in filenames:
                if SOURCE_EXTENSION_RE.match(name):
                    files.append(os.path.join(dirpath, name))
    return sorted(files)


def scan_file(path: str, banned: list[BannedHeader]) -> list[Violation]:
    """Return every direct include of a banned header in `path`."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        sys.exit(f"error: cannot read {path}: {exc.strerror}")

    violations = []
    for number, line in enumerate(text.splitlines(), start=1):
        if "#" not in line:
            continue
        for entry in banned:
            if entry.pattern.search(line):
                violations.append(Violation(path=path, line=number, banned=entry))
    return violations


def report(violations: list[Violation]) -> None:
    """Print violations grouped in source order."""
    print(_color(f"Banned includes ({len(violations)}):", "bold"))
    for violation in violations:
        location = f"{violation.path}:{violation.line}"
        print(
            f"{_color(location, 'red')}: error: banned header "
            f"'{violation.banned.header}' included"
        )
        if violation.banned.hint:
            print(f"    -> {violation.banned.hint}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        nargs="*",
        help=f"files to check (default: all sources under {'/'.join(DEFAULT_ROOTS)})",
    )
    parser.add_argument(
        "--list",
        dest="list_path",
        type=Path,
        default=DEFAULT_LIST,
        help=f"banned header list (default: {DEFAULT_LIST.name} next to this script)",
    )
    return parser.parse_args()


def main() -> int:
    """Scan the requested files and report any direct include of a banned header."""
    args = parse_args()
    banned = load_banned_headers(args.list_path)
    if not banned:
        message = f"warning: no banned headers listed in {args.list_path}"
        print(_color(message, "yellow"), file=sys.stderr)
        return 0

    files = args.files or iter_source_files(DEFAULT_ROOTS)

    violations = []
    for path in files:
        violations.extend(scan_file(path, banned))

    if violations:
        report(violations)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
