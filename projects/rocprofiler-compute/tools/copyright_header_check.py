#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Pre-commit hook to enforce the two-line copyright header on committed files."""

import subprocess
import sys

EXPECTED_LINES = (
    "Copyright (c) Advanced Micro Devices, Inc.",
    "SPDX-License-Identifier:  MIT",
)

COMMENT_PREFIXES = {
    ".py": "#",
    ".hip": "//",
    ".cpp": "//",
}

EXCLUDED_DIRS = (
    "src/vendored/",
    "docs/archive/",
)

EXCLUDED_FILES = ("__init__.py",)


def _check_header(filepath, prefix):
    expected = [
        f"{prefix} {EXPECTED_LINES[0]}",
        f"{prefix} {EXPECTED_LINES[1]}",
    ]
    try:
        with open(filepath, encoding="utf-8", errors="replace") as f:
            lines = [f.readline().rstrip("\n\r") for _ in range(3)]
    except OSError:
        return "  could not read file"

    # Allow shebang on line 1; header shifts to lines 2-3.
    if lines[0].startswith("#!"):
        header = lines[1:3]
        offset = 2
    else:
        header = lines[0:2]
        offset = 1

    errors = []
    for i, (got, want) in enumerate(zip(header, expected), start=offset):
        if got != want:
            errors.append(f"  line {i}: expected: {want!r}\n         got:      {got!r}")
    return "\n".join(errors) if errors else None


def _get_staged_files():
    prefix = subprocess.run(
        ["git", "rev-parse", "--show-prefix"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    result = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
        capture_output=True,
        text=True,
        check=True,
    )
    paths = []
    for line in result.stdout.splitlines():
        if prefix and line.startswith(prefix):
            paths.append(line[len(prefix) :])
        elif not prefix:
            paths.append(line)
    return paths


def main():
    failures = []

    for filepath in _get_staged_files():
        if any(filepath.startswith(d) for d in EXCLUDED_DIRS):
            continue
        basename = filepath.rsplit("/", 1)[-1]
        if basename in EXCLUDED_FILES:
            continue
        ext = ""
        for e in COMMENT_PREFIXES:
            if filepath.endswith(e):
                ext = e
                break
        if not ext:
            continue

        error = _check_header(filepath, COMMENT_PREFIXES[ext])
        if error:
            failures.append((filepath, error))

    if failures:
        print("Copyright header check failed:\n")
        for filepath, error in failures:
            print(f"{filepath}:")
            print(error)
            print()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
