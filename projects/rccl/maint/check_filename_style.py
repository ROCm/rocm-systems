#!/usr/bin/env python3
"""Enforce a filename-stem naming convention for a set of files.

Intended to run as a pre-commit `local` hook. pre-commit passes the list of
staged/changed files that matched the hook's `files:`/`exclude:` regexes as
positional arguments, so by default only new or modified files are checked --
legacy files are left alone until someone touches them.

Usage:
    check_filename_style.py --style snake  file1 file2 ...
    check_filename_style.py --style pascal file1 file2 ...

Only the filename *stem* (basename minus the final extension) is checked;
directories and extensions are ignored.
"""

import argparse
import pathlib
import re
import sys

# snake_case: lowercase words separated by underscores. Runs of underscores are
# allowed so generator conventions like `foo__funcs` / `foo__types` pass.
SNAKE = re.compile(r"^[a-z0-9]+(_+[a-z0-9]+)*$")

# PascalCase: each word starts uppercase, no separators. Digits allowed. This
# matches the GoogleTest-style test filenames (e.g. AllReduce, CollTrace).
PASCAL = re.compile(r"^([A-Z][a-z0-9]*)+$")

STYLES = {"snake": SNAKE, "pascal": PASCAL}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", required=True, choices=sorted(STYLES))
    parser.add_argument("files", nargs="*")
    args = parser.parse_args()

    pattern = STYLES[args.style]
    offenders = [
        f for f in args.files if not pattern.match(pathlib.Path(f).stem)
    ]

    for f in offenders:
        print(f"{f}: filename stem is not {args.style}_case")

    if offenders:
        print(
            f"\n{len(offenders)} file(s) violate the '{args.style}' naming "
            f"convention. Rename them, or add a path to the hook's `exclude:` "
            f"in .pre-commit-config.yaml if it is vendored/generated.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
