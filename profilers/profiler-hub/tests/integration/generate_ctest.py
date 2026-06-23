# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate CTest definitions from the profiler-hub pytest integration suite."""

from __future__ import annotations

import contextlib
import io
import sys

import pytest


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "usage: generate_ctest.py <pytest-suite-dir> <ctest-output-path>",
            file=sys.stderr,
        )
        return 2

    suite_dir, output_path = sys.argv[1:]
    args = [
        suite_dir,
        "--collect-only",
        "--ctest-mode=generate",
        "--ctest-output-path",
        output_path,
        "-q",
        "-p",
        "no:cacheprovider",
    ]

    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        retcode = pytest.main(args)

    if retcode != 0:
        print(stdout.getvalue(), end="")
        print(stderr.getvalue(), end="", file=sys.stderr)

    return retcode


if __name__ == "__main__":
    raise SystemExit(main())
