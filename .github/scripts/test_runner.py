#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Build the hip-tests ctest command and exec it.

--parallel is the ctest flag TheRock test_runner.py appends. A count of 0
omits the flag so ctest stays serial. This is not a rocJITsu option.
"""

import argparse
import os


def ctest_command(
    parallel: int,
    test_dir: str,
    label: str,
    timeout: int,
    exclude_regex: str,
) -> list[str]:
    cmd = [
        "ctest",
        "--test-dir",
        test_dir,
        "-L",
        label,
        "--exclude-regex",
        exclude_regex,
        "--output-on-failure",
        "--timeout",
        str(timeout),
        "-V",
    ]
    if parallel > 0:
        cmd.extend(["--parallel", str(parallel)])
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--exclude-regex", required=True)
    args = parser.parse_args()
    cmd = ctest_command(
        args.parallel,
        args.test_dir,
        args.label,
        args.timeout,
        args.exclude_regex,
    )
    os.execvp(cmd[0], cmd)


if __name__ == "__main__":
    raise SystemExit(main())
