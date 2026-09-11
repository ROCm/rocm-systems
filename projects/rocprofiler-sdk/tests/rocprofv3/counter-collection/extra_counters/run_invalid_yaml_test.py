#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Validate fatal extra-counter failure status, diagnostics, and timeout.

CTest cannot reliably combine expected signal termination with output matching,
so this wrapper converts the child result into a normal test pass or failure.
"""

import argparse
import re
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--expect",
        action="append",
        required=True,
        help="Regular expression that must appear in combined process output.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15,
        help="Maximum child-process runtime in seconds.",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command must follow '--'")

    return args


def to_text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def main():
    args = parse_args()

    try:
        result = subprocess.run(
            args.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        sys.stdout.write(to_text(error.stdout))
        print(
            f"Command did not terminate within {args.timeout:g} seconds",
            file=sys.stderr,
        )
        return 1

    sys.stdout.write(result.stdout)

    if result.returncode == 0:
        print("Expected command to terminate unsuccessfully", file=sys.stderr)
        return 1

    missing_patterns = [
        pattern
        for pattern in args.expect
        if re.search(pattern, result.stdout, re.DOTALL) is None
    ]
    if missing_patterns:
        print(
            f"Missing expected output: {', '.join(missing_patterns)}",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
