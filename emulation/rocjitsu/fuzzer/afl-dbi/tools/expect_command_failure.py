#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys


def parse_env(assignments):
    env = os.environ.copy()
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(f"invalid environment assignment: {assignment!r}")
        key, value = assignment.split("=", 1)
        if not key:
            raise ValueError(f"invalid environment assignment: {assignment!r}")
        env[key] = value
    return env


def main():
    parser = argparse.ArgumentParser(description="Run a command that is expected to fail.")
    parser.add_argument("--env", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--expected-code", type=int, default=None)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("missing command", file=sys.stderr)
        return 2

    try:
        env = parse_env(args.env)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2

    completed = subprocess.run(command, env=env, check=False)
    if args.expected_code is not None:
        if completed.returncode != args.expected_code:
            print(
                f"expected exit code {args.expected_code}, got {completed.returncode}",
                file=sys.stderr,
            )
            return 1
        return 0

    if completed.returncode == 0:
        print("expected command to fail, but it exited 0", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
