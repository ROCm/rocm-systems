#!/usr/bin/env python3
"""Runs a target smoke until it emits an exact independent success marker."""

from __future__ import annotations

import argparse
import os
import selectors
import signal
import subprocess
import sys
import time


def _stop_group(process: subprocess.Popen[bytes], sig: signal.Signals) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def _run(command: list[str], marker: bytes, timeout: float) -> int:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout
    tail = b""
    accepted = False
    timed_out = False
    while process.poll() is None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        events = selector.select(min(remaining, 0.25))
        if not events:
            continue
        chunk = os.read(process.stdout.fileno(), 65536)
        if not chunk:
            continue
        sys.stdout.buffer.write(chunk)
        sys.stdout.buffer.flush()
        accepted = marker in tail + chunk
        tail = (tail + chunk)[-(len(marker) - 1) :] if len(marker) > 1 else b""
        if accepted:
            break

    if accepted or timed_out:
        _stop_group(process, signal.SIGTERM)
    try:
        remainder, _ = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        _stop_group(process, signal.SIGKILL)
        remainder, _ = process.communicate()
    selector.close()
    if remainder:
        sys.stdout.buffer.write(remainder)
        sys.stdout.buffer.flush()
        accepted = accepted or marker in tail + remainder
    if accepted and not timed_out:
        return 0
    if timed_out:
        print(f"smoke timeout before success marker after {timeout:g}s", file=sys.stderr)
        return 124
    print("smoke process exited without the required success marker", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--success-marker", required=True)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if not args.success_marker:
        parser.error("--success-marker must not be empty")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return _run(args.command, args.success_marker.encode(), args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
