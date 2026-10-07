#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Run Catch2 quietly while preserving console diagnostics and JUnit results."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True, type=Path)
    parser.add_argument(
        "--timeout",
        type=float,
        help="seconds before the suite and its workloads are killed",
    )
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("a Catch2 command is required after --")
    return args


def test_result(case: ET.Element) -> str:
    if case.find("failure") is not None or case.find("error") is not None:
        return "FAIL"
    if case.find("skipped") is not None:
        return "SKIP"
    return "PASS"


def group_cases(root: ET.Element) -> dict[str, list[ET.Element]]:
    """Group <testcase> records by top-level case, in order of first appearance.

    Catch2 writes one record per SECTION named "Case/Section", and an assertion
    that fails inside a section is attached to that record only. A case whose
    assertions all live in sections has no record under its own name at all.
    Counting only slash-free names would therefore lose both.
    """
    groups: dict[str, list[ET.Element]] = {}
    for case in root.iterfind(".//testcase"):
        name = case.attrib.get("name")
        if name:
            groups.setdefault(name.split("/", 1)[0], []).append(case)
    return groups


def group_result(records: list[ET.Element]) -> str:
    """A failing section fails its case; a skip beats a pass."""
    results = {test_result(record) for record in records}
    for result in ("FAIL", "SKIP"):
        if result in results:
            return result
    return "PASS"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Durations name every case as it finishes, so the transcript of a hung
    # suite ends just before the case that hung.
    command = args.command + [
        "--durations",
        "yes",
        "--reporter",
        "console",
        "--reporter",
        f"junit::out={args.xml}",
    ]
    # A session of its own, so a timeout also kills the workloads the suite
    # spawned: they hold the pipe open and would block the read forever.
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            stdout, _ = process.communicate(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            # SIGTERM first: Catch2's fatal-signal handler names the running
            # case and flushes the console reporter, which SIGKILL would lose.
            stdout = ""
            for sig, grace in ((signal.SIGTERM, 10), (signal.SIGKILL, None)):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    pass
                try:
                    stdout, _ = process.communicate(timeout=grace)
                    break
                except subprocess.TimeoutExpired:
                    continue
            print(f"Catch2 timed out after {args.timeout:g} s; it hung after:")
            print(stdout, end="")
            return 1
    completed = subprocess.CompletedProcess(command, process.returncode, stdout)

    try:
        groups = group_cases(ET.parse(args.xml).getroot())
    except (ET.ParseError, OSError) as error:
        print(f"Could not read JUnit results: {error}", file=sys.stderr)
        print(completed.stdout, end="")
        return completed.returncode or 1

    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    for name, records in groups.items():
        result = group_result(records)
        counts[result] += 1
        print(f"{result}: {name}")
        if result == "FAIL":
            for record in records:
                for failure in record.findall("failure") + record.findall("error"):
                    if failure.text:
                        print(failure.text.strip())
    print(
        f"{counts['PASS']} passed, {counts['FAIL']} failed, "
        f"{counts['SKIP']} skipped"
    )

    # A crash can leave syntactically valid, but incomplete, JUnit without a
    # failed case. Preserve the raw transcript only for that exceptional path.
    # Catch2 returns 4 when the selected cases were all skipped; that is not a
    # crash and must not dump the banner. Any other non-zero status with no
    # failed case is one, skipped cases or not (a crash at exit after some
    # cases skipped, for instance, must still leave a log).
    if completed.returncode not in (0, 4) and counts["FAIL"] == 0:
        print("\nCatch2 terminated without a JUnit failure:")
        print(completed.stdout, end="")
    # Catch2 uses 4 for "tests were skipped / none ran". That is a skip, not a
    # failed suite: returning it would paint gfx90a red after every GPU case
    # correctly skipped for hipErrorNoDevice.
    if completed.returncode == 4 and counts["FAIL"] == 0:
        return 0
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
