#!/usr/bin/env python3
"""Run Catch2 quietly while preserving console diagnostics and JUnit results."""

from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
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


def main() -> int:
    args = parse_args()
    command = args.command + [
        "--reporter",
        "console",
        "--reporter",
        f"junit::out={args.xml}",
    ]
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    try:
        cases = [
            case
            for case in ET.parse(args.xml).iterfind(".//testcase")
            if "/" not in case.attrib["name"]
        ]
    except (ET.ParseError, OSError, KeyError) as error:
        print(f"Could not read JUnit results: {error}", file=sys.stderr)
        print(completed.stdout, end="")
        return completed.returncode or 1

    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    for case in cases:
        result = test_result(case)
        counts[result] += 1
        print(f"{result}: {case.attrib['name']}")
        if result == "FAIL":
            for failure in case.findall("failure") + case.findall("error"):
                if failure.text:
                    print(failure.text.strip())
    print(
        f"{counts['PASS']} passed, {counts['FAIL']} failed, "
        f"{counts['SKIP']} skipped"
    )

    # A crash can leave syntactically valid, but incomplete, JUnit without a
    # failed case. Preserve the raw transcript only for that exceptional path.
    # Catch2 returns 4 when the selected cases were all skipped; that is not a
    # crash and must not dump the banner.
    if completed.returncode and counts["FAIL"] == 0 and counts["SKIP"] == 0:
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
