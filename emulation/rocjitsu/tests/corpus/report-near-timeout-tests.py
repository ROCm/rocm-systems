#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Report passing corpus tests that ran close to their per-test timeout."""

# The reports are ranked together rather than per report, so one run of one
# simulated target does not hide a slower test on another. They must be
# produced with `junit_duration_report=call` so that the recorded duration
# covers the same phase as `timeout_func_only=true`.

from __future__ import annotations

import argparse
from collections.abc import Iterator
import math
from pathlib import Path
import re
import sys
from typing import NamedTuple
from xml.etree import ElementTree

DEFAULT_NEAR_TIMEOUT_RATIO = 0.9

MAX_LISTED_TESTS = 10
NEAR_TIMEOUT_TESTS_FOUND_EXIT_STATUS = 3
NON_PASSING_OUTCOME_TAGS = ("error", "failure", "skipped")
WHITESPACE_RE = re.compile(r"\s+")


class NearTimeoutTest(NamedTuple):
    target: str
    case_id: str
    duration: float


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _validate_args(args)
        near_timeout_tests = collect_near_timeout_tests(
            args.reports, args.timeout, args.near_timeout_ratio
        )
    except (OSError, ValueError, ElementTree.ParseError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    report_near_timeout_tests(near_timeout_tests, args.timeout, args.near_timeout_ratio)
    if near_timeout_tests:
        return NEAR_TIMEOUT_TESTS_FOUND_EXIT_STATUS
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reports",
        nargs="+",
        type=Path,
        metavar="REPORT",
        help="pytest JUnit XML report",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        required=True,
        help="per-test timeout in seconds that the reported runs enforced",
    )
    parser.add_argument(
        "--near-timeout-ratio",
        type=float,
        default=DEFAULT_NEAR_TIMEOUT_RATIO,
        help="report a passing test once it uses this fraction of the timeout "
        f"(default: {DEFAULT_NEAR_TIMEOUT_RATIO})",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("--timeout must be a positive number of seconds")
    ratio = args.near_timeout_ratio
    if not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError("--near-timeout-ratio must be greater than 0 and at most 1")


def collect_near_timeout_tests(
    reports: list[Path], timeout: float, ratio: float
) -> list[NearTimeoutTest]:
    threshold = timeout * ratio
    near_timeout_tests = []
    for report in reports:
        if not report.is_file():
            raise ValueError(f"JUnit XML report is not a file: {report}")
        target = _sanitized(report.stem)
        for case in _passing_cases(report):
            duration = _case_duration(case, report)
            if duration < threshold:
                continue
            near_timeout_tests.append(
                NearTimeoutTest(
                    target=target, case_id=_case_id(case), duration=duration
                )
            )
    near_timeout_tests.sort(
        key=lambda test: (-test.duration, test.case_id, test.target)
    )
    return near_timeout_tests


def report_near_timeout_tests(
    near_timeout_tests: list[NearTimeoutTest], timeout: float, ratio: float
) -> None:
    if not near_timeout_tests:
        print(
            f"No passing corpus test used {ratio:.0%} or more of the "
            f"{timeout:g}s timeout."
        )
        return

    summary = (
        f"{len(near_timeout_tests)} test(s) ran close to the {timeout:g}s "
        f"timeout, using {ratio:.0%} or more of it"
    )
    print(f"Warning: {summary}.")
    for test in near_timeout_tests[:MAX_LISTED_TESTS]:
        print(f"{test.duration:g}s, {test.target}, {test.case_id}")
    unlisted = len(near_timeout_tests) - MAX_LISTED_TESTS
    if unlisted > 0:
        print(f"... and {unlisted} more not listed.")


def _passing_cases(report: Path) -> Iterator[ElementTree.Element]:
    root = ElementTree.parse(report).getroot()
    for case in root.iter("testcase"):
        if any(case.find(tag) is not None for tag in NON_PASSING_OUTCOME_TAGS):
            continue
        yield case


def _case_duration(case: ElementTree.Element, report: Path) -> float:
    raw_duration = case.get("time")
    if raw_duration is None:
        raise ValueError(f"{report}: {_case_id(case)} has no duration")
    try:
        duration = float(raw_duration)
    except ValueError as error:
        raise ValueError(
            f"{report}: {_case_id(case)} has a non-numeric duration {raw_duration!r}"
        ) from error
    if not math.isfinite(duration) or duration < 0:
        raise ValueError(
            f"{report}: {_case_id(case)} has an out-of-range duration {raw_duration!r}"
        )
    return duration


def _case_id(case: ElementTree.Element) -> str:
    classname = case.get("classname", "")
    name = case.get("name", "<unnamed>")
    identifier = f"{classname}::{name}" if classname else name
    return _sanitized(identifier)


def _sanitized(value: str) -> str:
    # Keep each reported name on one line.
    return WHITESPACE_RE.sub(" ", value).strip()


if __name__ == "__main__":
    sys.exit(main())
