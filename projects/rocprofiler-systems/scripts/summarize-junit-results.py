#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Create compact Markdown summaries from JUnit XML artifacts."""

import argparse
import glob
import json
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

BUILD_GROUP_LABELS = {
    "ubuntu-22.04": "Ubuntu 22.04",
    "ubuntu-24.04": "Ubuntu 24.04",
    "debian": "Debian",
    "rhel": "RHEL",
    "build": "Build matrix",
    "system-deps": "System deps",
}
SANITIZER_LABELS = {
    "address": "Address",
    "thread": "Thread",
    "undefined": "Undefined",
}
SANITIZER_ORDER = ("thread", "undefined", "address")


@dataclass
class FailedTest:
    workflow: str
    artifact: str
    name: str


@dataclass
class Summary:
    label: str
    runs: int = 0
    tests: int = 0
    skipped: int = 0
    failed: int = 0
    time_seconds: float = 0.0
    failed_tests: List[FailedTest] = field(default_factory=list)

    def add(self, result: "JUnitResult") -> None:
        self.runs += 1
        self.tests += result.tests
        self.skipped += result.skipped
        self.failed += result.failed
        self.time_seconds += result.time_seconds
        self.failed_tests.extend(result.failed_tests)


@dataclass
class JUnitResult:
    path: Path
    artifact: str
    group_key: str
    group_label: str
    tests: int
    skipped: int
    failed: int
    time_seconds: float
    failed_tests: List[FailedTest]


@dataclass
class SanitizerStatus:
    sanitizer: str
    status: str


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize rocprofiler-systems JUnit artifacts as Markdown."
    )
    parser.add_argument(
        "--mode",
        choices=("build", "sanitizer"),
        required=True,
        help="Summary format to generate.",
    )
    parser.add_argument(
        "--junit-glob",
        action="append",
        default=[],
        help="Glob for JUnit XML files. May be provided more than once.",
    )
    parser.add_argument(
        "--status-glob",
        action="append",
        default=[],
        help="Glob for sanitizer status JSON files. May be provided more than once.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Markdown output path.",
    )
    parser.add_argument(
        "--commit-sha",
        default=os.environ.get("GITHUB_SHA", ""),
        help="Commit SHA represented by the summarized results.",
    )
    parser.add_argument(
        "--commit-url",
        default="",
        help="URL for the commit represented by the summarized results.",
    )
    return parser.parse_args(argv)


def collect_paths(patterns: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for item in glob.glob(pattern, recursive=True):
            path = Path(item)
            if path.is_file() and path not in seen:
                paths.append(path)
                seen.add(path)
    return sorted(paths)


def int_attr(element: ET.Element, name: str) -> int:
    value = element.attrib.get(name, "0")
    try:
        return int(value)
    except ValueError:
        return 0


def float_attr(element: ET.Element, name: str) -> float:
    value = element.attrib.get(name, "0")
    try:
        return float(value)
    except ValueError:
        return 0.0


def suites_from_root(root: ET.Element) -> List[ET.Element]:
    if root.tag == "testsuite":
        return [root]
    if root.tag == "testsuites":
        return [item for item in root if item.tag == "testsuite"]
    return root.findall(".//testsuite")


def artifact_name(path: Path) -> str:
    for parent in path.parents:
        if parent.name.startswith("junit-"):
            return parent.name
    return path.parent.name


def build_group_from_artifact(artifact: str) -> str:
    if "-system-deps-" in artifact:
        return "system-deps"
    for group in ("ubuntu-22.04", "ubuntu-24.04", "debian", "rhel"):
        if artifact.startswith(f"junit-build-{group}-"):
            return group
    if artifact.startswith("junit-build-"):
        return "build"
    return "other"


def sanitizer_from_artifact(artifact: str) -> str:
    for sanitizer in SANITIZER_LABELS:
        if f"-{sanitizer}-" in artifact or artifact.endswith(f"-{sanitizer}"):
            return sanitizer
    return "unknown"


def testcase_name(testcase: ET.Element) -> str:
    classname = testcase.attrib.get("classname", "")
    name = testcase.attrib.get("name", "")
    if classname and name:
        return f"{classname}.{name}"
    return name or classname or "<unnamed test>"


def parse_junit(path: Path, mode: str) -> Optional[JUnitResult]:
    artifact = artifact_name(path)
    if mode == "sanitizer":
        group_key = sanitizer_from_artifact(artifact)
        group_label = SANITIZER_LABELS.get(group_key, group_key.title())
    else:
        group_key = build_group_from_artifact(artifact)
        group_label = BUILD_GROUP_LABELS.get(group_key, group_key.title())

    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        print(f"warning: skipping unparseable JUnit XML {path}: {exc}", file=sys.stderr)
        return None
    suites = suites_from_root(root)
    tests = sum(int_attr(suite, "tests") for suite in suites)
    skipped = sum(int_attr(suite, "skipped") for suite in suites)
    failed = sum(
        int_attr(suite, "failures") + int_attr(suite, "errors") for suite in suites
    )
    time_seconds = sum(float_attr(suite, "time") for suite in suites)
    failed_tests: List[FailedTest] = []

    for testcase in root.findall(".//testcase"):
        has_failure = testcase.find("failure") is not None
        has_error = testcase.find("error") is not None
        if has_failure or has_error:
            failed_tests.append(
                FailedTest(
                    workflow=group_label,
                    artifact=artifact,
                    name=testcase_name(testcase),
                )
            )

    return JUnitResult(
        path=path,
        artifact=artifact,
        group_key=group_key,
        group_label=group_label,
        tests=tests,
        skipped=skipped,
        failed=failed,
        time_seconds=time_seconds,
        failed_tests=failed_tests,
    )


def format_count(value: int) -> str:
    return f"{value:,}"


def format_duration(seconds: float) -> str:
    if seconds <= 0:
        return "0s"
    rounded = int(round(seconds))
    minutes, secs = divmod(rounded, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def average_duration(summary: Summary) -> str:
    if summary.runs == 0:
        return "0s"
    return format_duration(summary.time_seconds / summary.runs)


def summarize_by_group(results: Iterable[JUnitResult]) -> Dict[str, Summary]:
    summaries: Dict[str, Summary] = {}
    for result in results:
        summary = summaries.setdefault(
            result.group_key, Summary(label=result.group_label)
        )
        summary.add(result)
    return summaries


def total_summary(summaries: Iterable[Summary]) -> Summary:
    total = Summary(label="Total")
    for summary in summaries:
        total.runs += summary.runs
        total.tests += summary.tests
        total.skipped += summary.skipped
        total.failed += summary.failed
        total.time_seconds += summary.time_seconds
        total.failed_tests.extend(summary.failed_tests)
    return total


def summary_row(summary: Summary) -> str:
    return (
        f"| {summary.label} | {summary.runs} | {average_duration(summary)} | "
        f"{format_count(summary.tests)} | {format_count(summary.skipped)} | "
        f"{format_count(summary.failed)} |"
    )


def build_failed_tests_section(failed_tests: Sequence[FailedTest]) -> List[str]:
    lines = ["### Failed Tests", ""]
    if not failed_tests:
        lines.append("No failed tests reported.")
        return lines

    lines.extend(
        [
            "| Workflow | Artifact | Failed Test |",
            "|---|---|---|",
        ]
    )
    for failed_test in failed_tests[:50]:
        lines.append(
            f"| {failed_test.workflow} | `{failed_test.artifact}` | "
            f"`{failed_test.name}` |"
        )
    if len(failed_tests) > 50:
        lines.append(f"| ... | ... | {len(failed_tests) - 50} more failed tests |")
    return lines


def heading(title: str, commit_sha: str, commit_url: str = "") -> str:
    if not commit_sha:
        return f"## {title}"

    short_sha = commit_sha[:7]
    commit_ref = f"`{short_sha}`"
    if commit_url:
        commit_ref = f"[`{short_sha}`]({commit_url})"
    return f"## {title} (for commit {commit_ref})"


def render_build_summary(
    results: Sequence[JUnitResult], commit_sha: str = "", commit_url: str = ""
) -> str:
    summaries = summarize_by_group(results)
    ordered_keys = [
        "ubuntu-22.04",
        "ubuntu-24.04",
        "debian",
        "rhel",
        "build",
        "system-deps",
    ]
    ordered_summaries = [summaries[key] for key in ordered_keys if key in summaries] + [
        summary for key, summary in sorted(summaries.items()) if key not in ordered_keys
    ]
    total = total_summary(ordered_summaries)

    lines = [
        heading("Test Results", commit_sha, commit_url),
        "",
        f"Workflow jobs covered: **{total.runs}**",
        "",
        "| Workflow | Runs | Avg Time | Tests | Skipped | Failed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(summary_row(summary) for summary in ordered_summaries)
    lines.append(summary_row(total))
    lines.append("")
    lines.extend(build_failed_tests_section(total.failed_tests))
    return "\n".join(lines) + "\n"


def normalize_status(status: str) -> str:
    normalized = status.strip().lower()
    if normalized == "success":
        return "Passed"
    if normalized == "failure":
        return "Failed"
    if normalized == "cancelled":
        return "Cancelled"
    if normalized == "skipped":
        return "Skipped"
    return status.strip().title() or "Unknown"


def read_statuses(paths: Iterable[Path]) -> Dict[str, SanitizerStatus]:
    statuses: Dict[str, SanitizerStatus] = {}
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            print(
                f"warning: skipping unreadable status file {path}: {exc}", file=sys.stderr
            )
            continue
        sanitizer = str(data.get("sanitizer", "")).strip().lower()
        status = str(data.get("status", "")).strip()
        if sanitizer:
            statuses[sanitizer] = SanitizerStatus(sanitizer=sanitizer, status=status)
    return statuses


def render_sanitizer_summary(
    results: Sequence[JUnitResult], statuses: Dict[str, SanitizerStatus]
) -> str:
    summaries = summarize_by_group(results)
    columns = [SANITIZER_LABELS[key] for key in SANITIZER_ORDER]
    values: List[str] = []

    for key in SANITIZER_ORDER:
        status = statuses.get(key)
        if status is not None:
            values.append(normalize_status(status.status))
            continue

        summary = summaries.get(key)
        if summary is None:
            values.append("Missing")
        elif summary.failed:
            values.append("Failed")
        else:
            values.append("Passed")

    return "\n".join(
        [
            "## Sanitizers",
            "",
            f"| {' | '.join(columns)} |",
            f"| {' | '.join(['---'] * len(columns))} |",
            f"| {' | '.join(values)} |",
            "",
        ]
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    junit_paths = collect_paths(args.junit_glob)
    status_paths = collect_paths(args.status_glob)
    results = [
        result
        for result in (parse_junit(path, args.mode) for path in junit_paths)
        if result is not None
    ]

    if args.mode == "build":
        markdown = render_build_summary(results, args.commit_sha, args.commit_url)
    else:
        markdown = render_sanitizer_summary(results, read_statuses(status_paths))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
