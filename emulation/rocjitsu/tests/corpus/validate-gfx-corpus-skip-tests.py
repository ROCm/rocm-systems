#!/usr/bin/env python3

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check documented corpus skips against JSON and the pinned corpus.

Usage:
  python validate-gfx-corpus-skip-tests.py --corpus-directory PATH
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import subprocess
import sys

TARGET_HEADING = re.compile(r"^## (?P<target>gfx\d+)$")
CATEGORY_HEADING = re.compile(r"^### (?P<category>.+): (?P<count>\d+)$")
TEST_ENTRY = re.compile(r"^- `(?P<test>[^`]+)`(?: .*)?$")
SKIP_FILE_NAME = re.compile(r"^(?P<target>gfx\d+)_skip_tests\.json$")
COLLECTED_CASE = re.compile(
    r"^tests/test_corpus\.py::test_corpus_case\[(?P<test>[^]]+)](?:@.*)?$"
)


def load_documented_tests(
    path: Path,
) -> tuple[dict[str, list[str]], list[str]]:
    tests_by_target: dict[str, dict[str, list[str]]] = {}
    declared_counts: dict[tuple[str, str], int] = {}
    errors = []
    target: str | None = None
    category: str | None = None

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        target_match = TARGET_HEADING.fullmatch(line)
        if target_match:
            target = target_match.group("target")
            category = None
            if target in tests_by_target:
                errors.append(f"{path}:{line_number}: duplicate {target} section")
            tests_by_target.setdefault(target, {})
            continue

        if line.startswith("## "):
            target = None
            category = None
            continue

        if line.startswith("### "):
            category_match = CATEGORY_HEADING.fullmatch(line)
            if target is None or category_match is None:
                errors.append(f"{path}:{line_number}: invalid category heading")
                category = None
                continue

            category = category_match.group("category")
            key = (target, category)
            if key in declared_counts:
                errors.append(
                    f"{path}:{line_number}: duplicate {target} {category} category"
                )
            declared_counts[key] = int(category_match.group("count"))
            tests_by_target[target].setdefault(category, [])
            continue

        test_match = TEST_ENTRY.fullmatch(line)
        if test_match and target is not None:
            if category is None:
                errors.append(f"{path}:{line_number}: test is outside a category")
                continue
            tests_by_target[target][category].append(test_match.group("test"))

    for (section_target, section_category), declared_count in declared_counts.items():
        actual_count = len(tests_by_target[section_target][section_category])
        if declared_count != actual_count:
            errors.append(
                f"{path}: {section_target} {section_category} declares "
                f"{declared_count} tests but lists {actual_count}"
            )

    flattened = {
        section_target: [
            test for category_tests in categories.values() for test in category_tests
        ]
        for section_target, categories in tests_by_target.items()
    }
    return flattened, errors


def load_json_tests(
    directory: Path,
) -> tuple[dict[str, dict[str, list[str]]], list[str]]:
    tests_by_target = {}
    errors = []

    for path in sorted(directory.glob("gfx*_skip_tests.json")):
        match = SKIP_FILE_NAME.fullmatch(path.name)
        if match is None:
            errors.append(f"{path}: unexpected skip-list filename")
            continue

        target = match.group("target")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{path}: {error}")
            continue

        if not isinstance(payload, dict):
            errors.append(f"{path}: top-level value must be an object")
            continue

        target_tests = {}
        for suite, tests in payload.items():
            if not isinstance(suite, str) or not isinstance(tests, list):
                errors.append(f"{path}: each suite must contain a list")
                continue
            if not all(isinstance(test, str) for test in tests):
                errors.append(f"{path}: suite {suite!r} contains a non-string test")
                continue
            target_tests[suite] = tests
            expected_prefix = f"{suite}.{target}."
            for test in tests:
                if not test.startswith(expected_prefix):
                    errors.append(
                        f"{path}: suite {suite!r} selector {test!r} must start "
                        f"with {expected_prefix!r}"
                    )
        tests_by_target[target] = target_tests

    return tests_by_target, errors


def flatten_json_tests(
    tests_by_target: dict[str, dict[str, list[str]]],
) -> dict[str, list[str]]:
    return {
        target: [test for tests in suites.values() for test in tests]
        for target, suites in tests_by_target.items()
    }


def collect_corpus_cases(
    corpus_directory: Path,
    tests_by_target: dict[str, dict[str, list[str]]],
) -> tuple[dict[str, set[str]], list[str]]:
    collected = {}
    errors = []
    test_file = corpus_directory / "tests" / "test_corpus.py"
    if not test_file.is_file():
        return {}, [f"{corpus_directory}: tests/test_corpus.py does not exist"]

    for target, suites in sorted(tests_by_target.items()):
        command = [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_corpus.py",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "--target",
            target,
            "--suite",
            ",".join(suites),
        ]
        result = subprocess.run(
            command,
            cwd=corpus_directory,
            check=False,
            capture_output=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            text=True,
        )
        if result.returncode != 0:
            details = (result.stdout + result.stderr).strip()
            errors.append(
                f"{corpus_directory}: pytest collection failed for {target}"
                + (f":\n{details}" if details else "")
            )
            continue

        target_cases = {
            match.group("test")
            for line in result.stdout.splitlines()
            if (match := COLLECTED_CASE.fullmatch(line.strip())) is not None
        }
        if not target_cases:
            errors.append(f"{corpus_directory}: pytest collected no cases for {target}")
            continue
        collected[target] = target_cases

    return collected, errors


def validate_collected_cases(
    tests_by_target: dict[str, dict[str, list[str]]],
    collected_by_target: dict[str, set[str]],
) -> list[str]:
    errors = []
    for target, suites in sorted(tests_by_target.items()):
        if target not in collected_by_target:
            continue
        collected = collected_by_target[target]
        for suite, selectors in suites.items():
            for selector in selectors:
                if selector not in collected:
                    errors.append(
                        f"{target}: {suite!r} selector does not match a collected "
                        f"corpus case: {selector}"
                    )
    return errors


def duplicate_tests(tests: list[str]) -> list[str]:
    return sorted(test for test, count in Counter(tests).items() if count > 1)


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-directory",
        type=Path,
        default=Path.cwd(),
        help="Pinned rocjitsu-test-corpus checkout (default: current directory)",
    )
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    args = parse_arguments(arguments)
    directory = Path(__file__).resolve().parent
    document = directory / "gfx-corpus-skip-tests.md"
    documented, errors = load_documented_tests(document)
    skip_lists_by_suite, json_errors = load_json_tests(directory)
    errors.extend(json_errors)
    skip_lists = flatten_json_tests(skip_lists_by_suite)

    documented_targets = set(documented)
    json_targets = set(skip_lists)
    for target in sorted(json_targets - documented_targets):
        errors.append(f"{document}: missing {target} section")
    for target in sorted(documented_targets - json_targets):
        errors.append(f"{document}: {target} has no skip-list JSON")

    for target in sorted(documented_targets & json_targets):
        documented_tests = documented[target]
        json_tests = skip_lists[target]

        for source, duplicates in (
            ("Markdown", duplicate_tests(documented_tests)),
            ("JSON", duplicate_tests(json_tests)),
        ):
            if duplicates:
                errors.append(
                    f"{target}: duplicate tests in {source}: " + ", ".join(duplicates)
                )

        missing = sorted(set(json_tests) - set(documented_tests))
        extra = sorted(set(documented_tests) - set(json_tests))
        if missing:
            errors.append(
                f"{target}: tests missing from Markdown: " + ", ".join(missing)
            )
        if extra:
            errors.append(f"{target}: tests absent from JSON: " + ", ".join(extra))
        if len(documented_tests) != len(json_tests):
            errors.append(
                f"{target}: Markdown lists {len(documented_tests)} tests, "
                f"JSON lists {len(json_tests)}"
            )

    collected, collection_errors = collect_corpus_cases(
        args.corpus_directory.resolve(), skip_lists_by_suite
    )
    errors.extend(collection_errors)
    errors.extend(validate_collected_cases(skip_lists_by_suite, collected))

    if errors:
        print("\n".join(f"error: {error}" for error in errors), file=sys.stderr)
        return 1

    total = 0
    for target in sorted(skip_lists, key=lambda value: int(value[3:])):
        count = len(skip_lists[target])
        total += count
        print(f"{target}: {count} tests")
    print(f"All {total} skip tests match the documentation and pinned corpus cases.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
