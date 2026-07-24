#!/usr/bin/env python3
"""Run the target-native gfx942 hip-moi suites through the RocJITsu simulator."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys

import consan_validation

TARGET = "gfx942"
BUILD_DIR_NAME = "hip-moi-build-gfx942-tests"
EXPECTED_TESTS = 14
_PASSED_TESTS = re.compile(r"^\[  PASSED  \] ([0-9]+) tests?\.$", re.MULTILINE)


@dataclass(frozen=True)
class Suite:
    id: str
    workload_id: str
    expected_tests: int


@dataclass(frozen=True)
class SuiteResult:
    suite: Suite
    passed_tests: int
    error: str | None

    @property
    def accepted(self) -> bool:
        return self.error is None and self.passed_tests == self.suite.expected_tests


SUITES = (
    Suite("jakub-matmul", "jakub-attention", 2),
    Suite("mfma-attention", "wmma-attention", 2),
    Suite("d128-block", "d128-block", 2),
    Suite("d128-pressure", "d128-pressure", 4),
    Suite("streamk-arrival", "streamk-arrival", 2),
    Suite("tree-atomic-or", "tree-atomic-or", 2),
)
SUITE_BY_ID = {suite.id: suite for suite in SUITES}


def _executable_suffix(suite: Suite) -> Path:
    relative_path = Path(
        consan_validation.resolved_workload_relative_path(
            TARGET,
            suite.workload_id,
        )
    )
    try:
        return relative_path.relative_to(BUILD_DIR_NAME)
    except ValueError as error:
        raise consan_validation.ValidationError(
            f"{suite.workload_id} must resolve beneath {BUILD_DIR_NAME}: "
            f"{relative_path}"
        ) from error


def _suite_command(
    suite: Suite,
    rocjitsu: Path,
    config: Path,
    hip_moi_build: Path,
) -> list[str]:
    return [
        str(rocjitsu),
        "--config",
        str(config),
        "--",
        str(hip_moi_build / _executable_suffix(suite)),
        "--gtest_brief=1",
    ]


def _print_child_output(completed: subprocess.CompletedProcess[str]) -> None:
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n")
    if completed.stderr:
        print(
            completed.stderr,
            end="" if completed.stderr.endswith("\n") else "\n",
            file=sys.stderr,
        )


def _run_suite(
    suite: Suite,
    rocjitsu: Path,
    config: Path,
    hip_moi_build: Path,
    timeout: float,
) -> SuiteResult:
    executable = hip_moi_build / _executable_suffix(suite)
    print(
        f"== {suite.id}: {executable.name} "
        f"({suite.expected_tests} expected tests) ==",
        flush=True,
    )
    if not executable.is_file():
        return SuiteResult(suite, 0, f"missing executable: {executable}")

    command = _suite_command(suite, rocjitsu, config, hip_moi_build)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return SuiteResult(suite, 0, f"timed out after {error.timeout} seconds")
    except OSError as error:
        return SuiteResult(suite, 0, f"failed to launch simulator: {error}")

    _print_child_output(completed)
    if completed.returncode != 0:
        return SuiteResult(
            suite,
            0,
            f"simulator exited with status {completed.returncode}",
        )
    matches = _PASSED_TESTS.findall(completed.stdout)
    if len(matches) != 1:
        return SuiteResult(suite, 0, "gtest did not report one passed-test count")
    passed_tests = int(matches[0])
    if passed_tests != suite.expected_tests:
        return SuiteResult(
            suite,
            passed_tests,
            f"expected {suite.expected_tests} tests, observed {passed_tests}",
        )
    return SuiteResult(suite, passed_tests, None)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rocjitsu", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "configs"
        / "gfx942_cdna3_kmd.json",
    )
    parser.add_argument("--hip-moi-build", type=Path, required=True)
    parser.add_argument("--suite", choices=tuple(SUITE_BY_ID))
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    rocjitsu = args.rocjitsu.expanduser().resolve()
    config = args.config.expanduser().resolve()
    hip_moi_build = args.hip_moi_build.expanduser().resolve()
    try:
        if (
            len(SUITE_BY_ID) != len(SUITES)
            or sum(suite.expected_tests for suite in SUITES) != EXPECTED_TESTS
        ):
            raise consan_validation.ValidationError(
                f"gfx942 suite registry must contain unique IDs for "
                f"{EXPECTED_TESTS} tests"
            )
        if not rocjitsu.is_file():
            raise consan_validation.ValidationError(
                f"rocjitsu executable is missing: {rocjitsu}"
            )
        if not config.is_file():
            raise consan_validation.ValidationError(
                f"simulator config is missing: {config}"
            )
        if not hip_moi_build.is_dir() or hip_moi_build.name != BUILD_DIR_NAME:
            raise consan_validation.ValidationError(
                f"--hip-moi-build must name an existing {BUILD_DIR_NAME} directory: "
                f"{hip_moi_build}"
            )
        suites = SUITES if args.suite is None else (SUITE_BY_ID[args.suite],)
        results = tuple(
            _run_suite(
                suite,
                rocjitsu,
                config,
                hip_moi_build,
                args.timeout,
            )
            for suite in suites
        )
    except (OSError, consan_validation.ValidationError) as error:
        print(f"gfx942 hip-moi simulator error: {error}", file=sys.stderr)
        return 2

    print("== gfx942 hip-moi simulator summary ==")
    for result in results:
        state = "PASS" if result.accepted else "FAIL"
        detail = (
            f"{result.passed_tests}/{result.suite.expected_tests} tests"
            if result.error is None
            else result.error
        )
        print(f"{state:4} {result.suite.id}: {detail}")
    passed_tests = sum(result.passed_tests for result in results)
    expected_tests = sum(result.suite.expected_tests for result in results)
    print(f"total: {passed_tests}/{expected_tests} tests")
    return 0 if all(result.accepted for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
