# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Output validators for rocprofiler-systems test results.

This module wraps the existing validation scripts from the tests/ directory:
- validate-perfetto-proto.py
- validate-rocpd.py
- validate-timemory-json.py
- validate-causal-json.py

We also provide the following validators:
- validate_file_exists
"""

from __future__ import annotations
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class ValidationResult:
    """Result of a validation operation.

    Attributes:
        is_valid: Whether the validation passed
        message: Description of result or error
        details: Additional details (e.g., query results)
        stdout: Standard output from validation script
        stderr: Standard error from validation script
        command: The command that was executed
    """

    is_valid: bool
    message: str
    details: Optional[dict[str, Any]] = None
    stdout: str = ""
    stderr: str = ""
    command: str = ""


ROCPROFSYS_ABORT_FAIL_REGEX = [
    r"### ERROR ###",
    r"unknown-hash=",
    r"address of faulting memory reference",
    r"exiting with non-zero exit code",
    r"terminate called after throwing an instance",
    r"calling abort\.\. in ",
    r"Exit code: [1-9]",
]

from rocprofsys.runners import TestResult


def validate_regex(
    test_result: TestResult,
    pass_regex: Optional[list[str]] = None,
    fail_regex: Optional[list[str]] = None,
    use_abort_fail_regex: bool = True,
) -> ValidationResult:
    """Validate the regex patterns in the test result.
    Does not check for result return code.

    Args:
        test_result: TestResult object (after test execution)
        pass_regex: Optional list of regex patterns that must be found for success
        fail_regex: Optional list of regex patterns that must NOT be found
        use_abort_fail_regex: Whether to validate against ROCPROFSYS_ABORT_FAIL_REGEX (default: True)

    Returns:
        ValidationResult with is_valid=True if all patterns pass, False otherwise
    """
    # Do not check for result return code

    # Build fail regex list
    fail_patterns: list[str] = []
    if fail_regex:
        fail_patterns.extend(fail_regex)
    if use_abort_fail_regex:
        fail_patterns.extend(ROCPROFSYS_ABORT_FAIL_REGEX)

    # Build combined regex with named groups
    all_patterns: list[str] = []
    fail_indices: set[str] = set()
    pass_indices: set[str] = set()

    if fail_patterns:
        for i, pattern in enumerate(fail_patterns):
            all_patterns.append(f"(?P<f{i}>{pattern})")
            fail_indices.add(f"f{i}")

    if pass_regex:
        for i, pattern in enumerate(pass_regex):
            all_patterns.append(f"(?P<p{i}>{pattern})")
            pass_indices.add(f"p{i}")

    if not all_patterns:
        return ValidationResult(is_valid=True, message="No patterns to validate")

    # Single scan with combined regex
    combined_regex = re.compile("|".join(all_patterns))
    found_pass: set[str] = set()

    for match in combined_regex.finditer(test_result.test_output):
        matched_group = match.lastgroup

        if matched_group in fail_indices:
            original_idx = int(matched_group[1:])
            return ValidationResult(
                is_valid=False,
                message=f"Fail pattern matched: {fail_patterns[original_idx]}",
            )

        if matched_group in pass_indices:
            found_pass.add(matched_group)

    # Check if all pass patterns were found
    if pass_regex:
        missing = pass_indices - found_pass
        if missing:
            missing_idx = int(next(iter(missing))[1:])
            return ValidationResult(
                is_valid=False,
                message=f"Pass pattern not found: {pass_regex[missing_idx]}",
            )

    return ValidationResult(is_valid=True, message="All patterns validated successfully")


def validate_file_exists(path: Path, description: str = "File") -> ValidationResult:
    """Validate that a file exists and is non-empty.

    Args:
        path: Path to check
        description: Description for error messages

    Returns:
        ValidationResult
    """

    if not path.exists():
        return ValidationResult(False, f"{description} not found: {path}")

    if path.stat().st_size == 0:
        return ValidationResult(False, f"{description} is empty: {path}")

    return ValidationResult(True, f"{description} exists: {path}")


def _run_validation_script(
    script_name: str,
    args: list[str],
    tests_dir: Path,
    timeout: int = 60,
) -> ValidationResult:
    """Run an existing validation script from the tests directory.

    Args:
        script_name: Name of the script (e.g., 'validate-perfetto-proto.py')
        args: Arguments to pass to the script
        tests_dir: Path to directory containing validation scripts
        timeout: Timeout in seconds

    Returns:
        ValidationResult with script output
    """
    script_path = tests_dir / script_name

    if not script_path.exists():
        return ValidationResult(False, f"Validation script not found: {script_path}")

    cmd = [sys.executable, str(script_path)] + args
    cmd_str = " ".join(shlex.quote(arg) for arg in cmd)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        if result.returncode == 0:
            message = result.stdout.strip()
        else:
            message = (
                result.stderr.strip()
                or result.stdout.strip()
                or f"Exit code: {result.returncode}"
            )

        return ValidationResult(
            is_valid=(result.returncode == 0),
            message=message,
            stdout=result.stdout,
            stderr=result.stderr,
            command=cmd_str,
        )

    except subprocess.TimeoutExpired:
        return ValidationResult(
            False, f"Validation timed out after {timeout}s", command=cmd_str
        )
    except Exception as e:
        return ValidationResult(False, f"Validation error: {e}", command=cmd_str)


# ============================================================================
# Perfetto Validation - wraps validate-perfetto-proto.py
# ============================================================================


def validate_perfetto_trace(
    trace_path: Path,
    tests_dir: Path,
    categories: Optional[list[str]] = None,
    labels: Optional[list[str]] = None,
    counts: Optional[list[int]] = None,
    depths: Optional[list[int]] = None,
    label_substrings: Optional[list[str]] = None,
    counter_names: Optional[list[str]] = None,
    key_names: Optional[list[str]] = None,
    key_counts: Optional[list[int]] = None,
    trace_processor_path: Optional[Path] = None,
    print_output: bool = False,
    timeout: int = 120,
) -> ValidationResult:
    """Validate a Perfetto trace file using validate-perfetto-proto.py.

    Args:
        trace_path: Path to perfetto-trace.proto file
        tests_dir: Path to directory containing validation scripts
        categories: List of categories to filter by (-m flag)
        labels: Expected labels (-l flag)
        counts: Expected counts (-c flag)
        depths: Expected depths (-d flag)
        label_substrings: Expected label substrings (-s flag)
        counter_names: Counter names to validate (--counter-names flag)
        key_names: Debug key names to check (--key-names flag)
        key_counts: Expected counts for debug keys (--key-counts flag)
        trace_processor_path: Path to trace_processor_shell (-t flag)
        print_output: Whether to print trace data (-p flag)
        timeout: Validation timeout in seconds

    Returns:
        ValidationResult with validation status
    """
    if not trace_path.exists():
        return ValidationResult(False, f"Trace file not found: {trace_path}")

    # Allow override of trace_processor_path to allow perfetto validation using older GLIBC versions
    env_path = os.environ.get("ROCPROFSYS_TRACE_PROC_SHELL")
    if env_path:
        trace_processor_path = Path(env_path)

    args = ["-i", str(trace_path)]

    if categories:
        args.extend(["-m"] + categories)

    if labels:
        args.extend(["-l"] + labels)
    elif label_substrings:
        args.extend(["-s"] + label_substrings)

    if counts:
        args.extend(["-c"] + [str(c) for c in counts])

    if depths:
        args.extend(["-d"] + [str(d) for d in depths])

    if counter_names:
        args.extend(["--counter-names"] + counter_names)

    if key_names:
        args.extend(["--key-names"] + key_names)

    if key_counts:
        args.extend(["--key-counts"] + [str(k) for k in key_counts])

    if trace_processor_path:
        args.extend(["-t", str(trace_processor_path)])

    if print_output:
        args.append("-p")

    return _run_validation_script("validate-perfetto-proto.py", args, tests_dir, timeout)


# ============================================================================
# ROCpd Database Validation - wraps validate-rocpd.py
# ============================================================================


def validate_rocpd_database(
    db_path: Path,
    tests_dir: Path,
    rules_files: Optional[list[Path]] = None,
    timeout: int = 60,
) -> ValidationResult:
    """Validate a ROCpd database file using validate-rocpd.py.

    Args:
        db_path: Path to rocpd.db file
        tests_dir: Path to directory containing validation scripts
        rules_files: List of JSON rules files to use for validation
        timeout: Validation timeout in seconds

    Returns:
        ValidationResult with validation status
    """
    if not db_path.exists():
        return ValidationResult(False, f"Database not found: {db_path}")

    args = ["-db", str(db_path)]

    if rules_files:
        existing_rules = [str(r) for r in rules_files if r.exists()]
        if existing_rules:
            args.extend(["-r"] + existing_rules)

    return _run_validation_script("validate-rocpd.py", args, tests_dir, timeout)


# ============================================================================
# Timemory JSON Validation - wraps validate-timemory-json.py
# ============================================================================


def validate_timemory_json(
    json_path: Path,
    tests_dir: Path,
    metric: str,
    labels: Optional[list[str]] = None,
    counts: Optional[list[int]] = None,
    depths: Optional[list[int]] = None,
    print_output: bool = False,
    timeout: int = 60,
) -> ValidationResult:
    """Validate a timemory JSON output file using validate-timemory-json.py.

    Args:
        json_path: Path to JSON file
        metric: Metric name to validate (-m flag)
        tests_dir: Path to directory containing validation scripts
        labels: Expected labels (-l flag)
        counts: Expected counts (-c flag)
        depths: Expected depths (-d flag)
        print_output: Whether to print data (-p flag)
        timeout: Validation timeout in seconds

    Returns:
        ValidationResult with validation status
    """
    if not json_path.exists():
        return ValidationResult(False, f"JSON file not found: {json_path}")

    args = ["-i", str(json_path), "-m", metric]

    if labels:
        args.extend(["-l"] + labels)

    if counts:
        args.extend(["-c"] + [str(c) for c in counts])

    if depths:
        args.extend(["-d"] + [str(d) for d in depths])

    if print_output:
        args.append("-p")

    return _run_validation_script("validate-timemory-json.py", args, tests_dir, timeout)


# ============================================================================
# Causal JSON Validation - wraps validate-causal-json.py
# ============================================================================


def validate_causal_json(
    json_path: Path,
    tests_dir: Path,
    ci_mode: bool = False,
    additional_args: Optional[list[str]] = None,
    timeout: int = 60,
) -> ValidationResult:
    """Validate a causal profiling JSON output file using validate-causal-json.py.

    Args:
        json_path: Path to causal JSON file
        tests_dir: Path to directory containing validation scripts
        ci_mode: Whether running in CI mode (--ci flag)
        additional_args: Additional arguments to pass to the script
        timeout: Validation timeout in seconds

    Returns:
        ValidationResult with validation status
    """
    if not json_path.exists():
        return ValidationResult(False, f"JSON file not found: {json_path}")

    args = [str(json_path)]

    if ci_mode:
        args.append("--ci")

    if additional_args:
        args.extend(additional_args)

    return _run_validation_script("validate-causal-json.py", args, tests_dir, timeout)
