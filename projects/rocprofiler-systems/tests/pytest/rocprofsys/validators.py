# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

"""
Output validators for rocprofiler-systems test results.

This module wraps the existing validation scripts from the tests/ directory:
- validate-perfetto-proto.py
- validate-rocpd.py
- validate-timemory-json.py
- validate-causal-json.py

We also provide the following validators:
- validate_file_exists
- validate_regex_patterns (Stricter than CMake's regex pattern validator)

This ensures consistency between pytest and CMake/CTest validation.
"""

from __future__ import annotations

import re
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
    """

    is_valid: bool
    message: str
    details: Optional[dict[str, Any]] = None
    stdout: str = ""
    stderr: str = ""

def validate_file_exists(path: Path, description: str = "File") -> ValidationResult:
    """Validate that a file exists and is non-empty.

    Args:
        path: Path to check
        description: Description for error messages

    Returns:
        ValidationResult
    """

    if not path.exists():
        return ValidationResult(
            False,
            f"{description} not found: {path}"
        )

    if path.stat().st_size == 0:
        return ValidationResult(
            False,
            f"{description} exists: {path}"
        )

    return ValidationResult(
        True,
        f"{description} exists: {path}"
    )


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
        return ValidationResult(
            False, f"Validation script not found: {script_path}"
        )

    cmd = [sys.executable, str(script_path)] + args

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return ValidationResult(
            is_valid=(result.returncode == 0),
            message=result.stdout.strip() if result.returncode == 0 else result.stderr.strip(),
            stdout=result.stdout,
            stderr=result.stderr,
        )

    except subprocess.TimeoutExpired:
        return ValidationResult(
            False, f"Validation timed out after {timeout}s"
        )
    except Exception as e:
        return ValidationResult(False, f"Validation error: {e}")


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
    metric: str,
    tests_dir: Path,
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


# ============================================================================
# Regex Pattern Validation
# ============================================================================


def validate_regex_patterns(
    content: str,
    patterns: list[str],
) -> ValidationResult:
    """Check if all regex patterns are found in the content.

    Args:
        content: String content to search (e.g., file contents, stdout, stderr)
        patterns: List of regex patterns that must all be found

    Returns:
        ValidationResult indicating whether all patterns were found

    """
    if not patterns:
        return ValidationResult(True, "No patterns to validate")

    missing_patterns = []
    for pattern in patterns:
        if not re.search(pattern, content):
            missing_patterns.append(pattern)

    if missing_patterns:
        return ValidationResult(
            is_valid=False,
            message=f"Missing {len(missing_patterns)} pattern(s): {', '.join(missing_patterns)}",
        )

    return ValidationResult(
        is_valid=True,
        message=f"All {len(patterns)} pattern(s) found",
    )
