# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


def _python_for_validation_scripts() -> str:
    """Return the Python executable to use when running validation scripts.

    When running inside a PyInstaller/frozen binary, sys.executable is the
    binary itself, which does not accept script paths and validation args.
    Use system Python instead.
    """
    if getattr(sys, "frozen", False):
        env_py = os.environ.get("ROCPROFSYS_VALIDATION_PYTHON")
        if env_py:
            return env_py
        path = shutil.which("python3") or shutil.which("python")
        return path or "python3"
    return sys.executable


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


def _validate_regex(
    text: str,
    pass_regex: Optional[list[str]] = None,
    fail_regex: Optional[list[str]] = None,
    use_abort_fail_regex: bool = False,
) -> ValidationResult:
    """Validate the regex patterns in some given text.

    Args:
        text: Text to validate
        pass_regex: Optional list of regex patterns that must be found for success
        fail_regex: Optional list of regex patterns that must NOT be found
        use_abort_fail_regex: Whether to validate against ROCPROFSYS_ABORT_FAIL_REGEX (default: True)

    Returns:
        ValidationResult with is_valid=True if all patterns pass, False otherwise
    """
    # Build fail regex list
    fail_patterns: list[str] = []
    if fail_regex:
        fail_patterns.extend(fail_regex)
    if use_abort_fail_regex:
        fail_patterns.extend(ROCPROFSYS_ABORT_FAIL_REGEX)

    if not fail_patterns and not pass_regex:
        return ValidationResult(is_valid=True, message="No patterns to validate")

    # Use re.DOTALL so '.' matches newlines (like CMake regex behavior)
    flags = re.DOTALL

    # Fail patterns: one combined alternation, short-circuit on first hit
    if fail_patterns:
        fail_re = re.compile(
            "|".join(f"(?P<f{i}>{p})" for i, p in enumerate(fail_patterns)), flags
        )
        m = fail_re.search(text)
        if m is not None:
            idx = int(m.lastgroup[1:])
            return ValidationResult(
                is_valid=False,
                message=f"Fail pattern matched: {fail_patterns[idx]}",
            )

    # Pass patterns: individual re.search per pattern
    if pass_regex:
        for pattern in pass_regex:
            if re.search(pattern, text, flags) is None:
                return ValidationResult(
                    is_valid=False,
                    message=f"Pass pattern not found: {pattern}",
                )

    return ValidationResult(is_valid=True, message="All patterns validated successfully")


def validate_regex(
    test_result: TestResult,
    pass_regex: Optional[list[str]] = None,
    fail_regex: Optional[list[str]] = None,
    use_abort_fail_regex: bool = True,
) -> ValidationResult:
    return _validate_regex(
        test_result.test_output, pass_regex, fail_regex, use_abort_fail_regex
    )


def validate_file_regex(
    file_path: Path,
    pass_regex: Optional[list[str]] = None,
    fail_regex: Optional[list[str]] = None,
    use_abort_fail_regex: bool = True,
) -> ValidationResult:
    if not file_path.exists():
        return ValidationResult(False, f"File not found: {file_path}")
    with open(file_path, "r") as f:
        text = f.read()
    return _validate_regex(text, pass_regex, fail_regex, use_abort_fail_regex)


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

    python_exe = _python_for_validation_scripts()
    cmd = [python_exe, str(script_path)] + args
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
    check_counter_pairing: bool = False,
    timeout: int = 120,
) -> ValidationResult:
    """Validate a Perfetto trace file using validate-perfetto-proto.py.

    Slice validation mode is inferred by validate-perfetto-proto.py: pass ``depths``
    (-d) for positional row-by-row checks; omit ``depths`` for aggregate-by-name
    (sum counts across depths). Omit ``counts`` (-c) in aggregate mode for
    presence-only checks.

    Args:
        trace_path: Path to perfetto-trace.proto file
        tests_dir: Path to directory containing validation scripts
        categories: List of categories to filter by (-m flag)
        labels: Expected labels (-l flag)
        counts: Expected counts (-c flag)
        depths: Expected depths (-d flag); omit for aggregate-by-name validation
        label_substrings: Expected label substrings (-s flag)
        counter_names: Counter names to validate (--counter-names flag)
        key_names: Debug key names to check (--key-names flag)
        key_counts: Expected counts for debug keys (--key-counts flag)
        trace_processor_path: Path to trace_processor_shell (-t flag)
        print_output: Whether to print trace data (-p flag)
        check_counter_pairing: Verify counter tracks have paired start/end entries
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

    if check_counter_pairing:
        args.append("--check-counter-pairing")

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
    gpu_category_to_skip: Optional[list[str]] = None,
) -> ValidationResult:
    """Validate a ROCpd database file using validate-rocpd.py.

    Args:
        db_path: Path to rocpd.db file
        tests_dir: Path to directory containing validation scripts
        rules_files: List of JSON rules files to use for validation
        timeout: Validation timeout in seconds
        gpu_category_to_skip: GPU categories to skip tagged validation queries for
            (instinct, radeon, apu). Omit or pass empty to run all queries

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

    if gpu_category_to_skip:
        args.extend(["--gpu-category-to-skip"] + gpu_category_to_skip)

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

    args = ["-i", str(json_path)]

    if ci_mode:
        args.append("--ci")

    if additional_args:
        args.extend(additional_args)

    return _run_validation_script("validate-causal-json.py", args, tests_dir, timeout)


def validate_unified_memory_outputs(
    output_dir: Path,
    tests_dir: Path,
    timeout: int = 60,
) -> ValidationResult:
    """Validate unified-memory text and JSON outputs in a test output tree."""
    txt_matches = sorted(output_dir.rglob("unified_memory*.txt"))
    json_matches = sorted(output_dir.rglob("unified_memory*.json"))

    if not txt_matches:
        return ValidationResult(False, f"No unified_memory*.txt found under {output_dir}")
    if not json_matches:
        return ValidationResult(
            False, f"No unified_memory*.json found under {output_dir}"
        )

    txt_file = txt_matches[0]
    json_file = json_matches[0]

    if txt_file.parent != json_file.parent:
        return ValidationResult(
            False,
            "Unified-memory outputs landed in different directories: "
            f"{txt_file.parent} vs {json_file.parent}",
        )

    return _run_validation_script(
        "validate-unified-memory.py",
        ["--output-dir", str(txt_file.parent)],
        tests_dir,
        timeout,
    )
