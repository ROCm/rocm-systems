#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Check or fix the mechanically enforceable rocjitsu C++ style rules."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from stylist_rules import Violation, check_style_rules, normalize_direct_rules

LOGGER = logging.getLogger("stylist")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STYLE_FILE = PROJECT_ROOT / ".clang-format"
BASELINE_FILE = Path(__file__).with_name("stylist_baseline.json")
SOURCE_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".h", ".hh", ".hpp"})
MAX_JOB_COUNT = 256
TEMPLATE_DECLARATION_START_PATTERN = re.compile(r"^\s*template\s*<")
TEMPLATE_CLASS_PARAMETER_PATTERN = re.compile(
    r"(?P<boundary>^|[<,])(?P<space>\s*)class(?=\s+[A-Za-z_]\w*)"
)
HEADER_GUARD_OPEN_PATTERN = re.compile(
    r"^#ifndef (?P<guard>[A-Z][A-Z0-9_]+_H_)\n#define (?P=guard)\n",
    re.MULTILINE,
)
IGNORED_DIRECTORIES = frozenset(
    {
        ".cache",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".venv",
        "__pycache__",
        "_build",
        "build",
    }
)


@dataclass(frozen=True)
class StyleResult:
    """Describe the formatter result for one source file."""

    path: Path
    changed: bool = False
    error: str | None = None
    violations: tuple[Violation, ...] = ()


def _parse_arguments(arguments: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check or fix rocjitsu C/C++ formatting in parallel.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="report files that need changes without modifying them",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(8, os.cpu_count() or 1),
        metavar="COUNT",
        help="maximum parallel formatter processes (default: %(default)s)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="report files changed in fix mode and print a summary",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also require complete Doxygen on public header APIs",
    )
    parser.add_argument(
        "--format-only",
        action="store_true",
        help="skip semantic diagnostics and apply only deterministic rewrites",
    )
    parser.add_argument(
        "--all-violations",
        action="store_true",
        help="report baseline violations in addition to newly introduced ones",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="replace the semantic baseline with current strict-rule findings",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        metavar="PATH",
        help="files or directories to process (default: all of rocjitsu)",
    )
    parsed_arguments = parser.parse_args(arguments)
    if parsed_arguments.jobs < 1:
        parser.error("--jobs must be greater than zero")
    if parsed_arguments.jobs > MAX_JOB_COUNT:
        parser.error(f"--jobs must not exceed {MAX_JOB_COUNT}")
    if parsed_arguments.strict and parsed_arguments.format_only:
        parser.error("--strict and --format-only cannot be used together")
    if parsed_arguments.all_violations and parsed_arguments.format_only:
        parser.error("--all-violations and --format-only cannot be used together")
    if parsed_arguments.update_baseline and (
        parsed_arguments.check
        or parsed_arguments.format_only
        or parsed_arguments.all_violations
    ):
        parser.error(
            "--update-baseline cannot be combined with --check, --format-only, "
            "or --all-violations"
        )
    return parsed_arguments


def _is_ignored_directory(name: str) -> bool:
    return name in IGNORED_DIRECTORIES or name.startswith("build-")


def _collect_files(raw_paths: list[str]) -> tuple[list[Path], list[str]]:
    input_paths = [Path(path) for path in raw_paths] if raw_paths else [PROJECT_ROOT]
    source_files: set[Path] = set()
    errors: list[str] = []

    for input_path in input_paths:
        absolute_path = input_path.resolve()
        if not absolute_path.exists():
            errors.append(f"{input_path}: no such file or directory")
            continue
        if absolute_path.is_file():
            if absolute_path.suffix.lower() in SOURCE_SUFFIXES:
                source_files.add(absolute_path)
            continue
        if not absolute_path.is_dir():
            errors.append(f"{input_path}: unsupported file type")
            continue

        for directory_path, directory_names, file_names in os.walk(absolute_path):
            directory_names[:] = [
                name for name in directory_names if not _is_ignored_directory(name)
            ]
            current_directory = Path(directory_path)
            source_files.update(
                current_directory / name
                for name in file_names
                if Path(name).suffix.lower() in SOURCE_SUFFIXES
            )

    return sorted(source_files), errors


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _violation_key(path: Path, violation: Violation) -> str:
    return f"{_display_path(path)}\t{violation.rule}\t{violation.message}"


def _load_baseline() -> dict[str, int]:
    try:
        baseline = json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"unable to read semantic baseline: {error}") from error
    if not isinstance(baseline, dict) or not all(
        isinstance(key, str) and isinstance(count, int) and count >= 0
        for key, count in baseline.items()
    ):
        raise RuntimeError("semantic baseline must map violation keys to counts")
    return baseline


def _new_violations(
    results: list[StyleResult], baseline: dict[str, int]
) -> list[tuple[Path, Violation]]:
    remaining_counts = baseline.copy()
    new_violations: list[tuple[Path, Violation]] = []
    for result in results:
        for violation in result.violations:
            key = _violation_key(result.path, violation)
            if remaining_counts.get(key, 0) > 0:
                remaining_counts[key] -= 1
            else:
                new_violations.append((result.path, violation))
    return new_violations


def _write_baseline(results: list[StyleResult]) -> None:
    baseline: dict[str, int] = {}
    for result in results:
        for violation in result.violations:
            key = _violation_key(result.path, violation)
            baseline[key] = baseline.get(key, 0) + 1
    BASELINE_FILE.write_text(
        json.dumps(dict(sorted(baseline.items())), indent=2) + "\n",
        encoding="utf-8",
    )


def _normalize_template_parameters(contents: str) -> str:
    normalized_lines: list[str] = []
    for line in contents.splitlines(keepends=True):
        declaration_start = TEMPLATE_DECLARATION_START_PATTERN.match(line)
        if declaration_start is None:
            normalized_lines.append(line)
            continue

        parameter_start = declaration_start.end() - 1
        # Find the closing bracket for the outer template declaration while
        # preserving nested template defaults and quoted non-type parameters.
        depth = 0
        quote: str | None = None
        escaped = False
        parameter_end: int | None = None
        for position in range(parameter_start, len(line)):
            character = line[position]
            if quote is not None:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == quote:
                    quote = None
                continue
            if character in {'"', "'"}:
                quote = character
            elif line.startswith("//", position):
                break
            elif character == "<":
                depth += 1
            elif character == ">":
                depth -= 1
                if depth == 0:
                    parameter_end = position
                    break

        if parameter_end is None:
            normalized_lines.append(line)
            continue
        parameters = line[parameter_start + 1 : parameter_end]
        normalized_parameters = TEMPLATE_CLASS_PARAMETER_PATTERN.sub(
            r"\g<boundary>\g<space>typename", parameters
        )
        normalized_lines.append(
            line[: parameter_start + 1] + normalized_parameters + line[parameter_end:]
        )
    return "".join(normalized_lines)


def _normalize_header_guard(contents: str) -> str:
    opening_match = HEADER_GUARD_OPEN_PATTERN.search(contents)
    if opening_match is None:
        return contents
    guard = re.escape(opening_match.group("guard"))
    closing_pattern = re.compile(rf"\n#endif(?:\s*//\s*{guard})?\s*\Z")
    if closing_pattern.search(contents) is None:
        return contents
    without_opening = HEADER_GUARD_OPEN_PATTERN.sub("#pragma once\n", contents, count=1)
    return closing_pattern.sub("\n", without_opening, count=1)


def _normalize_contents(path: Path, contents: str) -> str:
    normalized_contents = _normalize_header_guard(contents)
    normalized_contents = _normalize_template_parameters(normalized_contents)
    return normalize_direct_rules(path, normalized_contents)


def _formatted_contents(
    clang_format: str, path: Path, contents: bytes
) -> tuple[bytes | None, str | None]:
    completed_process = subprocess.run(
        [
            clang_format,
            f"--assume-filename={path}",
            f"--style=file:{STYLE_FILE}",
        ],
        input=contents,
        capture_output=True,
        check=False,
    )
    if completed_process.returncode == 0:
        try:
            completed_process.stdout.decode("utf-8")
        except UnicodeDecodeError as error:
            return None, f"clang-format produced invalid UTF-8: {error}"
        return completed_process.stdout, None
    detail = completed_process.stderr.decode("utf-8", errors="replace").strip()
    return None, detail or "clang-format failed"


def _prepare_file(
    clang_format: str, path: Path, strict: bool, format_only: bool
) -> tuple[bytes, bytes, tuple[Violation, ...]] | StyleResult:
    try:
        original_bytes = path.read_bytes()
        original_contents = original_bytes.decode("utf-8")
        normalized_bytes = _normalize_contents(path, original_contents).encode("utf-8")
        formatted_bytes, error = _formatted_contents(
            clang_format, path, normalized_bytes
        )
    except (OSError, UnicodeDecodeError) as error:
        return StyleResult(path=path, error=str(error))
    if error is not None:
        return StyleResult(path=path, error=error)
    assert formatted_bytes is not None
    violations = (
        ()
        if format_only
        else tuple(check_style_rules(path, original_contents, strict=strict))
    )
    return original_bytes, formatted_bytes, violations


def _check_file(
    clang_format: str, path: Path, strict: bool, format_only: bool
) -> StyleResult:
    prepared_file = _prepare_file(clang_format, path, strict, format_only)
    if isinstance(prepared_file, StyleResult):
        return prepared_file
    original_bytes, formatted_bytes, violations = prepared_file
    return StyleResult(
        path=path,
        changed=formatted_bytes != original_bytes,
        violations=violations,
    )


def _fix_file(
    clang_format: str, path: Path, strict: bool, format_only: bool
) -> StyleResult:
    prepared_file = _prepare_file(clang_format, path, strict, format_only)
    if isinstance(prepared_file, StyleResult):
        return prepared_file
    original_bytes, formatted_bytes, violations = prepared_file
    try:
        if formatted_bytes != original_bytes:
            path.write_bytes(formatted_bytes)
    except OSError as error:
        return StyleResult(path=path, error=str(error))
    return StyleResult(
        path=path,
        changed=formatted_bytes != original_bytes,
        violations=violations,
    )


def _process_files(
    clang_format: str,
    source_files: list[Path],
    check_only: bool,
    strict: bool,
    format_only: bool,
    job_count: int,
) -> list[StyleResult]:
    operation = _check_file if check_only else _fix_file
    with concurrent.futures.ThreadPoolExecutor(max_workers=job_count) as executor:
        futures = [
            executor.submit(operation, clang_format, source_file, strict, format_only)
            for source_file in source_files
        ]
        return [future.result() for future in futures]


def main(arguments: list[str] | None = None) -> int:
    """Check or fix rocjitsu C++ style and return a conventional exit status."""
    parsed_arguments = _parse_arguments(
        arguments if arguments is not None else sys.argv[1:]
    )
    logging.basicConfig(
        format="stylist: %(levelname)s: %(message)s",
        level=logging.INFO if parsed_arguments.verbose else logging.WARNING,
    )

    clang_format = shutil.which("clang-format")
    if clang_format is None:
        LOGGER.error("clang-format was not found in PATH")
        return 2

    source_files, path_errors = _collect_files(parsed_arguments.paths)
    for path_error in path_errors:
        LOGGER.error(path_error)
    if path_errors:
        return 2

    results = _process_files(
        clang_format,
        source_files,
        parsed_arguments.check or parsed_arguments.update_baseline,
        parsed_arguments.strict or parsed_arguments.update_baseline,
        parsed_arguments.format_only,
        parsed_arguments.jobs,
    )

    failed_results = [result for result in results if result.error is not None]
    for failed_result in failed_results:
        LOGGER.error("%s: %s", _display_path(failed_result.path), failed_result.error)
    if failed_results:
        return 2

    if parsed_arguments.update_baseline:
        try:
            _write_baseline(results)
        except OSError as error:
            LOGGER.error("unable to write semantic baseline: %s", error)
            return 2
        violation_count = sum(len(result.violations) for result in results)
        LOGGER.warning("recorded %d strict semantic violation(s)", violation_count)
        return 0

    try:
        baseline = {} if parsed_arguments.all_violations else _load_baseline()
    except RuntimeError as error:
        LOGGER.error("%s", error)
        return 2
    new_violations = _new_violations(results, baseline)
    for path, violation in new_violations:
        LOGGER.warning(
            "%s:%d: [%s] %s",
            _display_path(path),
            violation.line,
            violation.rule,
            violation.message,
        )

    changed_results = [result for result in results if result.changed]
    if parsed_arguments.check:
        for changed_result in changed_results:
            LOGGER.warning("%s needs formatting", _display_path(changed_result.path))
        if changed_results:
            LOGGER.warning("%d file(s) need formatting", len(changed_results))
            return 1
    elif parsed_arguments.verbose:
        for changed_result in changed_results:
            LOGGER.info("formatted %s", _display_path(changed_result.path))

    LOGGER.info(
        "%s %d file(s); %d changed",
        "checked" if parsed_arguments.check else "processed",
        len(results),
        len(changed_results),
    )
    if new_violations:
        LOGGER.warning("%d new semantic style violation(s)", len(new_violations))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
