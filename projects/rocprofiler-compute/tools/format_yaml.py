#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Write the metric equations of the metric library YAML files in one form, which
CONTRIBUTING.md describes."""

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

# Make src/ importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from utils.metrics.expression import normalize_equation

# =============================================================================
# Configuration - Edit these to accommodate future YAML updates
# =============================================================================

# Path pattern of the metric library files, whose equations are normalized (a
# substring of the path)
LIBRARY_FILE_PATTERN = "metric_library/gfx"
# Path pattern of build copies, which are skipped
BUILD_DIR_PATTERN = "/build/"

# YAML keys of the metric library that contain metric equations
EQUATION_KEYS = {"formula", "peak", "peak_empirical"}


def extract_metric_equations(
    data: Dict[str, Any], path: str = ""
) -> List[Tuple[str, str]]:
    """Return (key path, equation) of each equation field in YAML data."""
    equations = []

    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            if key in EQUATION_KEYS and isinstance(value, str):
                equations.append((current_path, value))
            elif isinstance(value, dict):
                equations.extend(extract_metric_equations(value, current_path))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        equations.extend(
                            extract_metric_equations(item, f"{current_path}[{i}]")
                        )
    return equations


def _apply_fixes(content: str, fixes: List[Tuple[str, str]]) -> str:
    """Replace each original equation with its normalized form in the YAML text."""
    result = content
    for original, normalized in fixes:
        pattern = re.escape(original).replace(r"\ ", r"\s+")
        result = re.sub(pattern, normalized, result, count=1)
    return result


def check_file(
    path: Path, fix: bool = False, show_diff: bool = False
) -> Tuple[bool, List[str]]:
    """Check a metric library file for formatting issues. Returns (is_valid,
    issues); other files have none."""
    if LIBRARY_FILE_PATTERN not in str(path) or BUILD_DIR_PATTERN in str(path):
        return True, []

    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        data = yaml.safe_load(content)
    except (yaml.YAMLError, OSError, UnicodeDecodeError) as e:
        return False, [f"Failed to parse YAML: {e}"]

    if data is None:
        return True, []

    parse_errors = []
    format_issues = []
    fixes = []

    for eq_path, value in extract_metric_equations(data):
        try:
            normalized = normalize_equation(value)
        except ValueError as e:
            parse_errors.append(f"  {eq_path}: cannot parse equation: {e}\n    {value}")
            continue
        if normalized != value:
            fixes.append((value, normalized))
            format_issues.append(f"  {eq_path}: equation needs normalization")
            if show_diff:
                format_issues.append(f"    - {value}")
                format_issues.append(f"    + {normalized}")

    if fix and fixes:
        with open(path, "w", encoding="utf-8") as f:
            f.write(_apply_fixes(content, fixes))
        format_issues = []

    issues = parse_errors + format_issues
    return not issues, issues


def main() -> int:
    parser = argparse.ArgumentParser(
        description="YAML metric equation and description formatter for "
        "rocprofiler-compute"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="Check without modifying")
    mode.add_argument("--fix", action="store_true", help="Auto-fix in place")
    mode.add_argument("--diff", action="store_true", help="Show proposed changes")
    parser.add_argument("files", nargs="*", type=Path, help="Files to check")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()
    files = args.files or [Path(line.strip()) for line in sys.stdin if line.strip()]

    if not files:
        print("No files to check")
        return 0

    all_valid = True
    total_issues = 0

    for file_path in files:
        if not file_path.exists() or file_path.suffix.lower() not in {".yaml", ".yml"}:
            continue

        is_valid, issues = check_file(file_path, fix=args.fix, show_diff=args.diff)

        if not is_valid:
            all_valid = False
            total_issues += len(issues)
            print(f"{file_path}:")
            for issue in issues:
                print(issue)
        elif args.verbose:
            print(f"{file_path}: OK")

    if not all_valid:
        print(f"\n{total_issues} formatting issue(s) found")
        return 1

    if args.verbose:
        print(f"\nAll {len(files)} files passed format checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
