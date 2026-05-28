#!/usr/bin/env python3
##############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
##############################################################################
"""Validate AI harness entry points and rules (rocprofiler-compute).

This is a lightweight integrity check intended for pre-commit / CI:

- Ensures a small set of project AI entry files and rule files exist.
- Avoids validating an entire skill library (to prevent duplication with
  OpenSpec + Superpowers).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


REQUIRED_PATHS: tuple[str, ...] = (
    # Harness validator itself
    "scripts/ai_dev_guide.py",
    # Agent entry point(s)
    "AGENTS.md",
    # Core rules in this repo
    ".ai/rules/python-style.md",
    ".ai/rules/ruff-tooling.md",
    ".ai/rules/commit-workflow.md",
    ".ai/rules/pr-workflow.md",
    # Domain-specific rules for profiler work
    ".ai/rules/profiling_infra.md",
    ".ai/rules/security.md",
    # Spec-driven workflow configuration
    "openspec/config.yaml",
    # Team proposal doc (workflow contract)
    "docs/design/spec-driven-development-proposal.md",
)


def _check_required_paths(verbose: bool) -> list[str]:
    errors: list[str] = []
    for rel in REQUIRED_PATHS:
        path = PROJECT_ROOT / rel
        if not path.is_file():
            errors.append(f"Missing required file: {rel}")
        elif verbose:
            print(f"OK {rel}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate AI harness integrity.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print checked paths.")
    args = parser.parse_args()

    if not (PROJECT_ROOT / "pyproject.toml").is_file():
        print(
            "Run from the rocprofiler-compute project root (pyproject.toml must exist).",
            file=sys.stderr,
        )
        return 2

    errors = _check_required_paths(args.verbose)
    if errors:
        print("ai_dev_guide: failed", file=sys.stderr)
        for line in errors:
            print(f"  {line}", file=sys.stderr)
        return 1

    if args.verbose:
        print("ai_dev_guide: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

