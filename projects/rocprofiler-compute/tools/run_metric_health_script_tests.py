#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Run unit tests for the metric-health utility scripts.

Those scripts are not product code. Their tests stay out of ctest. A local
pre-commit hook runs them when one of the scripts changes. GitHub Actions
uses the same hook config, so this runner does nothing there.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_PATHS: Tuple[str, ...] = (
    "tests/unit/test_generate_metric_health_report.py",
    "tests/unit/test_run_metric_health_test_report_local_driver.py",
)


def main() -> int:
    """Run the utility-script unit tests from the project root."""
    if os.environ.get("GITHUB_ACTIONS") == "true":
        return 0
    # Skip tests/conftest.py. It imports the product stack, and these script
    # tests only use pytest's own fixtures.
    command = [sys.executable, "-m", "pytest", "--noconftest", *TEST_PATHS]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    return completed.returncode


if __name__ == "__main__":
    sys.exit(main())
