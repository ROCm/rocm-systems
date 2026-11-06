# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
import subprocess

import pytest


@pytest.mark.parametrize(
    "options",
    [
        ["--start", "main"],
        ["--timeout", "5"],
        ["--start", "main", "--timeout", "0ms"],
        ["--batch"],
        ["--att-no-intercept"],
        ["--attach", "1"],
        ["-p1"],
        ["-iinput.yaml"],
        ["--selected-regions-ref-count"],
        ["--start", "main", "--stop", "end", "--skip", "-1"],
        ["--start", "main", "--stop", "end", "--skip", "1.5"],
        ["--start", "main", "--stop", "end", "--skip", "2147483648"],
        ["--skip", "100"],
    ],
)
def test_reject_invalid_workflow(launcher, options):
    result = subprocess.run(
        [launcher, *options, "--", "/bin/true"], capture_output=True, text=True, timeout=5
    )
    assert result.returncode == 2
    assert "error:" in result.stderr


def test_help_explains_workflow(launcher):
    result = subprocess.run(
        [launcher, "--help"], capture_output=True, text=True, timeout=5
    )
    assert result.returncode == 0
    assert "--start" in result.stdout and "--timeout" in result.stdout
    assert "--skip" in result.stdout and "hit 101" in result.stdout
    assert "att arm" in result.stdout
