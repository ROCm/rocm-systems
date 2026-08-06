# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for the profile helpers in ``utils_profile``.

The profiler's own output is logged at debug and the workload's at error.
"""

import pytest

from utils import utils_profile

# Emitted by rocprofiler-sdk and by the rocprofiler-compute tool.
_PROFILER_LINES = (
    "W0809 13:41:09.848003     633 tool.cpp:3477] rocprofv3 caught signal 6...",
    "I0000 00:00:1786282869.617312     633 tool.cpp:3775] rocprofv3_main(3, ...)",
    "[rocprofiler-sdk] tool initialization ::     0.146483 sec",
    "[rocprofiler-compute] In tool init",
)

_WORKLOAD_LINES = (
    "WARNING: All log messages before absl::InitializeLog() are written to STDERR",
    "Iteration 4096 loss 0.31",
    "W1234 is a tag the workload prints",
)


@pytest.fixture
def logged_level(monkeypatch) -> dict:
    """Return the level each classified line is logged at."""
    levels: dict[str, str] = {}
    monkeypatch.setattr(
        utils_profile, "console_debug", lambda line: levels.setdefault(line, "debug")
    )
    monkeypatch.setattr(
        utils_profile,
        "console_error",
        lambda line, exit=True: levels.setdefault(line, "error"),
    )
    return levels


@pytest.mark.parametrize("line", _PROFILER_LINES)
def test_the_profilers_own_output_is_logged_at_debug(
    line: str, logged_level: dict
) -> None:
    utils_profile._classify_output_line(line)

    assert logged_level[line] == "debug"


@pytest.mark.parametrize("line", _WORKLOAD_LINES)
def test_the_workloads_output_is_logged_at_error(line: str, logged_level: dict) -> None:
    utils_profile._classify_output_line(line)

    assert logged_level[line] == "error"
