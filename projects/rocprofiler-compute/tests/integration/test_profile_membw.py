# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Integration tests for memory bandwidth analysis: profile then analyze."""

from pathlib import Path

import common
import pytest

from tests.integration.common import config

# Every membw analysis run emits exactly one of these strings.
MEMBW_OUTCOME_STRINGS = (
    "Memory Bandwidth Guided Analysis",
    "Memory Bandwidth Analysis: No bottlenecks detected (GL1 / GL2 / EA).",
    "Memory Bandwidth Analysis: Bottlenecks detected (see chart annotations).",
    "Memory Bandwidth Analysis: Unavailable",
    "Memory Bandwidth Analysis: Partial data",
    "Memory Bandwidth Analysis: Inconclusive",
)


def assert_membw_analysis_ran(output: str) -> None:
    """Assert that the membw analysis pipeline executed and produced output."""
    assert any(s in output for s in MEMBW_OUTCOME_STRINGS), (
        "Expected one of the membw analysis output strings:\n"
        f"  {MEMBW_OUTCOME_STRINGS}\n"
        "but none were found.  Output tail:\n"
        f"  ...{output[-500:]}"
    )


def test_membw_profile_and_analyze(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
    soc,
    capsys,
):
    """Profile with --membw-analysis, then analyze and verify guidance output."""
    if soc != "MI350":
        pytest.skip(f"membw analysis requires MI350, got {soc!r}")

    workload_dir = common.get_output_dir()

    binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options=["--experimental", "--membw-analysis"],
        roof=False,
    )

    assert common.check_file_pattern(
        "membw_analysis: true", f"{workload_dir}/profiling_config.yaml"
    )

    results_files = sorted(Path(workload_dir).glob("results_*.csv.gz"))
    assert len(results_files) > 0

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
    ])
    assert code == 0

    output = capsys.readouterr().out
    assert_membw_analysis_ran(output)

    common.clean_output_dir(config["cleanup"], workload_dir)


def test_membw_analyze_block_filter(
    binary_handler_profile_rocprof_compute,
    binary_handler_analyze_rocprof_compute,
    soc,
    capsys,
):
    """Profile with --membw-analysis, analyze with -b 3 30, verify output."""
    if soc != "MI350":
        pytest.skip(f"membw analysis requires MI350, got {soc!r}")

    workload_dir = common.get_output_dir()

    binary_handler_profile_rocprof_compute(
        config,
        workload_dir,
        options=["--experimental", "--membw-analysis"],
        roof=False,
    )

    code = binary_handler_analyze_rocprof_compute([
        "analyze",
        "--path",
        workload_dir,
        "-b",
        "3",
        "30",
    ])
    assert code == 0

    output = capsys.readouterr().out
    assert "Memory Chart" in output
    assert_membw_analysis_ran(output)

    common.clean_output_dir(config["cleanup"], workload_dir)
