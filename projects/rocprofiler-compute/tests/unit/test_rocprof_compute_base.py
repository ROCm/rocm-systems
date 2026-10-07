# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for rocprof-compute option sanitization."""

import argparse

import pytest

from rocprof_compute_base import RocProfCompute


def make_analyze_instance(output_format, list_stats):
    """Build an instance for option validation without initializing the CLI."""
    instance = RocProfCompute.__new__(RocProfCompute)
    instance._RocProfCompute__args = argparse.Namespace(
        mode="analyze", output_format=output_format, list_stats=list_stats
    )
    instance._RocProfCompute__mode = "analyze"
    instance._RocProfCompute__analyze_mode = None
    return instance


@pytest.mark.parametrize("output_format", ["csv", "db"])
def test_sanitize_rejects_list_stats_with_db_output(output_format, caplog):
    instance = make_analyze_instance(output_format, list_stats=True)

    with pytest.raises(SystemExit) as error:
        instance.sanitize()

    assert error.value.code == 1
    assert "--list-stats cannot be used with --output-format csv or db" in caplog.text
    assert "Use --output-format stdout or txt" in caplog.text


@pytest.mark.parametrize(
    ("output_format", "list_stats", "expected_mode"),
    [
        ("stdout", True, "cli"),
        ("txt", True, "cli"),
        ("stdout", False, "cli"),
        ("txt", False, "cli"),
        ("csv", False, "db"),
        ("db", False, "db"),
    ],
)
def test_sanitize_accepts_supported_analysis_options(
    output_format, list_stats, expected_mode
):
    instance = make_analyze_instance(output_format, list_stats)

    instance.sanitize()

    assert instance._RocProfCompute__analyze_mode == expected_mode
