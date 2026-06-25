# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for src/utils/tty.py."""

import argparse

import pandas as pd

from utils.tty import format_table_output


def make_args() -> argparse.Namespace:
    """Minimal args for the plain-table render path."""
    return argparse.Namespace(decimal=2, view=None, normal_unit="per_wave")


def test_format_table_output_suppresses_empty_column() -> None:
    """A non-PC-sampling table with an all-'N/A' column is suppressed."""
    df = pd.DataFrame({"Metric": ["a", "b"], "Value": ["N/A", "N/A"]})
    content = format_table_output(
        make_args(),
        {"id": 1101, "title": "Some Table"},
        df,
        "metric_table",
        runs={"only": object()},
    )
    assert content == ""


def test_format_table_output_keeps_pc_sampling_table_21_1() -> None:
    """PC sampling table 21.1 is shown even with an all-'N/A' source column."""
    df = pd.DataFrame({
        "source_line": ["N/A", "N/A"],
        "instruction": ["v_mov", "v_add"],
        "count": [3, 1],
    })
    content = format_table_output(
        make_args(),
        {"id": 2101, "title": "PC Sampling"},
        df,
        "pc_sampling_table",
        runs={"only": object()},
    )
    assert content != ""
    assert "v_mov" in content
