# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for rocprof_compute_base.py."""

import argparse
import sys
from unittest.mock import Mock

import common
import pytest

from rocprof_compute_base import RocProfCompute

MODULE = "rocprof_compute_base"


def build_args(
    *,
    mode="analyze",
    list_stats=False,
    output_format="stdout",
    list_operators=None,
    operator=None,
) -> argparse.Namespace:
    """Build a Namespace with the fields handle_analyze_args reads."""
    return argparse.Namespace(
        mode=mode,
        list_stats=list_stats,
        output_format=output_format,
        list_operators=list_operators,
        operator=operator,
    )


def rocprof_compute_with_args(args: argparse.Namespace) -> RocProfCompute:
    """Construct a RocProfCompute without invoking __init__."""
    instance = RocProfCompute.__new__(RocProfCompute)
    instance._RocProfCompute__args = args
    instance._RocProfCompute__mode = args.mode
    return instance


def print_startup_help(monkeypatch, argv) -> None:
    """Run RocProfCompute.parse_args on argv, which prints the help and exits."""
    instance = RocProfCompute.__new__(RocProfCompute)
    instance._RocProfCompute__supported_archs = common.SUPPORTED_ARCHS
    instance._RocProfCompute__version = {"ver_pretty": "unit test"}
    monkeypatch.setattr(sys, "argv", ["rocprof-compute", *argv])
    with pytest.raises(SystemExit):
        instance.parse_args()


@pytest.mark.parametrize(
    "argv, debug_option, shown",
    [
        (["--help"], "--debug-list-metrics-gfx-id", False),
        (["--debug", "--help"], "--debug-list-metrics-gfx-id", True),
        (["profile", "--help", "--debug"], "--debug-list-metrics-gfx-id", True),
        (["analyze", "--help"], "--debug-single-metric", False),
        (["analyze", "--debug", "--help"], "--debug-single-metric", True),
        (["analyze", "--help", "--debug"], "--debug-single-metric", True),
        (["analyze", "--debug", "--help"], "--debug-list-metrics-gfx-id", False),
    ],
)
def test_parse_args_shows_debug_options_with_debug(
    monkeypatch, capsys, argv, debug_option, shown
):
    """--debug, before or after --help, lists the debug options of the mode."""
    print_startup_help(monkeypatch, argv)
    assert (debug_option in capsys.readouterr().out) is shown


@pytest.mark.parametrize(
    "output_format, expect_error",
    [("stdout", False), ("txt", False), ("csv", True), ("db", True)],
)
def test_handle_analyze_args_list_stats_output_format(output_format, expect_error):
    """--list-stats is rejected with csv and db output."""
    args = build_args(list_stats=True, output_format=output_format)
    instance = rocprof_compute_with_args(args)
    if expect_error:
        with pytest.raises(SystemExit):
            instance.handle_analyze_args()
    else:
        instance.handle_analyze_args()


def test_handle_analyze_args_listing_ignores_operator_filter(monkeypatch):
    """--list-operators wins, and a warning says --operator is ignored."""
    warning = common.patch_console(monkeypatch, MODULE, "warning")["warning"]
    args = build_args(list_operators=["torch"], operator=["torch:*relu*"])
    rocprof_compute_with_args(args).handle_analyze_args()
    assert "--operator is ignored" in warning.call_args.args[1]


@pytest.mark.parametrize(
    "operator_flags",
    [{"list_operators": ["torch"]}, {"operator": ["torch:*relu*"]}],
)
@pytest.mark.parametrize(
    "output_format, expect_error",
    [("stdout", False), ("txt", True), ("csv", True), ("db", True)],
)
def test_handle_analyze_args_operator_flags_require_stdout(
    operator_flags, output_format, expect_error, monkeypatch
):
    """--list-operators and --operator are rejected unless output is stdout."""
    error = common.patch_console(
        monkeypatch, MODULE, "error", error=Mock(side_effect=SystemExit(1))
    )["error"]
    args = build_args(output_format=output_format, **operator_flags)
    instance = rocprof_compute_with_args(args)
    if expect_error:
        with pytest.raises(SystemExit):
            instance.handle_analyze_args()
        assert "only supported with --output-format stdout" in error.call_args.args[1]
    else:
        instance.handle_analyze_args()
