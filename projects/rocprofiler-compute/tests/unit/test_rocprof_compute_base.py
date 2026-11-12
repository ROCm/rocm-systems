# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for rocprof_compute_base.py."""

import argparse
from pathlib import Path
from unittest.mock import Mock

import common
import pytest
from common import SUPPORTED_ARCHS

from argparser import build_parser
from rocprof_compute_base import RocProfCompute

MODULE = "rocprof_compute_base"
SELECTION_CONFLICT = (
    "--set and --roofline-bench-only cannot be used with each other or with block "
    "selection (--block, --speed-of-light, --memory-chart, --roofline)."
)


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


def parse_profile_args(options: list[str]) -> argparse.Namespace:
    argv = ["profile", *options, "--", "./app"]
    parser = build_parser(Path.cwd(), SUPPORTED_ARCHS, {"ver_pretty": "test"}, argv)
    return parser.parse_args(argv)


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


@pytest.mark.parametrize(
    ("options", "expected_roof_only"),
    [(["--roofline"], True), (["--roofline", "-b", "2"], False)],
    ids=["roofline_alone", "with_block"],
)
def test_handle_profile_args_roof_only(options, expected_roof_only):
    """--roofline is applied as block 4 before roof_only is computed."""
    args = parse_profile_args(options)
    rocprof_compute_with_args(args).handle_profile_args()
    assert args.roof_only is expected_roof_only


@pytest.mark.parametrize(
    ("options", "error"),
    [
        (["-b", "2", "--roofline"], None),
        (["--speed-of-light", "--memory-chart", "--roofline"], None),
        (["--set", "launch_stats"], None),
        (["--speed-of-light", "--set", "launch_stats"], SELECTION_CONFLICT),
        (["--roofline", "--roofline-bench-only"], SELECTION_CONFLICT),
        (["--set", "launch_stats", "--roofline-bench-only"], SELECTION_CONFLICT),
        (["--roofline-bench-only", "--no-roof"],
         "--roofline-bench-only cannot be used with --no-roof."),
    ],
    ids=[
        "block_with_roofline",
        "all_shortcuts",
        "set_alone",
        "shortcut_with_set",
        "roofline_with_bench_only",
        "set_with_bench_only",
        "bench_only_with_no_roof",
    ],
)  # fmt: skip
def test_validate_profile_mode_arguments(options, error, caplog):
    """Block selection, --set and --roofline-bench-only exclude each other."""
    instance = rocprof_compute_with_args(parse_profile_args(options))
    instance.handle_profile_args()
    if error is None:
        instance._validate_profile_mode_arguments()
        return
    with pytest.raises(SystemExit):
        instance._validate_profile_mode_arguments()
    assert error in caplog.text


@pytest.mark.parametrize(
    ("options", "error"),
    [
        (["--list-available-metrics", "--speed-of-light"],
         "Cannot use --list-available-metrics with -b/--block, --speed-of-light, "
         "--memory-chart or --roofline."),
        (["--list-metrics", "sol", "--roofline"],
         "Pass block ids or aliases to --list-metrics instead of -b/--block, "
         "--speed-of-light, --memory-chart or --roofline, for example "
         "--list-metrics sol."),
        (["--list-blocks", "gfx950", "-b", "2"],
         "Cannot use --list-blocks with -b/--block, --speed-of-light, "
         "--memory-chart or --roofline."),
    ],
    ids=["list_available_metrics", "list_metrics", "list_blocks"],
)  # fmt: skip
def test_validate_list_option_exclusions(options, error, caplog):
    """The error names the options the user can actually pass."""
    instance = rocprof_compute_with_args(parse_profile_args(options))
    instance.handle_profile_args()
    with pytest.raises(SystemExit):
        instance._validate_list_option_exclusions()
    assert error in caplog.text


def test_validate_list_option_exclusions_gfx_id_requires_list_metrics(monkeypatch):
    """--debug-list-metrics-gfx-id is rejected without --list-metrics."""
    error = common.patch_console(
        monkeypatch, MODULE, "error", error=Mock(side_effect=SystemExit(1))
    )["error"]
    args = argparse.Namespace(
        mode=None, debug_list_metrics_gfx_id="gfx90a", list_metrics=None
    )
    with pytest.raises(SystemExit):
        rocprof_compute_with_args(args)._validate_list_option_exclusions()
    assert "with --list-metrics" in error.call_args.args[0]
