# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for rocprof_compute_base.py."""

import argparse
import signal
from unittest.mock import Mock

import common
import pytest

import rocprof_compute_base
from rocprof_compute_base import RocProfCompute

MODULE = "rocprof_compute_base"
# Profile selection fields that _validate_profile_mode_arguments reads
PROFILE_SELECTION = {
    "filter_blocks": [],
    "set_selected": None,
    "bench_only": False,
    "no_roof": False,
}
# List option fields that _validate_list_option_exclusions reads
LIST_OPTIONS = {
    "debug_list_metrics_gfx_id": None,
    "list_metrics": None,
    "list_blocks": None,
    "list_available_metrics": False,
    "filter_blocks": [],
}


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
    "selection, expect_error",
    [
        ({"filter_blocks": ["2", "4"]}, False),
        ({"set_selected": "launch_stats"}, False),
        ({"bench_only": True}, False),
        ({"filter_blocks": ["2"], "set_selected": "launch_stats"}, True),
        ({"filter_blocks": ["4"], "bench_only": True}, True),
        ({"set_selected": "launch_stats", "bench_only": True}, True),
        ({"bench_only": True, "no_roof": True}, True),
    ],
    ids=[
        "blocks",
        "set",
        "bench_only",
        "blocks_with_set",
        "blocks_with_bench_only",
        "set_with_bench_only",
        "bench_only_with_no_roof",
    ],
)
def test_validate_profile_mode_arguments(selection, expect_error):
    """Blocks, --set and --roofline-bench-only exclude each other."""
    args = argparse.Namespace(mode="profile", **{**PROFILE_SELECTION, **selection})
    instance = rocprof_compute_with_args(args)
    if expect_error:
        with pytest.raises(SystemExit):
            instance._validate_profile_mode_arguments()
    else:
        instance._validate_profile_mode_arguments()


@pytest.mark.parametrize(
    "options, named_option",
    [
        ({"debug_list_metrics_gfx_id": "gfx90a"}, "with --list-metrics"),
        ({"list_metrics": ["sol"], "filter_blocks": ["2"]}, "--list-metrics"),
        ({"list_blocks": "gfx950", "filter_blocks": ["2"]}, "--list-blocks"),
        (
            {"list_available_metrics": True, "filter_blocks": ["2"]},
            "--list-available-metrics",
        ),
    ],
    ids=[
        "gfx_id_requires_list_metrics",
        "list_metrics_with_blocks",
        "list_blocks_with_blocks",
        "list_available_metrics_with_blocks",
    ],
)
def test_validate_list_option_exclusions(monkeypatch, options, named_option):
    """Each rejected combination names the list option in the error."""
    error = common.patch_console(
        monkeypatch, MODULE, "error", error=Mock(side_effect=SystemExit(1))
    )["error"]
    args = argparse.Namespace(mode=None, **{**LIST_OPTIONS, **options})
    with pytest.raises(SystemExit):
        rocprof_compute_with_args(args)._validate_list_option_exclusions()
    assert named_option in error.call_args.args[0]


def test_sigterm_exits_so_cleanup_runs():
    """SIGTERM raises SystemExit, so finally blocks clean up temporary folders."""
    with pytest.raises(SystemExit) as exit_info:
        rocprof_compute_base._exit_on_sigterm(signal.SIGTERM, None)

    assert exit_info.value.code == 128 + signal.SIGTERM
