# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for rocprof_compute_base.py."""

import argparse
from pathlib import Path

import pytest
from common import SUPPORTED_ARCHS

from argparser import omniarg_parser
from rocprof_compute_base import RocProfCompute

SELECTION_CONFLICT = (
    "--set and --roofline-bench-only cannot be used with each other or with block "
    "selection (--block, --speed-of-light, --memory-chart, --roofline)."
)


def make_rpc_with_args(args: argparse.Namespace) -> RocProfCompute:
    """Construct a RocProfCompute without invoking __init__."""
    instance = RocProfCompute.__new__(RocProfCompute)
    # Name-mangled private attributes read by the methods under test
    instance._RocProfCompute__args = args
    instance._RocProfCompute__mode = args.mode
    return instance


def parse_profile_args(options: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    omniarg_parser(parser, Path.cwd(), SUPPORTED_ARCHS, {"ver_pretty": "test"})
    return parser.parse_args(["profile", *options, "--", "./app"])


@pytest.mark.parametrize(
    ("options", "expected_roof_only"),
    [(["--roofline"], True), (["--roofline", "-b", "2"], False)],
    ids=["roofline_alone", "with_block"],
)
def test_handle_profile_args_roof_only(options, expected_roof_only):
    """--roofline is applied as block 4 before roof_only is computed."""
    args = parse_profile_args(options)
    make_rpc_with_args(args).handle_profile_args()
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
    instance = make_rpc_with_args(parse_profile_args(options))
    instance.handle_profile_args()
    if error is None:
        instance._validate_profile_mode_arguments()
        return
    with pytest.raises(SystemExit):
        instance._validate_profile_mode_arguments()
    assert error in caplog.text
