# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for rocprof_compute_base.py."""

import argparse

import pytest

from rocprof_compute_base import RocProfCompute


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


def test_handle_analyze_args_listing_skips_operator_parse(monkeypatch):
    """--list-operators ignores --operator, including an unknown prefix."""
    args = build_args(list_operators=["torch"], operator=["unknown:*"])
    warnings = []
    monkeypatch.setattr(
        "rocprof_compute_base.console_warning",
        lambda *argv: warnings.append(argv),
    )
    instance = rocprof_compute_with_args(args)
    instance.handle_analyze_args()
    assert args.list_operators == ["torch"]
    assert args.operator_patterns is None
    assert "ignored" in warnings[-1][1]


@pytest.mark.parametrize("flag", ["list_operators", "operator"])
@pytest.mark.parametrize(
    "output_format, expect_error",
    [("stdout", False), ("txt", True), ("csv", True), ("db", True)],
)
def test_handle_analyze_args_operator_flags_require_stdout(
    flag, output_format, expect_error, monkeypatch
):
    """--list-operators and --operator are rejected unless output is stdout."""
    if flag == "list_operators":
        args = build_args(output_format=output_format, list_operators=["torch"])
    else:
        args = build_args(output_format=output_format, operator=["torch:*relu*"])
    instance = rocprof_compute_with_args(args)
    if expect_error:
        error_text = None

        def record_error(*argv, **_kwargs):
            nonlocal error_text
            error_text = argv[1]
            raise SystemExit(1)

        monkeypatch.setattr("rocprof_compute_base.console_error", record_error)
        with pytest.raises(SystemExit):
            instance.handle_analyze_args()
        assert "only supported with --output-format stdout" in error_text
    else:
        instance.handle_analyze_args()
