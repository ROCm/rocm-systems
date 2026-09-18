# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for rocprof_compute_analyze/analysis_cli.py."""

import pytest

# -- parse_operator_patterns (torch_operator) -------------------------------


@pytest.mark.torch_ops
def test_parse_patterns_basic():
    """Single and multiple patterns are parsed correctly."""
    from argparse import Namespace

    from rocprof_compute_analyze.analysis_cli import parse_operator_patterns

    args = Namespace(torch_operator=["relu"])
    assert parse_operator_patterns(args, "torch_operator") == ["relu"]

    args = Namespace(torch_operator=["relu", "conv2d"])
    assert parse_operator_patterns(args, "torch_operator") == ["relu", "conv2d"]


@pytest.mark.torch_ops
def test_parse_patterns_comma_split():
    """Comma-separated patterns in a single arg are split."""
    from argparse import Namespace

    from rocprof_compute_analyze.analysis_cli import parse_operator_patterns

    args = Namespace(torch_operator=["relu,conv2d"])
    assert parse_operator_patterns(args, "torch_operator") == ["relu", "conv2d"]


@pytest.mark.torch_ops
def test_parse_patterns_whitespace():
    """Leading/trailing whitespace is stripped."""
    from argparse import Namespace

    from rocprof_compute_analyze.analysis_cli import parse_operator_patterns

    args = Namespace(torch_operator=["  relu  ", " conv2d , linear "])
    result = parse_operator_patterns(args, "torch_operator")
    assert result == ["relu", "conv2d", "linear"]


@pytest.mark.torch_ops
def test_parse_patterns_empty():
    """Flag given with no args defaults to '**'; absent flag returns empty."""
    from argparse import Namespace

    from rocprof_compute_analyze.analysis_cli import parse_operator_patterns

    parse = parse_operator_patterns
    assert parse(Namespace(torch_operator=[]), "torch_operator") == ["**"]
    assert parse(Namespace(torch_operator=None), "torch_operator") == []
    assert parse(Namespace(), "torch_operator") == []


# -- parse_operator_patterns / triton backend selection ---------------------


@pytest.mark.torch_ops
def test_parse_operator_patterns_generic_attr():
    """parse_operator_patterns reads the given dest attribute."""
    from argparse import Namespace

    from rocprof_compute_analyze.analysis_cli import parse_operator_patterns

    args = Namespace(triton_operator=["*matmul*,*softmax*"], torch_operator=None)
    assert parse_operator_patterns(args, "triton_operator") == [
        "*matmul*",
        "*softmax*",
    ]
    assert parse_operator_patterns(args, "triton_operator") != parse_operator_patterns(
        args, "torch_operator"
    )
    assert parse_operator_patterns(
        Namespace(triton_operator=[]), "triton_operator"
    ) == ["**"]


@pytest.mark.torch_ops
def test_parse_patterns_star():
    """'*' is passed through as-is by the pattern parser."""
    from argparse import Namespace

    from rocprof_compute_analyze.analysis_cli import parse_operator_patterns

    args = Namespace(torch_operator=["*"])
    assert parse_operator_patterns(args, "torch_operator") == ["*"]

    args = Namespace(torch_operator=["*,torch.relu"])
    assert parse_operator_patterns(args, "torch_operator") == ["*", "torch.relu"]
