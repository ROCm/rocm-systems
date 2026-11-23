# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for operator pattern binding."""

import pytest

from ml_api_tracing.analyze.operator_patterns import (
    bind_operator_patterns,
    check_backends_traced,
    split_backend_prefix,
    traced_frameworks_from_config,
)


@pytest.mark.parametrize(
    "profiling_config, expected",
    [
        ({}, []),
        ({"trace_ml_api": None}, []),
        ({"trace_ml_api": ["torch", "triton"]}, ["torch", "triton"]),
    ],
)
def test_traced_frameworks_from_config(profiling_config, expected):
    assert traced_frameworks_from_config(profiling_config) == expected


def test_check_backends_traced_accepts_user_regions():
    check_backends_traced(["torch", "user"], ["torch"])


@pytest.mark.parametrize(
    "backends, traced_frameworks, message",
    [
        pytest.param(["triton"], ["torch"], "--trace-ml-api triton", id="untraced"),
        pytest.param([], [], "Profile again with --trace-ml-api", id="no_trace"),
    ],
)
def test_check_backends_traced_rejects(backends, traced_frameworks, message):
    with pytest.raises(ValueError, match=message):
        check_backends_traced(backends, traced_frameworks)


@pytest.mark.parametrize(
    "pattern, expected",
    [
        ("torch:*conv*", ("torch", "*conv*")),
        ("torch:*/aten::addmm", ("torch", "*/aten::addmm")),
        ("*/aten::addmm", (None, "*/aten::addmm")),
        ("*relu*", (None, "*relu*")),
        ("user:", ("user", "**")),
    ],
)
def test_split_backend_prefix(pattern, expected):
    assert split_backend_prefix(pattern) == expected


@pytest.mark.parametrize(
    "patterns, traced_frameworks, expected",
    [
        pytest.param(
            ["torch:*conv*", "*relu*", "triton:*matmul*"],
            ["torch", "triton"],
            {"torch": ["*conv*", "*relu*"], "triton": ["*matmul*"]},
            id="prefix_carries_forward",
        ),
        pytest.param(
            ["*relu*"],
            ["torch"],
            {"torch": ["*relu*"], "user": ["*relu*"]},
            id="no_prefix_matches_traced_and_user",
        ),
        pytest.param(
            [],
            ["triton"],
            {"triton": ["**"], "user": ["**"]},
            id="no_patterns_match_everything",
        ),
        pytest.param(
            ["user:*UserDefined*"],
            ["torch"],
            {"user": ["*UserDefined*"]},
            id="user_prefix",
        ),
    ],
)
def test_bind_operator_patterns(patterns, traced_frameworks, expected):
    assert bind_operator_patterns(patterns, traced_frameworks) == expected


@pytest.mark.parametrize(
    "patterns, traced_frameworks, message",
    [
        pytest.param(["toch:*relu*"], ["torch"], "prefix 'toch'", id="unknown"),
        pytest.param(["triton:*mm*"], ["torch"], "api triton", id="untraced"),
        pytest.param(["*relu*"], [], "Profile again", id="no_trace"),
    ],
)
def test_bind_operator_patterns_rejects(patterns, traced_frameworks, message):
    with pytest.raises(ValueError, match=message):
        bind_operator_patterns(patterns, traced_frameworks)


def test_operator_comma_joined_token_is_one_glob():
    assert bind_operator_patterns(
        ["torch:*conv*,*relu*,triton:*matmul*"], ["torch"]
    ) == {"torch": ["*conv*,*relu*,triton:*matmul*"]}
