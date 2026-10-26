# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/ml_api_selectors.py."""

import pytest

from utils.inject_roctx.constants import KNOWN_ML_API_BACKENDS
from utils.ml_api_selectors import (
    resolve_list_selectors,
    resolve_operator_patterns,
    resolve_profile_selection,
)


def test_absent_profile_selection_is_empty():
    assert resolve_profile_selection(None) == []


def test_bare_profile_selection_saves_known_backends():
    assert resolve_profile_selection([]) == list(KNOWN_ML_API_BACKENDS)


def test_named_torch_profile_selection_saves_only_torch():
    assert resolve_profile_selection(["torch"]) == ["torch"]


def test_user_profile_selection_raises():
    with pytest.raises(ValueError, match="user is not a --trace-ml-api value"):
        resolve_profile_selection(["user"])


def test_comma_and_space_profile_tokens_save_the_same_pair():
    comma = resolve_profile_selection(["torch,triton"])
    space = resolve_profile_selection(["torch", "triton"])
    assert comma == space == ["torch", "triton"]


def test_colon_operator_token_is_explicit_torch_pattern():
    assert resolve_operator_patterns(["torch:*/aten::addmm"]) == {
        "torch": ["*/aten::addmm"]
    }


def test_unscoped_aten_pattern_keeps_the_double_colon_name():
    pattern = ["*/aten::addmm"]
    assert resolve_operator_patterns(pattern) == {
        name: pattern for name in (*KNOWN_ML_API_BACKENDS, "user")
    }


def test_comma_patterns_stay_on_the_same_prefix():
    assert resolve_operator_patterns(["torch:*conv*,*relu*"]) == {
        "torch": ["*conv*", "*relu*"]
    }


def test_split_operator_tokens_stay_by_backend():
    assert resolve_operator_patterns(["torch:*conv*", "triton:*matmul*"]) == {
        "torch": ["*conv*"],
        "triton": ["*matmul*"],
    }


def test_bare_pattern_keeps_the_preceding_prefix_across_words():
    assert resolve_operator_patterns(
        ["torch:*conv*", "*relu*", "triton:*matmul*"]
    ) == {
        "torch": ["*conv*", "*relu*"],
        "triton": ["*matmul*"],
    }


def test_comma_joined_selectors_stay_by_backend():
    assert resolve_operator_patterns(["torch:*conv*,triton:*matmul*"]) == {
        "torch": ["*conv*"],
        "triton": ["*matmul*"],
    }


def test_space_joined_selectors_stay_by_backend():
    assert resolve_operator_patterns(["torch:*conv* triton:*matmul*"]) == {
        "torch": ["*conv*"],
        "triton": ["*matmul*"],
    }


def test_empty_operator_flag_is_unscoped_double_star():
    assert resolve_operator_patterns([]) == {
        name: ["**"] for name in (*KNOWN_ML_API_BACKENDS, "user")
    }


def test_empty_list_selectors_stay_empty():
    assert resolve_list_selectors([]) == []


def test_list_selectors_accept_user():
    assert resolve_list_selectors(["user"]) == ["user"]
