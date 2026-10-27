# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/ml_api_selectors.py."""

from utils.inject_roctx.constants import KNOWN_ML_API_BACKENDS
from utils.ml_api_selectors import (
    UNSCOPED_OPERATOR_KEY,
    resolve_operator_patterns,
    resolve_profile_selection,
)


def test_absent_profile_selection_is_empty():
    assert resolve_profile_selection(None) == []


def test_bare_profile_selection_saves_known_backends():
    assert resolve_profile_selection([]) == list(KNOWN_ML_API_BACKENDS)


def test_comma_and_space_profile_tokens_save_the_same_pair():
    comma = resolve_profile_selection(["torch,triton"])
    space = resolve_profile_selection(["torch", "triton"])
    assert comma == space == ["torch", "triton"]


def test_double_colon_stays_in_the_operator_pattern():
    assert resolve_operator_patterns(["torch:*/aten::addmm"]) == {
        "torch": ["*/aten::addmm"]
    }
    assert resolve_operator_patterns(["*/aten::addmm"]) == {
        UNSCOPED_OPERATOR_KEY: ["*/aten::addmm"]
    }


def test_operator_space_and_comma_forms_select_the_same_patterns():
    expected = {
        "torch": ["*conv*", "*relu*"],
        "triton": ["*matmul*"],
    }
    forms = [
        "torch:*conv* *relu* triton:*matmul*",
        "torch:*conv* *relu*,triton:*matmul*",
        "torch:*conv*,*relu* triton:*matmul*",
        "torch:*conv*,*relu*,triton:*matmul*",
    ]
    for form in forms:
        assert resolve_operator_patterns([form]) == expected


def test_empty_operator_flag_is_unscoped_double_star():
    assert resolve_operator_patterns([]) == {UNSCOPED_OPERATOR_KEY: ["**"]}
