# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/ml_api_selectors.py."""

import pytest

from utils.inject_roctx.constants import KNOWN_ML_API_BACKENDS
from utils.ml_api_selectors import (
    UNSCOPED_OPERATOR_KEY,
    authorize_unscoped_operator_patterns,
    resolve_operator_patterns,
    resolve_profile_selection,
    sanitize_ml_api_selector_args,
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


def test_sanitize_rejects_named_backend_missing_from_workload():
    with pytest.raises(ValueError, match="triton"):
        sanitize_ml_api_selector_args(["triton"], None, ["torch"])
    with pytest.raises(ValueError, match="triton"):
        sanitize_ml_api_selector_args(
            None, resolve_operator_patterns(["triton:*matmul*"]), ["torch"]
        )


def test_sanitize_rejects_operator_when_workload_has_no_trace():
    with pytest.raises(ValueError, match="was not profiled"):
        sanitize_ml_api_selector_args(None, resolve_operator_patterns([]), [])


def test_sanitize_does_not_treat_unscoped_as_a_named_backend():
    sanitize_ml_api_selector_args(
        None, resolve_operator_patterns(["*relu*"]), ["torch"]
    )
    sanitize_ml_api_selector_args(None, resolve_operator_patterns([]), ["triton"])


def test_authorize_copies_unscoped_onto_collected_frameworks_and_user():
    assert authorize_unscoped_operator_patterns(
        resolve_operator_patterns(["*relu*"]), ["torch"]
    ) == {"torch": ["*relu*"], "user": ["*relu*"]}
    assert authorize_unscoped_operator_patterns(
        resolve_operator_patterns([]), ["triton"]
    ) == {"triton": ["**"], "user": ["**"]}


def test_authorize_does_not_invent_uncollected_frameworks():
    bound = authorize_unscoped_operator_patterns(
        resolve_operator_patterns([]), ["triton"]
    )
    assert "torch" not in bound
