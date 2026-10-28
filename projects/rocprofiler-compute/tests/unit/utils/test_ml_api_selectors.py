# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/ml_api_selectors.py."""

import pytest

from utils.inject_roctx.constants import KNOWN_ML_API_BACKENDS
from utils.ml_api_selectors import (
    UNSCOPED_OPERATOR_KEY,
    check_list_operators_against_traced_frameworks,
    operator_patterns_for_traced_frameworks,
    resolve_operator_patterns,
    resolve_profile_selection,
)


def test_absent_profile_selection_is_empty():
    assert resolve_profile_selection(None) == []


def test_bare_profile_selection_saves_known_backends():
    assert resolve_profile_selection([]) == list(KNOWN_ML_API_BACKENDS)


def test_space_profile_tokens_save_known_backends():
    assert resolve_profile_selection(["torch", "triton"]) == ["torch", "triton"]


def test_double_colon_stays_in_the_operator_pattern():
    assert resolve_operator_patterns(["torch:*/aten::addmm"]) == {
        "torch": ["*/aten::addmm"]
    }
    assert resolve_operator_patterns(["*/aten::addmm"]) == {
        UNSCOPED_OPERATOR_KEY: ["*/aten::addmm"]
    }


def test_operator_space_separated_patterns_keep_prefix():
    expected = {
        "torch": ["*conv*", "*relu*"],
        "triton": ["*matmul*"],
    }
    assert (
        resolve_operator_patterns(["torch:*conv*", "*relu*", "triton:*matmul*"])
        == expected
    )
    assert resolve_operator_patterns(["torch:*conv*,*relu*,triton:*matmul*"]) == {
        "torch": ["*conv*,*relu*,triton:*matmul*"]
    }


def test_empty_operator_flag_is_unscoped_double_star():
    assert resolve_operator_patterns([]) == {UNSCOPED_OPERATOR_KEY: ["**"]}


def test_operator_rejects_unknown_prefix():
    with pytest.raises(ValueError, match="jax"):
        resolve_operator_patterns(["jax:*foo*"])
    with pytest.raises(ValueError, match="all_supported_backends"):
        resolve_operator_patterns(["all_supported_backends:*relu*"])
    with pytest.raises(ValueError, match="toch"):
        resolve_operator_patterns(["toch:*relu* torch:*relu*"])


def test_list_operators_rejects_backend_missing_from_workload():
    with pytest.raises(ValueError, match="triton"):
        check_list_operators_against_traced_frameworks(["triton"], ["torch"])


def test_operator_patterns_reject_backend_missing_from_workload():
    with pytest.raises(ValueError, match="triton"):
        operator_patterns_for_traced_frameworks(
            resolve_operator_patterns(["triton:*matmul*"]), ["torch"]
        )


def test_operator_patterns_reject_when_workload_has_no_trace():
    with pytest.raises(ValueError, match="was not profiled"):
        operator_patterns_for_traced_frameworks(resolve_operator_patterns([]), [])


def test_operator_patterns_attach_unscoped_to_traced_frameworks_and_user():
    assert operator_patterns_for_traced_frameworks(
        resolve_operator_patterns(["*relu*"]), ["torch"]
    ) == {"torch": ["*relu*"], "user": ["*relu*"]}
    patterns_by_backend = operator_patterns_for_traced_frameworks(
        resolve_operator_patterns([]), ["triton"]
    )
    assert patterns_by_backend == {"triton": ["**"], "user": ["**"]}
    assert "torch" not in patterns_by_backend


def test_list_operators_bare_flag_does_not_require_a_named_backend():
    check_list_operators_against_traced_frameworks([], ["torch"])
