# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for utils/ml_api_selectors.py."""

import pytest

from utils.ml_api_selectors import (
    JAX_NOT_COLLECTED_WARNING,
    resolve_list_selectors,
    resolve_operator_patterns,
    resolve_profile_selection,
)


def test_absent_profile_selection_disables_marker_trace():
    selection = resolve_profile_selection(None)
    assert selection.saved == []
    assert selection.injectable == []
    assert selection.marker_trace is False
    assert selection.warning is None


def test_bare_profile_selection_saves_defaults_without_jax_warning():
    selection = resolve_profile_selection([])
    assert selection.saved == ["torch", "triton", "jax"]
    assert selection.injectable == ["torch", "triton"]
    assert selection.marker_trace is True
    assert selection.warning is None


def test_named_torch_profile_selection_injects_only_torch():
    selection = resolve_profile_selection(["torch"])
    assert selection.saved == ["torch"]
    assert selection.injectable == ["torch"]
    assert selection.marker_trace is True
    assert selection.warning is None


def test_user_profile_selection_raises():
    with pytest.raises(ValueError, match="user is not a --trace-ml-api value"):
        resolve_profile_selection(["user"])


def test_jax_profile_selection_enables_marker_trace_without_injection():
    selection = resolve_profile_selection(["jax"])
    assert selection.saved == ["jax"]
    assert selection.injectable == []
    assert selection.marker_trace is True
    assert selection.warning == JAX_NOT_COLLECTED_WARNING


def test_torch_and_jax_profile_selection_injects_torch_and_warns():
    selection = resolve_profile_selection(["torch", "jax"])
    assert selection.saved == ["torch", "jax"]
    assert selection.injectable == ["torch"]
    assert selection.warning == JAX_NOT_COLLECTED_WARNING


def test_comma_and_space_profile_tokens_save_the_same_pair():
    comma = resolve_profile_selection(["torch,triton"])
    space = resolve_profile_selection(["torch", "triton"])
    assert comma.saved == space.saved == ["torch", "triton"]


def test_colon_operator_token_is_explicit_torch_pattern():
    resolved = resolve_operator_patterns(["torch:*/aten::addmm"])
    assert resolved.explicit == {"torch": ["*/aten::addmm"]}
    assert resolved.unscoped == []
    assert resolved.jax_requested is False


def test_unscoped_aten_pattern_is_not_copied_onto_triton():
    resolved = resolve_operator_patterns(["*/aten::addmm"])
    assert resolved.explicit == {}
    assert resolved.unscoped == ["*/aten::addmm"]


def test_comma_patterns_stay_on_the_same_prefix():
    resolved = resolve_operator_patterns(["torch:*conv*,*relu*"])
    assert resolved.explicit == {"torch": ["*conv*", "*relu*"]}
    assert resolved.unscoped == []


def test_split_operator_tokens_stay_by_backend():
    resolved = resolve_operator_patterns(["torch:*conv*", "triton:*matmul*"])
    assert resolved.explicit == {"torch": ["*conv*"], "triton": ["*matmul*"]}
    assert resolved.unscoped == []


def test_empty_operator_flag_is_unscoped_double_star():
    resolved = resolve_operator_patterns([])
    assert resolved.explicit == {}
    assert resolved.unscoped == ["**"]
    assert resolved.jax_requested is False


def test_jax_operator_sets_flag_without_match_pattern():
    resolved = resolve_operator_patterns(["jax:*dot*"])
    assert resolved.jax_requested is True
    assert resolved.explicit == {}
    assert resolved.unscoped == []


def test_empty_list_selectors_stay_empty():
    assert resolve_list_selectors([]) == []


def test_list_selectors_accept_user():
    assert resolve_list_selectors(["user"]) == ["user"]
