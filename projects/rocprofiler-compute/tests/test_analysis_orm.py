# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for analysis_orm.py static methods."""

import json
import math

import numpy as np

from utils.analysis_orm import Database


def test_sanitize_replaces_nan_with_none():
    result = Database._sanitize_for_json({"a": float("nan"), "b": 1.0})
    assert result == {"a": None, "b": 1.0}


def test_sanitize_replaces_inf_with_none():
    result = Database._sanitize_for_json({"a": float("inf"), "b": float("-inf")})
    assert result == {"a": None, "b": None}


def test_sanitize_replaces_numpy_nan_with_none():
    result = Database._sanitize_for_json({"a": np.float64("nan")})
    assert result == {"a": None}


def test_sanitize_recurses_into_nested_containers():
    result = Database._sanitize_for_json({"a": [{"b": float("nan")}]})
    assert result == {"a": [{"b": None}]}


def test_sanitize_preserves_finite_and_non_float_values():
    payload = {"x": 1, "y": "text", "z": 2.5, "w": None}
    assert Database._sanitize_for_json(payload) == payload


def test_sanitized_dict_serializes_to_valid_json():
    sanitized = Database._sanitize_for_json({"a": float("nan"), "b": 1})
    assert json.loads(json.dumps(sanitized, allow_nan=False)) == {"a": None, "b": 1}


def test_sanitize_leaves_unmangled_dict_unchanged():
    payload = {"a": 1.0, "b": [2, 3], "c": {"d": "e"}}
    assert Database._sanitize_for_json(payload) == payload
    assert math.isfinite(payload["a"])
