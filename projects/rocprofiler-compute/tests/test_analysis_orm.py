# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for analysis_orm.py static methods."""

import numpy as np
import pytest

from utils.analysis_orm import Database


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"k": float("nan")}, {"k": None}),
        ({"k": float("inf")}, {"k": None}),
        ({"k": float("-inf")}, {"k": None}),
        ({"k": np.float64("nan")}, {"k": None}),
        (
            {"i": 1, "f": 2.5, "s": "text", "n": None},
            {"i": 1, "f": 2.5, "s": "text", "n": None},
        ),
        ({"a": [{"b": float("nan")}]}, {"a": [{"b": None}]}),
        ({"a": ({"b": float("inf")},)}, {"a": [{"b": None}]}),
    ],
    ids=[
        "nan",
        "inf",
        "neg_inf",
        "numpy_nan",
        "valid_passthrough",
        "nested_list",
        "nested_tuple",
    ],
)
def test_json_sanitize(value, expected):
    assert Database._json_sanitize(value) == expected
