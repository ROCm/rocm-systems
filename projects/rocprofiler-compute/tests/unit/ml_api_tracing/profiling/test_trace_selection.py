# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for ML API trace selection."""

import pytest

from ml_api_tracing.constants import KNOWN_ML_API_BACKENDS
from ml_api_tracing.profiling.trace_selection import select_traced_frameworks


@pytest.mark.parametrize(
    "raw, expected",
    [
        pytest.param(None, None, id="absent"),
        pytest.param([], list(KNOWN_ML_API_BACKENDS), id="bare"),
        pytest.param(
            ["triton", "torch", "triton"],
            ["triton", "torch"],
            id="ordered-unique",
        ),
    ],
)
def test_select_traced_frameworks(raw, expected):
    assert select_traced_frameworks(raw) == expected
