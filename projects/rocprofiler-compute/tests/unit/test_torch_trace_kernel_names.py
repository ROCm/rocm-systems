# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for ``torch_trace_kernel_names``."""

import pytest

from torch_trace_kernel_names import normalize_kernel_name


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("foo.kd", "foo"),
        ("foo [clone .kd]", "foo"),
        ("foo.kd [clone .kd]", "foo"),
        ("foo [clone .kd] [clone .kd]", "foo"),
        ("foo [clone .kd].kd", "foo"),
        ("foo.kdbar", "foo.kdbar"),
        ("foo", "foo"),
    ],
)
def test_normalize_kernel_name(raw, expected):
    assert normalize_kernel_name(raw) == expected
