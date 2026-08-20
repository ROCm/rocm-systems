# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for ``normalize_kernel_name`` and ``normalize_kernel_names``."""

import pytest

pytest.importorskip("torch")
pytest.importorskip("pandas")

from torch_trace_coverage_utils import (  # noqa: E402
    normalize_kernel_name,
    normalize_kernel_names,
)


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
    """Trailing ``.kd`` and ``[clone ...]`` suffixes are stripped."""
    assert normalize_kernel_name(raw) == expected


def test_normalize_kernel_names():
    """Each name in a set is normalized."""
    assert normalize_kernel_names({"foo.kd", "bar [clone .kd]", "baz"}) == {
        "foo",
        "bar",
        "baz",
    }
