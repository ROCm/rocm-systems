# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for ML API trace analysis error messages."""

import pytest

from ml_api_tracing.analyze.errors import UnaccountedKernelError


def test_unaccounted_kernel_error_lists_the_first_ten_kernel_names():
    kernel_names = ["z", "a", "k8", "k7", "k6", "k5", "k4", "k3", "k2", "k1", "k0"]
    error = UnaccountedKernelError(12, kernel_names)
    assert str(error) == (
        "12 unmatched dispatches; first 10 Kernel_Names: "
        "z, a, k8, k7, k6, k5, k4, k3, k2, k1"
    )


@pytest.mark.parametrize(
    ("unmatched_count", "kernel_names", "expected_message"),
    [
        pytest.param(0, [], "0 unmatched dispatches", id="empty"),
        pytest.param(2, [], "2 unmatched dispatches", id="no-kernel-names"),
        pytest.param(2, [""], "2 unmatched dispatches", id="empty-kernel-name"),
        pytest.param(
            10,
            list("abcdefghij"),
            "10 unmatched dispatches; Kernel_Names: a, b, c, d, e, f, g, h, i, j",
            id="exact-preview-limit",
        ),
    ],
)
def test_unaccounted_kernel_error_message(
    unmatched_count, kernel_names, expected_message
):
    assert (
        str(UnaccountedKernelError(unmatched_count, kernel_names)) == expected_message
    )
