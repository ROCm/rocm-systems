# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for ML API trace analysis error messages."""

from ml_api_tracing.analyze.errors import UnaccountedKernelError


def test_unaccounted_kernel_error_lists_the_first_ten_kernel_names():
    kernel_names = ["z", "a", "k8", "k7", "k6", "k5", "k4", "k3", "k2", "k1", "k0"]
    error = UnaccountedKernelError(12, kernel_names)
    assert str(error) == (
        "12 unmatched dispatches; first 10 Kernel_Names: "
        "z, a, k8, k7, k6, k5, k4, k3, k2, k1"
    )
