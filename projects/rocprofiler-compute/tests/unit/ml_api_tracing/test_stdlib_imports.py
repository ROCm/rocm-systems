# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Import checks for the stdlib-only ML API tracing modules."""

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "ml_api_tracing.constants",
        "ml_api_tracing.marker_format",
        "ml_api_tracing.profiling.trace_selection",
    ],
)
def test_module_imports_fresh_with_only_stdlib_under_profile_guard(
    assert_stdlib_import, module_name
):
    assert_stdlib_import(module_name)
