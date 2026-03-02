# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Tests for HIP stream API
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.rocm_min_version("7.0"),
    pytest.mark.hip_stream,
    pytest.mark.ci_enable,
]


# =============================================================================
# HIP stream tests
# =============================================================================


@pytest.mark.parametrize("mode", ["sampling", "sys_run"])
@pytest.mark.parametrize(
    "type",
    [
        pytest.param("group-by-queue", marks=pytest.mark.group_by_queue),
        pytest.param("group-by-stream", marks=pytest.mark.group_by_stream),
    ],
)
class TestHipStream(RocprofsysTest):
    def test_transpose(self, mode, type, num_processes):
        if type == "group-by-queue":
            env = {"ROCPROFSYS_ROCM_GROUP_BY_QUEUE": "YES"}
        else:
            env = {"ROCPROFSYS_ROCM_GROUP_BY_QUEUE": "NO"}

        result = self.run_test(
            mode,
            "transpose",
            env=env,
            check_target_arch=True,
            timeout=120,
            mpi_ranks=num_processes,
        )
        self.assert_regex(result)
