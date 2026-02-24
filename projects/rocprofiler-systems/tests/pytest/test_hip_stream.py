# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Tests for HIP stream API
"""

from __future__ import annotations
import pytest

# =============================================================================
# HIP stream tests
# =============================================================================


@pytest.mark.gpu
@pytest.mark.rocm_min_version("7.0")
@pytest.mark.group_by_queue
class TestTransposeGroupByQueue:
    """Tests for transpose with group by queue"""

    def test_sampling(
        self,
        run_test,
        base_env: dict[str, str],
        assert_regex,
    ):
        env = base_env.copy()
        env["ROCPROFSYS_ROCM_GROUP_BY_QUEUE"] = "YES"
        result = run_test(
            "sampling",
            target="transpose",
            env=env,
            timeout=120,
        )

        assert_regex(result)

    def test_sys_run(
        self,
        run_test,
        base_env: dict[str, str],
        assert_regex,
    ):
        env = base_env.copy()
        env["ROCPROFSYS_ROCM_GROUP_BY_QUEUE"] = "YES"

        result = run_test(
            "sys_run",
            target="transpose",
            env=env,
            timeout=120,
        )

        assert_regex(result)


@pytest.mark.gpu
@pytest.mark.rocm_min_version("7.0")
@pytest.mark.group_by_stream
class TestTransposeGroupByStream:
    def test_sampling(
        self,
        run_test,
        base_env: dict[str, str],
        assert_regex,
    ):
        env = base_env.copy()
        env["ROCPROFSYS_ROCM_GROUP_BY_QUEUE"] = "NO"

        result = run_test(
            "sampling",
            target="transpose",
            env=env,
            timeout=120,
        )

        assert_regex(result)

    def test_sys_run(
        self,
        run_test,
        base_env: dict[str, str],
        assert_regex,
    ):
        env = base_env.copy()
        env["ROCPROFSYS_ROCM_GROUP_BY_QUEUE"] = "NO"

        result = run_test(
            "sys_run",
            target="transpose",
            env=env,
            timeout=120,
        )

        assert_regex(result)
