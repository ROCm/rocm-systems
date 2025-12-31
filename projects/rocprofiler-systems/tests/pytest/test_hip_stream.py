# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
Tests for HIP stream API
"""

import pytest

# =============================================================================
# HIP stream tests
# =============================================================================


@pytest.mark.gpu
@pytest.mark.rocm_min_version("7.0")
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
