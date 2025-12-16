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
from pathlib import Path

from rocprofsys import (
    RocprofsysConfig,
    SamplingRunner,
    SysRunRunner,
)

# =============================================================================
# HIP stream tests
# =============================================================================

@pytest.mark.gpu
@pytest.mark.rocm_min_version("7.0")
class TestTransposeGroupByQueue:
    """Tests for transpose with group by queue"""

    def test_sampling(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        base_env: dict[str, str],
    ):
        """Test transpose with group by queue"""
        env = base_env.copy()
        env["ROCPROFSYS_ROCM_GROUP_BY_QUEUE"] = "YES"
        runner = SamplingRunner(
            config=rocprof_config,
            target="transpose",
            output_dir=test_output_dir,
            env=env,
            timeout=120,
        )
        result = runner.run()
        assert result.success, f"Sampling failed: {result.stderr}"

    def test_sys_run(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        base_env: dict[str, str],
    ):
        """Test transpose with group by queue"""
        env = base_env.copy()
        env["ROCPROFSYS_ROCM_GROUP_BY_QUEUE"] = "YES"
        runner = SysRunRunner(
            config=rocprof_config,
            target="transpose",
            output_dir=test_output_dir,
            env=env,
            timeout=120,
        )
        result = runner.run()
        assert result.success, f"Sys run failed: {result.stderr}"

@pytest.mark.gpu
@pytest.mark.rocm_min_version("7.0")
class TestTransposeGroupByStream:
    """Tests for transpose with group by stream"""

    def test_sampling(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        base_env: dict[str, str],
    ):
        """Test transpose with group by stream"""
        env = base_env.copy()
        env["ROCPROFSYS_ROCM_GROUP_BY_QUEUE"] = "NO"
        runner = SamplingRunner(
            config=rocprof_config,
            target="transpose",
            output_dir=test_output_dir,
            env=env,
            timeout=120,
        )
        result = runner.run()
        assert result.success, f"Sampling failed: {result.stderr}"

    def test_sys_run(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        base_env: dict[str, str],
    ):
        """Test transpose with group by stream"""
        env = base_env.copy()
        env["ROCPROFSYS_ROCM_GROUP_BY_QUEUE"] = "NO"
        runner = SysRunRunner(
            config=rocprof_config,
            target="transpose",
            output_dir=test_output_dir,
            env=env,
            timeout=120,
        )
        result = runner.run()
        assert result.success, f"Sys run failed: {result.stderr}"
