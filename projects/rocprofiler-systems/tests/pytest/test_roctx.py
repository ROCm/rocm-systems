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
Tests for the ROCTX marker API integration with rocprofiler-systems.
Equivalent to rocprof-sys-roctx-tests.cmake
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest

from rocprofsys import (
    RocprofsysConfig,
    BaselineRunner,
    SamplingRunner,
    BinaryRewriteRunner,
    SysRunRunner,
    validate_rocpd_database,
)


# =============================================================================
# rocTX fixtures
# =============================================================================


@pytest.fixture
def roctx_env(base_env: dict[str, str]) -> dict[str, str]:
    """Environment variables for rocTX tests."""
    return {
        "ROCPROFSYS_ROCM_DOMAINS": "hip_runtime_api,marker_api,kernel_dispatch",
    }


# ============================================================================
# Test Class: rocTX Tests
# ============================================================================


@pytest.mark.gpu
class TestRoctx:
    """Tests for rocTX marker API."""

    @pytest.fixture
    def roctx_rules(self, validation_rules_dir: Path) -> list[Path]:
        """Get validation rules for rocTX tests."""
        rules_dir = validation_rules_dir / "roctx"
        return [
            rules_dir / "validation-rules.json",
            rules_dir / "amd-smi-rules.json",
            rules_dir / "sdk-metrics-rules.json",
        ]

    REWRITE_ARGS = ["-e", "-v", "2", "--instrument-loops"]

    def test_baseline(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        roctx_env: dict[str, str],
        collect_result,
    ):
        try:
            runner = BaselineRunner(
                config=rocprof_config,
                target="roctx",
                output_dir=test_output_dir,
                env=roctx_env,
                timeout=120,
            )
        except FileNotFoundError:
            pytest.skip("roctx target not built")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"rocTX baseline failed: {result.test_output}")

    def test_sampling(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        roctx_env: dict[str, str],
        roctx_rules: list[Path],
        use_rocpd: bool,
        subtests,
        collect_result,
    ):
        env = roctx_env.copy()
        if use_rocpd:
            env["ROCPROFSYS_USE_ROCPD"] = "ON"
        try:
            runner = SamplingRunner(
                config=rocprof_config,
                target="roctx",
                output_dir=test_output_dir,
                env=env,
                timeout=120,
            )
        except FileNotFoundError:
            pytest.skip("roctx target not built")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"rocTX sampling failed: {result.test_output}")

        # ROCpd validation
        with subtests.test("ROCpd validation"):
            if not use_rocpd:
                pytest.skip("ROCpd is not enabled")
            rocpd_file = result.rocpd_file
            if rocpd_file is None:
                pytest.fail(f"ROCpd database not created")
            existing_rules = [r for r in roctx_rules if r.exists()]
            if not existing_rules:
                pytest.skip("No validation rules found")
            validation = validate_rocpd_database(
                rocpd_file,
                rocprof_config.rocprofsys_tests_dir,
                rules_files=existing_rules,
            )
            if not validation.is_valid:
                pytest.fail(f"ROCpd validation failed: {validation.message}")

    def test_binary_rewrite(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        roctx_env: dict[str, str],
        collect_result,
    ):
        try:
            runner = BinaryRewriteRunner(
                config=rocprof_config,
                target="roctx",
                output_dir=test_output_dir,
                rewrite_args=self.REWRITE_ARGS,
                env=roctx_env,
                timeout=120,
            )
        except FileNotFoundError:
            pytest.skip("roctx target not built")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"rocTX binary rewrite test failed: {result.test_output}")

    def test_sys_run(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        roctx_env: dict[str, str],
        collect_result,
    ):
        try:
            runner = SysRunRunner(
                config=rocprof_config,
                target="roctx",
                output_dir=test_output_dir,
                env=roctx_env,
                timeout=120,
            )
        except FileNotFoundError:
            pytest.skip("roctx target not built")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"rocTX sys run failed: {result.test_output}")
