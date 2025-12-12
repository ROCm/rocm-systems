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
Tests for ROCTX marker API integration with rocprofiler-systems.

Equivalent to rocprof-sys-roctx-tests.cmake
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest

from rocprofsys import (
    RocprofsysConfig,
    BaselineRunner,
    SamplingRunner,
    BinaryRewriteRunner,
    RuntimeInstrumentRunner,
    SysRunRunner,
    validate_perfetto_trace,
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
    REWRITE_ARGS = ["-e", "-v", "2", "--instrument-loops"]

    def test_roctx_baseline(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        roctx_env: dict[str, str],
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
        assert result.success, f"ROCTX baseline failed: {result.stderr}"

    def test_roctx_sampling(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        roctx_env: dict[str, str],
    ):
        try:
            runner = SamplingRunner(
                config=rocprof_config,
                target="roctx",
                output_dir=test_output_dir,
                env=roctx_env,
                timeout=120,
            )

        except FileNotFoundError:
            pytest.skip("roctx target not built")
        result = runner.run()
        assert result.success, f"ROCTX sampling failed: {result.stderr}"

    def test_roctx_binary_rewrite(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        roctx_env: dict[str, str],
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

        rewrite_result = runner.rewrite()
        assert rewrite_result.success, f"Rewrite failed: {rewrite_result.stderr}"

    def test_roctx_sys_run(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        roctx_env: dict[str, str],
    ):
        try:
            runner = SysRunRunner(
                config=rocprof_config,
                target="roctx",
                output_dir=test_output_dir,
                env=roctx_env,
            )
        except FileNotFoundError:
            pytest.skip("roctx target not built")

        result = runner.run()
        assert result.success, f"ROCTX sys run failed: {result.stderr}"


# ============================================================================
# Test Class: rocTX ROCpd Tests
# ============================================================================

@pytest.mark.gpu
@pytest.mark.rocpd
class TestRoctxROCpd:
    """Tests for Roctx with ROCpd database output."""

    @pytest.fixture
    def roctx_rules(self, validation_rules_dir: Path) -> list[Path]:
        """Get validation rules for roctx tests."""
        rules_dir = validation_rules_dir / "roctx"
        return [
            rules_dir / "validation-rules.json",
            rules_dir / "amd-smi-rules.json",
            rules_dir / "sdk-metrics-rules.json",
        ]

    def test_validate_rocpd(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        roctx_env: dict[str, str],
        roctx_rules: list[Path],
    ):
        """Validate roctx ROCpd database."""
        env = roctx_env.copy()
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
        assert result.success, f"ROCTX ROCpd validation failed: {result.stderr}"

        rocpd_file = result.rocpd_file
        assert rocpd_file is not None, "ROCpd database not created"

        existing_rules = [r for r in roctx_rules if r.exists()]
        if not existing_rules:
            pytest.skip("No validation rules found")

        validation = validate_rocpd_database(
            rocpd_file,
            rocprof_config.rocprofsys_tests_dir,
            rules_files=existing_rules,
        )
        assert validation.is_valid, f"ROCpd validation failed: {validation.message}"
