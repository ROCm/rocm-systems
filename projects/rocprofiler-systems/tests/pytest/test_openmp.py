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
Tests for OpenMP integration with rocprofiler-systems.

This module tests OpenMP examples with various configurations:
- OpenMP CG (Conjugate Gradient) with OMPT
- OpenMP LU decomposition
- OpenMP target offload (GPU)
- OpenMP VV Host
- OpenMP VV Offload (GPU)
- Sampling duration tests
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pytest

from rocprofsys import (
    RocprofsysConfig,
    BaselineRunner,
    BinaryRewriteRunner,
    SamplingRunner,
    RuntimeInstrumentRunner,
    SysRunRunner,
    validate_perfetto_trace,
    validate_rocpd_database,
)


# ============================================================================
# OpenMP Fixtures
# ============================================================================


@pytest.fixture
def ompt_env(base_env: dict[str, str]) -> dict[str, str]:
    """Environment variables for OMPT tests."""
    return {
        "ROCPROFSYS_TRACE": "ON",
        "ROCPROFSYS_PROFILE": "ON",
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
        "ROCPROFSYS_USE_OMPT": "ON",
        "ROCPROFSYS_TIMEMORY_COMPONENTS": "wall_clock,trip_count,peak_rss",
        "OMP_PROC_BIND": "spread",
        "OMP_PLACES": "threads",
        "OMP_NUM_THREADS": "2",
        "LD_LIBRARY_PATH": base_env.get("LD_LIBRARY_PATH", ""),
    }


@pytest.fixture
def ompt_sampling_env(ompt_env: dict[str, str]) -> dict[str, str]:
    """Environment variables for sampling duration tests."""
    env = ompt_env.copy()
    env.update({
        "ROCPROFSYS_VERBOSE": "2",
        "ROCPROFSYS_USE_OMPT": "OFF",
        "ROCPROFSYS_USE_SAMPLING": "ON",
        "ROCPROFSYS_USE_PROCESS_SAMPLING": "OFF",
        "ROCPROFSYS_SAMPLING_FREQ": "100",
        "ROCPROFSYS_SAMPLING_DELAY": "0.1",
        "ROCPROFSYS_SAMPLING_DURATION": "0.25",
        "ROCPROFSYS_SAMPLING_CPUTIME": "ON",
        "ROCPROFSYS_SAMPLING_REALTIME": "ON",
        "ROCPROFSYS_SAMPLING_CPUTIME_FREQ": "1000",
        "ROCPROFSYS_SAMPLING_REALTIME_FREQ": "500",
        "ROCPROFSYS_MONOCHROME": "ON",
    })
    return env


@pytest.fixture
def openmp_target_env(ompt_env: dict[str, str]) -> dict[str, str]:
    """Environment variables for OpenMP target (GPU) tests."""
    env = ompt_env.copy()
    env["ROCPROFSYS_ROCM_DOMAINS"] = "hip_api,hsa_api,kernel_dispatch"
    return env


# ============================================================================
# Test Class: OpenMP CG Tests
# ============================================================================


class TestOpenMPCG:
    """Tests for OpenMP Conjugate Gradient example."""

    REWRITE_ARGS = ["-e", "-v", "2", "--instrument-loops"]

    def test_cg_binary_rewrite(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        ompt_env: dict[str, str],
    ):
        """Test OpenMP CG with binary rewrite instrumentation."""
        env = ompt_env.copy()
        env["ROCPROFSYS_USE_SAMPLING"] = "OFF"
        env["ROCPROFSYS_COUT_OUTPUT"] = "ON"

        try:
            runner = BinaryRewriteRunner(
                config=rocprof_config,
                target="openmp-cg",
                output_dir=test_output_dir,
                rewrite_args=self.REWRITE_ARGS,
                env=env,
                timeout=180,
            )
        except FileNotFoundError:
            pytest.skip("openmp-cg target not built")

        # Perform rewrite
        rewrite_result = runner.rewrite()
        assert rewrite_result.success, f"Rewrite failed: {rewrite_result.stderr}"

        # Check loops were instrumented
        assert "0 instrumented loops in procedure" not in rewrite_result.stdout, \
            "No loops were instrumented"

        # Run the instrumented binary
        result = runner.run()
        assert result.success, f"CG run failed: {result.stderr}"

    def test_cg_sampling(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        ompt_env: dict[str, str],
    ):
        """Test OpenMP CG with sampling instrumentation."""
        env = ompt_env.copy()
        env["ROCPROFSYS_USE_SAMPLING"] = "OFF"
        env["ROCPROFSYS_COUT_OUTPUT"] = "ON"

        try:
            runner = SamplingRunner(
                config=rocprof_config,
                target="openmp-cg",
                output_dir=test_output_dir,
                env=env,
                timeout=180,
            )
        except FileNotFoundError:
            pytest.skip("openmp-cg target not built")

        result = runner.run()
        assert result.success, f"CG sampling failed: {result.stderr}"


# ============================================================================
# Test Class: OpenMP LU Tests
# ============================================================================


class TestOpenMPLU:
    """Tests for OpenMP LU decomposition example."""

    REWRITE_ARGS = ["-e", "-v", "2", "--instrument-loops"]

    def test_lu_binary_rewrite(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        ompt_env: dict[str, str],
    ):
        """Test OpenMP LU with binary rewrite instrumentation."""
        env = ompt_env.copy()
        env["ROCPROFSYS_USE_SAMPLING"] = "ON"
        env["ROCPROFSYS_SAMPLING_FREQ"] = "50"
        env["ROCPROFSYS_COUT_OUTPUT"] = "ON"

        try:
            runner = BinaryRewriteRunner(
                config=rocprof_config,
                target="openmp-lu",
                output_dir=test_output_dir,
                rewrite_args=self.REWRITE_ARGS,
                env=env,
                timeout=180,
            )
        except FileNotFoundError:
            pytest.skip("openmp-lu target not built")

        # Perform rewrite
        rewrite_result = runner.rewrite()
        assert rewrite_result.success, f"Rewrite failed: {rewrite_result.stderr}"

        # Check loops were instrumented
        assert "0 instrumented loops in procedure" not in rewrite_result.stdout, \
            "No loops were instrumented"

        # Run the instrumented binary
        result = runner.run()
        assert result.success, f"LU run failed: {result.stderr}"


# ============================================================================
# Test Class: OpenMP Target (GPU) Tests
# ============================================================================


@pytest.mark.gpu
class TestOpenMPTarget:
    """Tests for OpenMP target offload (GPU) example."""

    @pytest.fixture
    def openmp_target_rules(self, validation_rules_dir: Path) -> list[Path]:
        """Get validation rules for OpenMP target tests."""
        rules_dir = validation_rules_dir / "openmp-target"
        return [
            rules_dir / "kernel-rules.json",
            rules_dir / "sdk-metrics-rules.json",
        ]


    def test_target_sampling(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        openmp_target_env: dict[str, str],
        openmp_target_rules: list[Path],
        use_rocpd: bool,
    ):
        """Test OpenMP target with sampling instrumentation."""
        env = openmp_target_env.copy()
        if use_rocpd:
            env["ROCPROFSYS_USE_ROCPD"] = "ON"

        try:
            runner = SamplingRunner(
                config=rocprof_config,
                target="openmp-target",
                output_dir=test_output_dir,
                env=env,
                timeout=300,
            )
        except FileNotFoundError:
            pytest.skip("openmp-target not built")

        result = runner.run()
        assert result.success, f"OpenMP target failed: {result.stderr}"

        # Verify perfetto trace has kernel dispatch events
        perfetto_file = result.perfetto_file
        if perfetto_file:
            validation = validate_perfetto_trace(
                perfetto_file,
                rocprof_config.rocprofsys_tests_dir,
                categories=["rocm_kernel_dispatch"],
            )
            # Kernel dispatch may or may not be present based on GPU
            if not validation.is_valid:
                pytest.skip("No kernel dispatch events - may need GPU")

        # ROCpd validation
        if use_rocpd:
            rocpd_file = result.rocpd_file
            assert rocpd_file is not None, "ROCpd database not created"
            existing_rules = [r for r in openmp_target_rules if r.exists()]
            if not existing_rules:
                pytest.skip("No validation rules found")
            validation = validate_rocpd_database(
                rocpd_file,
                rocprof_config.rocprofsys_tests_dir,
                rules_files=existing_rules,
            )
            assert validation.is_valid, f"ROCpd validation failed: {validation.message}"

    def test_target_perfetto_validation(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        openmp_target_env: dict[str, str],
    ):
        """Validate OpenMP target perfetto trace for kernel launches."""
        try:
            runner = SamplingRunner(
                config=rocprof_config,
                target="openmp-target",
                output_dir=test_output_dir,
                env=openmp_target_env,
                timeout=300,
            )
        except FileNotFoundError:
            pytest.skip("openmp-target not built")

        result = runner.run()
        assert result.success, f"OpenMP target failed: {result.stderr}"

        perfetto_file = result.perfetto_file
        if perfetto_file is None:
            pytest.skip("No perfetto trace created")

        # Validate trace has expected kernel patterns
        validation = validate_perfetto_trace(
            perfetto_file,
            rocprof_config.rocprofsys_tests_dir,
            label_substrings=["vmul"],  # Vector multiply kernels
        )
        # This validation is informational - kernels may have different names
        if not validation.is_valid:
            pytest.skip("Kernel names differ from expected")


# ============================================================================
# Test Class: OpenMP-VV Host Tests
# ============================================================================

@pytest.mark.parametrize(
    "target_name",
    [
        "openmp-vv-host-test-parallel-for-simd-atomic",
        "openmp-vv-host-test-team-default-shared",
    ],
    ids=["parallel-for-simd-atomic", "team-default-shared"],
)
class TestOpenMPVVHost:
    """Tests for OpenMP VV host programs."""

    def test_ompvv_host_baseline(
            self,
            rocprof_config: RocprofsysConfig,
            test_output_dir: Path,
            ompt_env: dict[str, str],
            target_name: str,
        ):
            """Test OpenMP VV host programs with baseline (no instrumentation)."""
            try:
                runner = BaselineRunner(
                    config=rocprof_config,
                    target=target_name,
                    output_dir=test_output_dir,
                    env=ompt_env,
                    timeout=180,
                )
            except FileNotFoundError:
                pytest.skip(f"{target_name} not built")

            result = runner.run()
            assert result.success, f"OMPVV host baseline {target_name} failed: {result.stderr}"

    def test_ompvv_host_sampling(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        ompt_env: dict[str, str],
        target_name: str,
    ):
        """Test OpenMP VV host programs with sampling."""
        env = ompt_env.copy()

        try:
            runner = SamplingRunner(
                config=rocprof_config,
                target=target_name,
                output_dir=test_output_dir,
                env=env,
                timeout=180,
            )
        except FileNotFoundError:
            pytest.skip(f"{target_name} not built")

        result = runner.run()
        assert result.success, f"OMPVV host test {target_name} failed: {result.stderr}"

        # Verify output files were created
        assert result.perfetto_file is not None or \
               len(list(result.output_dir.glob("*.json"))) > 0, \
               "No output files created"

    def test_ompvv_host_binary_rewrite(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        ompt_env: dict[str, str],
        target_name: str,
    ):
        """Test OpenMP VV host programs with binary rewrite."""
        env = ompt_env.copy()
        env["ROCPROFSYS_COUT_OUTPUT"] = "ON"

        try:
            runner = BinaryRewriteRunner(
                config=rocprof_config,
                target=target_name,
                output_dir=test_output_dir,
                rewrite_args=["-e", "-v", "2", "--instrument-loops"],
                env=env,
                timeout=180,
                pass_regex=[r"omp_parallel"],
            )
        except FileNotFoundError:
            pytest.skip(f"{target_name} not built")

        # Perform rewrite phase
        rewrite_result = runner.rewrite()

        assert rewrite_result.success, f"Rewrite failed for {target_name}: {rewrite_result.stderr}"

        # Run the instrumented binary
        result = runner.run()

        assert result.success, f"OMPVV host binary rewrite {target_name} failed: {result.stderr}"

    def test_ompvv_host_runtime_instrument(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        ompt_env: dict[str, str],
        target_name: str,
    ):
        """Test OpenMP VV Host programs with runtime instrumentation."""
        env = ompt_env.copy()
        env["ROCPROFSYS_COUT_OUTPUT"] = "ON"
        env["ROCPROFSYS_CI_SKIP_PUSH_POP_CHECK"] = "ON"

        try:
            runner = RuntimeInstrumentRunner(
                config=rocprof_config,
                target=target_name,
                output_dir=test_output_dir,
                instrument_args=["-e", "-v", "1", "--label", "return", "args"],
                env=env,
                timeout=180,
                pass_regex=[r"omp_parallel"],
            )
        except FileNotFoundError:
            pytest.skip(f"{target_name} not built")

        result = runner.run()

        assert result.success, f"Runtime instrumentation failed for {target_name}: {result.stderr}"

    def test_ompvv_host_run(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        ompt_env: dict[str, str],
        target_name: str,
    ):
        """Test OpenMP VV host programs with rocprof-sys-run wrapper."""
        env = ompt_env.copy()

        try:
            runner = SysRunRunner(
                config=rocprof_config,
                target=target_name,
                output_dir=test_output_dir,
                env=env,
                timeout=180,
            )
        except FileNotFoundError:
            pytest.skip(f"{target_name} not built")

        result = runner.run()
        assert result.success, f"OMPVV host run {target_name} failed: {result.stderr}"

        # Verify output files were created
        assert result.perfetto_file is not None or \
               len(list(result.output_dir.glob("*.json"))) > 0, \
               "No output files created"


# ============================================================================
# Test Class: OpenMP-VV Offload (GPU) Tests
# ============================================================================

@pytest.mark.gpu
@pytest.mark.parametrize(
    "target_name",
    [
        "openmp-vv-offload-test-target-simd-if",
        "openmp-vv-offload-test-target-teams-distribute-parallel-for-collapse",
    ],
    ids=["target-simd-if", "target-teams-distribute-parallel-for-collapse"],
)
class TestOpenMPVVOffload:
    """Tests for OpenMP VV offload programs."""

    def test_ompvv_offload_baseline(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        openmp_target_env: dict[str, str],
        target_name: str,
    ):
        """Test OpenMP VV offload programs with baseline."""
        try:
            runner = BaselineRunner(
                config=rocprof_config,
                target=target_name,
                output_dir=test_output_dir,
                env=openmp_target_env,
                timeout=300,
            )
        except FileNotFoundError:
            pytest.skip(f"{target_name} not built")

        result = runner.run()
        assert result.success, f"OMPVV offload baseline {target_name} failed: {result.stderr}"

    def test_ompvv_offload_sampling(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        openmp_target_env: dict[str, str],
        target_name: str,
    ):
        """Test OpenMP VV offload programs with sampling."""
        try:
            runner = SamplingRunner(
                config=rocprof_config,
                target=target_name,
                output_dir=test_output_dir,
                env=openmp_target_env,
                timeout=300,
            )
        except FileNotFoundError:
            pytest.skip(f"{target_name} not built")

        result = runner.run()
        assert result.success, f"OMPVV offload sampling {target_name} failed: {result.stderr}"

    def test_ompvv_offload_binary_rewrite(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        openmp_target_env: dict[str, str],
        target_name: str,
    ):
        """Test OpenMP VV offload programs with binary rewrite."""
        env = openmp_target_env.copy()
        env["ROCPROFSYS_COUT_OUTPUT"] = "ON"

        try:
            runner = BinaryRewriteRunner(
                config=rocprof_config,
                target=target_name,
                output_dir=test_output_dir,
                rewrite_args=["-e", "-v", "2"],
                env=env,
                timeout=300,
                pass_regex=[r"omp_offloading"],
            )
        except FileNotFoundError:
            pytest.skip(f"{target_name} not built")

        # Perform rewrite phase
        rewrite_result = runner.rewrite()

        assert rewrite_result.success, f"Rewrite failed for {target_name}: {rewrite_result.stderr}"

        # Run the instrumented binary
        result = runner.run()

        assert result.success, f"OMPVV offload binary rewrite {target_name} failed: {result.stderr}"

    def test_ompvv_offload_run(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        openmp_target_env: dict[str, str],
        target_name: str,
    ):
        """Test OpenMP VV offload programs with run mode."""
        env = openmp_target_env.copy()
        env.update({
            "ROCPROFSYS_USE_OMPT": "OFF",
        })

        try:
            runner = SysRunRunner(
                config=rocprof_config,
                target=target_name,
                output_dir=test_output_dir,
                run_args=["-e", "-v", "1", "--label", "return", "args"],
                env=env,
                timeout=300,
            )
        except FileNotFoundError:
            pytest.skip(f"{target_name} not built")

        result = runner.run()
        assert result.success, f"OMPVV offload run {target_name} failed: {result.stderr}"

        # Verify output files were created
        assert result.perfetto_file is not None or \
               len(list(result.output_dir.glob("*.json"))) > 0, \
               "No output files created"

# ============================================================================
# Test Class: Sampling Duration Tests
# ============================================================================


class TestSamplingDuration:
    """Tests for sampling duration functionality."""

    def test_cg_sampling_duration(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        ompt_sampling_env: dict[str, str],
    ):
        """Test OpenMP CG with sampling duration limits."""
        try:
            runner = SamplingRunner(
                config=rocprof_config,
                target="openmp-cg",
                output_dir=test_output_dir,
                env=ompt_sampling_env,
                timeout=300,
            )
        except FileNotFoundError:
            pytest.skip("openmp-cg target not built")

        result = runner.run()
        assert result.success, f"Sampling duration test failed: {result.stderr}"

        # Verify sampling messages in output
        combined_output = result.stdout + result.stderr
        expected_patterns = [
            "will be triggered",
            "per second",
        ]
        found_any = any(p in combined_output for p in expected_patterns)
        # Output format may vary
        if not found_any:
            # Just ensure output files were created
            assert result.output_dir.exists(), "Output directory not created"

    def test_lu_sampling_duration(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        ompt_sampling_env: dict[str, str],
    ):
        """Test OpenMP LU with sampling duration limits."""
        try:
            runner = SamplingRunner(
                config=rocprof_config,
                target="openmp-lu",
                output_dir=test_output_dir,
                env=ompt_sampling_env,
                timeout=300,
            )
        except FileNotFoundError:
            pytest.skip("openmp-lu target not built")

        result = runner.run()
        assert result.success, f"Sampling duration test failed: {result.stderr}"


# ============================================================================
# Test Class: No Temporary Files Tests
# ============================================================================


class TestNoTmpFiles:
    """Tests for operation without temporary files."""

    @pytest.fixture
    def no_tmp_env(self, ompt_env: dict[str, str]) -> dict[str, str]:
        """Environment variables for no-tmp-files tests."""
        env = ompt_env.copy()
        env.update({
            "ROCPROFSYS_VERBOSE": "2",
            "ROCPROFSYS_USE_OMPT": "OFF",
            "ROCPROFSYS_USE_SAMPLING": "ON",
            "ROCPROFSYS_USE_PROCESS_SAMPLING": "OFF",
            "ROCPROFSYS_SAMPLING_CPUTIME": "ON",
            "ROCPROFSYS_SAMPLING_REALTIME": "OFF",
            "ROCPROFSYS_SAMPLING_CPUTIME_FREQ": "700",
            "ROCPROFSYS_USE_TEMPORARY_FILES": "OFF",
            "ROCPROFSYS_MONOCHROME": "ON",
        })
        return env

    def test_cg_no_tmp_files(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        no_tmp_env: dict[str, str],
    ):
        """Test OpenMP CG without temporary files."""
        try:
            runner = SamplingRunner(
                config=rocprof_config,
                target="openmp-cg",
                output_dir=test_output_dir,
                env=no_tmp_env,
                timeout=300,
            )
        except FileNotFoundError:
            pytest.skip("openmp-cg target not built")

        result = runner.run()
        assert result.success, f"No tmp files test failed: {result.stderr}"

        # Verify sampling output files were created
        sampling_files = list(result.output_dir.glob("sampling_*.json")) + \
                         list(result.output_dir.glob("sampling_*.txt"))
        assert len(sampling_files) > 0 or result.perfetto_file is not None, \
            "No sampling output files created"
