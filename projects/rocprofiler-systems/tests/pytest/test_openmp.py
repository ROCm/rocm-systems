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

Note: OMPT backend is unavailable and tests are skipped if no GPU is available.
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
    env.update(
        {
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
        }
    )
    return env


@pytest.fixture
def openmp_target_env(ompt_env: dict[str, str]) -> dict[str, str]:
    """Environment variables for OpenMP target (GPU) tests."""
    env = ompt_env.copy()
    env["ROCPROFSYS_ROCM_DOMAINS"] = "hip_api,hsa_api,kernel_dispatch"
    return env


@pytest.fixture
def ompt_no_tmp_env(ompt_env: dict[str, str]) -> dict[str, str]:
    """Environment variables for no-tmp-files tests."""
    env = ompt_env.copy()
    env.update(
        {
            "ROCPROFSYS_VERBOSE": "2",
            "ROCPROFSYS_USE_OMPT": "OFF",
            "ROCPROFSYS_USE_SAMPLING": "ON",
            "ROCPROFSYS_USE_PROCESS_SAMPLING": "OFF",
            "ROCPROFSYS_SAMPLING_CPUTIME": "ON",
            "ROCPROFSYS_SAMPLING_REALTIME": "OFF",
            "ROCPROFSYS_SAMPLING_CPUTIME_FREQ": "700",
            "ROCPROFSYS_USE_TEMPORARY_FILES": "OFF",
            "ROCPROFSYS_MONOCHROME": "ON",
        }
    )
    return env


@pytest.fixture
def openmp_target_rules(validation_rules_dir: Path) -> list[Path]:
    """Get validation rules for OpenMP target tests."""
    rules_dir = validation_rules_dir / "openmp-target"
    return [
        rules_dir / "kernel-rules.json",
        rules_dir / "sdk-metrics-rules.json",
    ]


# ============================================================================
# Test Class: OpenMP CG Tests
# ============================================================================


@pytest.mark.gpu
class TestOpenMPCG:
    """Tests for OpenMP Conjugate Gradient example."""

    REWRITE_ARGS = ["-e", "-v", "2", "--instrument-loops"]

    def test_sampling(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        ompt_env: dict[str, str],
        collect_result,
    ):
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
            pytest.skip("openmp-cg binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"CG sampling failed: {result.test_output}")

    def test_binary_rewrite(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        ompt_env: dict[str, str],
        collect_result,
    ):
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
            pytest.skip("openmp-cg binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"CG binary rewrite test failed: {result.test_output}")


# ============================================================================
# Test Class: OpenMP LU Tests
# ============================================================================


@pytest.mark.gpu
class TestOpenMPLU:
    """Tests for OpenMP LU decomposition example."""

    REWRITE_ARGS = ["-e", "-v", "2", "--instrument-loops"]

    def test_binary_rewrite(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        ompt_env: dict[str, str],
        collect_result,
    ):
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
            pytest.skip("openmp-lu binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"LU binary rewrite test failed: {result.test_output}")


# ============================================================================
# Test Class: OpenMP Target (GPU) Tests
# ============================================================================


@pytest.mark.gpu
class TestOpenMPTarget:
    """Tests for OpenMP target offload (GPU) example."""

    def test_sampling(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        openmp_target_env: dict[str, str],
        openmp_target_rules: list[Path],
        use_rocpd: bool,
        use_perfetto: bool,
        subtests,
        collect_result,
    ):
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
            pytest.skip("openmp-target binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"OpenMP target sampling failed: {result.test_output}")

        # Verify perfetto trace has kernel dispatch events
        with subtests.test("Perfetto validation"):
            if not use_perfetto:
                pytest.skip("Perfetto is not enabled")
            perfetto_file = result.perfetto_file
            if not perfetto_file:
                pytest.fail(f"Perfetto trace not created")
            validation = validate_perfetto_trace(
                perfetto_file,
                rocprof_config.rocprofsys_tests_dir,
                categories=["rocm_kernel_dispatch"],
            )
            # Kernel dispatch may or may not be present based on GPU
            if not validation.is_valid:
                pytest.skip("No kernel dispatch events")

        with subtests.test("Perfetto Kernel Dispatch Validation"):
            if not use_perfetto:
                pytest.skip("Perfetto is not enabled")
            perfetto_file = result.perfetto_file
            if not perfetto_file:
                pytest.fail(f"Perfetto trace not created")
            # Validate trace has expected kernel patterns
            validation = validate_perfetto_trace(
                perfetto_file,
                rocprof_config.rocprofsys_tests_dir,
                label_substrings=["vmul"],  # Vector multiply kernels
            )
            # This validation is informational - kernels may have different names
            if not validation.is_valid:
                pytest.skip("Kernel names differ from expected")

        # ROCpd validation
        with subtests.test("ROCpd validation"):
            if not use_rocpd:
                pytest.skip("ROCpd is not enabled")
            rocpd_file = result.rocpd_file
            if rocpd_file is None:
                pytest.fail(f"ROCpd database not created")
            existing_rules = [r for r in openmp_target_rules if r.exists()]
            if not existing_rules:
                pytest.skip("No validation rules found")
            validation = validate_rocpd_database(
                rocpd_file,
                rocprof_config.rocprofsys_tests_dir,
                rules_files=existing_rules,
            )
            if not validation.is_valid:
                pytest.fail(f"ROCpd validation failed: {validation.message}")


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
@pytest.mark.gpu
class TestOpenMPVVHost:
    """Tests for OpenMP VV host programs."""

    def test_baseline(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        ompt_env: dict[str, str],
        target_name: str,
        collect_result,
    ):
        try:
            runner = BaselineRunner(
                config=rocprof_config,
                target=target_name,
                output_dir=test_output_dir,
                env=ompt_env,
                timeout=180,
            )
        except FileNotFoundError:
            pytest.skip(f"{target_name} binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"OMPVV host baseline {target_name} failed: {result.test_output}")

    def test_sampling(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        ompt_env: dict[str, str],
        target_name: str,
        use_perfetto: bool,
        subtests,
        collect_result,
    ):
        try:
            runner = SamplingRunner(
                config=rocprof_config,
                target=target_name,
                output_dir=test_output_dir,
                env=ompt_env,
                timeout=180,
            )
        except FileNotFoundError:
            pytest.skip(f"{target_name} binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"OMPVV host test {target_name} failed: {result.test_output}")

        with subtests.test("Perfetto validation"):
            if not use_perfetto:
                pytest.skip("Perfetto is not enabled")
            perfetto_file = result.perfetto_file
            if not perfetto_file:
                pytest.fail(f"Perfetto trace not created")
            validation = validate_perfetto_trace(
                perfetto_file,
                rocprof_config.rocprofsys_tests_dir,
            )
            if not validation.is_valid:
                pytest.fail(f"Perfetto validation failed: {validation.message}")

    def test_binary_rewrite(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        ompt_env: dict[str, str],
        target_name: str,
        collect_result,
    ):
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

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"Binary rewrite test failed: {result.test_output}")

    def test_runtime_instrument(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        ompt_env: dict[str, str],
        target_name: str,
        collect_result,
    ):
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
            pytest.skip(f"{target_name} binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(
                f"Runtime instrumentation failed for {target_name}: {result.test_output}"
            )

    def test_sys_run(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        ompt_env: dict[str, str],
        target_name: str,
        use_perfetto: bool,
        subtests,
        collect_result,
    ):
        try:
            runner = SysRunRunner(
                config=rocprof_config,
                target=target_name,
                output_dir=test_output_dir,
                env=ompt_env,
                timeout=180,
            )
        except FileNotFoundError:
            pytest.skip(f"{target_name} not built")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"OMPVV host run {target_name} failed: {result.test_output}")

        with subtests.test("Perfetto validation"):
            if not use_perfetto:
                pytest.skip("Perfetto is not enabled")
            perfetto_file = result.perfetto_file
            if not perfetto_file:
                pytest.fail(f"Perfetto trace not created")
            validation = validate_perfetto_trace(
                perfetto_file,
                rocprof_config.rocprofsys_tests_dir,
            )
            if not validation.is_valid:
                pytest.fail(f"Perfetto validation failed: {validation.message}")


# ============================================================================
# Test Class: OpenMP-VV Offload (GPU) Tests
# ============================================================================


@pytest.mark.parametrize(
    "target_name",
    [
        "openmp-vv-offload-test-target-simd-if",
        "openmp-vv-offload-test-target-teams-distribute-parallel-for-collapse",
    ],
    ids=["target-simd-if", "target-teams-distribute-parallel-for-collapse"],
)
@pytest.mark.gpu
class TestOpenMPVVOffload:
    """Tests for OpenMP VV offload programs."""

    def test_baseline(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        openmp_target_env: dict[str, str],
        target_name: str,
        collect_result,
    ):
        try:
            runner = BaselineRunner(
                config=rocprof_config,
                target=target_name,
                output_dir=test_output_dir,
                env=openmp_target_env,
                timeout=300,
            )
        except FileNotFoundError:
            pytest.skip(f"{target_name} binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(
                f"OMPVV offload baseline {target_name} failed: {result.test_output}"
            )

    def test_sampling(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        openmp_target_env: dict[str, str],
        target_name: str,
        collect_result,
    ):
        try:
            runner = SamplingRunner(
                config=rocprof_config,
                target=target_name,
                output_dir=test_output_dir,
                env=openmp_target_env,
                timeout=300,
            )
        except FileNotFoundError:
            pytest.skip(f"{target_name} binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(
                f"OMPVV offload sampling {target_name} failed: {result.test_output}"
            )

    def test_binary_rewrite(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        openmp_target_env: dict[str, str],
        target_name: str,
        collect_result,
    ):
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
            pytest.skip(f"{target_name} binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"Binary rewrite test failed: {result.test_output}")

    def test_sys_run(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        openmp_target_env: dict[str, str],
        target_name: str,
        use_perfetto: bool,
        subtests,
        collect_result,
    ):
        try:
            runner = SysRunRunner(
                config=rocprof_config,
                target=target_name,
                output_dir=test_output_dir,
                run_args=["-e", "-v", "1", "--label", "return", "args"],
                env=openmp_target_env,
                timeout=300,
            )
        except FileNotFoundError:
            pytest.skip(f"{target_name} binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"OMPVV offload run {target_name} failed: {result.test_output}")

        with subtests.test("Perfetto validation"):
            if not use_perfetto:
                pytest.skip("Perfetto is not enabled")
            perfetto_file = result.perfetto_file
            if not perfetto_file:
                pytest.fail(f"Perfetto trace not created")
            validation = validate_perfetto_trace(
                perfetto_file,
                rocprof_config.rocprofsys_tests_dir,
            )
            if not validation.is_valid:
                pytest.fail(f"Perfetto validation failed: {validation.message}")


# ============================================================================
# Test Class: Sampling Duration Tests
# ============================================================================


@pytest.mark.gpu
class TestSamplingDuration:
    """Tests for sampling duration functionality."""

    # Regex patterns from CMake _ompt_sampling_samp_regex and _ompt_sampling_file_regex
    SAMPLING_PASS_REGEX = [
        r"Sampler for thread 0 will be triggered 1000\.0x per second of CPU-time",
        r"Sampler for thread 0 will be triggered 500\.0x per second of wall-time",
        r"Sampling will be disabled after 0\.250000 seconds",
        r"Sampling duration of 0\.250000 seconds has elapsed\. Shutting down sampling",
        r"sampling_percent\.(json|txt)",
        r"sampling_cpu_clock\.(json|txt)",
        r"sampling_wall_clock\.(json|txt)",
    ]

    def test_cg_sampling_duration(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        ompt_sampling_env: dict[str, str],
        collect_result,
    ):
        try:
            runner = SamplingRunner(
                config=rocprof_config,
                target="openmp-cg",
                output_dir=test_output_dir,
                env=ompt_sampling_env,
                timeout=300,
                pass_regex=self.SAMPLING_PASS_REGEX,
            )
        except FileNotFoundError:
            pytest.skip("openmp-cg binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"Sampling duration test failed: {result.test_output}")

    def test_lu_sampling_duration(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        ompt_sampling_env: dict[str, str],
        collect_result,
    ):
        """Test OpenMP LU with sampling duration limits."""
        try:
            runner = SamplingRunner(
                config=rocprof_config,
                target="openmp-lu",
                output_dir=test_output_dir,
                env=ompt_sampling_env,
                timeout=300,
                pass_regex=self.SAMPLING_PASS_REGEX,
            )
        except FileNotFoundError:
            pytest.skip("openmp-lu binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"LU sampling duration test failed: {result.test_output}")


# ============================================================================
# Test Class: No Temporary Files Tests
# ============================================================================


@pytest.mark.gpu
class TestNoTmpFiles:
    """Tests for operation without temporary files."""

    NOTMP_SAMPLING_FILE_REGEX = [
        r"sampling_percent\.(json|txt)",
        r"sampling_cpu_clock\.(json|txt)",
        r"sampling_wall_clock\.(json|txt)",
    ]

    def test_cg_no_tmp_files(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
        ompt_no_tmp_env: dict[str, str],
        collect_result,
        subtests,
    ):
        """Test OpenMP CG without temporary files."""
        try:
            runner = SamplingRunner(
                config=rocprof_config,
                target="openmp-cg",
                output_dir=test_output_dir,
                env=ompt_no_tmp_env,
                pass_regex=self.NOTMP_SAMPLING_FILE_REGEX,
                timeout=300,
            )
        except FileNotFoundError:
            pytest.skip("openmp-cg binary not found")

        result = runner.run()
        collect_result(result)
        if not result.success:
            pytest.fail(f"No tmp files test failed: {result.test_output}")

        with subtests.test("Sampling output files validation"):
            sampling_files = list(result.output_dir.glob("sampling_*.json")) + list(
                result.output_dir.glob("sampling_*.txt")
            )
            if not sampling_files:
                pytest.fail(f"No sampling output files created")
            if not result.perfetto_file:
                pytest.fail(f"Perfetto trace not created")
