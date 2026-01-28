# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

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
import pytest
from pathlib import Path

# OpenMP will not be traced if no GPU is available, this includes CPU-only
pytestmark = [pytest.mark.gpu, pytest.mark.openmp]

# ============================================================================
# OpenMP Fixtures
# ============================================================================


@pytest.fixture
def ompt_env() -> dict[str, str]:
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


class TestOpenMPCG:
    """Tests for OpenMP Conjugate Gradient example."""

    REWRITE_ARGS = ["-e", "-v", "2", "--instrument-loops"]

    def test_sampling(
        self,
        ompt_env: dict[str, str],
        run_test,
        assert_regex,
    ):
        env = ompt_env.copy()
        env["ROCPROFSYS_USE_SAMPLING"] = "OFF"
        env["ROCPROFSYS_COUT_OUTPUT"] = "ON"

        result = run_test(
            "sampling",
            target="openmp-cg",
            env=env,
            timeout=180,
            no_check_target_arch=True,
        )
        assert_regex(result)

    def test_binary_rewrite(
        self,
        run_test,
        ompt_env: dict[str, str],
        assert_regex,
    ):
        env = ompt_env.copy()
        env["ROCPROFSYS_USE_SAMPLING"] = "OFF"
        env["ROCPROFSYS_COUT_OUTPUT"] = "ON"

        result = run_test(
            "binary_rewrite",
            target="openmp-cg",
            rewrite_args=self.REWRITE_ARGS,
            env=env,
            timeout=180,
            no_check_target_arch=True,
        )

        assert_regex(result)


# ============================================================================
# Test Class: OpenMP LU Tests
# ============================================================================


class TestOpenMPLU:
    """Tests for OpenMP LU decomposition example."""

    REWRITE_ARGS = ["-e", "-v", "2", "--instrument-loops"]

    def test_binary_rewrite(
        self,
        run_test,
        ompt_env: dict[str, str],
        assert_regex,
    ):
        env = ompt_env.copy()
        env["ROCPROFSYS_USE_SAMPLING"] = "ON"
        env["ROCPROFSYS_SAMPLING_FREQ"] = "50"
        env["ROCPROFSYS_COUT_OUTPUT"] = "ON"

        result = run_test(
            "binary_rewrite",
            target="openmp-lu",
            rewrite_args=self.REWRITE_ARGS,
            env=env,
            timeout=180,
            no_check_target_arch=True,
        )
        assert_regex(result)


# ============================================================================
# Test Class: OpenMP Target (GPU) Tests
# ============================================================================


@pytest.mark.openmp_target
class TestOpenMPTarget:
    """Tests for OpenMP target offload (GPU) example."""

    @pytest.mark.rocpd("openmp_target_env")
    def test_sampling(
        self,
        run_test,
        openmp_target_env: dict[str, str],
        openmp_target_rules: list[Path],
        assert_regex,
        assert_perfetto,
        assert_rocpd,
    ):
        result = run_test(
            "sampling",
            target="openmp-target",
            env=openmp_target_env,
            timeout=300,
            no_check_target_arch=True,
        )

        assert_regex(result)
        assert_rocpd(result, rules_files=openmp_target_rules)
        assert_perfetto(
            result,
            subtest_name="Perfetto Kernel Dispatch Validation",
            categories=["rocm_kernel_dispatch"],
            label_substrings=[
                "Z4vmulIiEvPT_S1_S1_i_l51.kd",
                "Z4vmulIfEvPT_S1_S1_i_l51.kd",
                "Z4vmulIdEvPT_S1_S1_i_l51.kd",
            ],
            depths=[0, 0, 0],
            counts=[4, 4, 4],
        )


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
@pytest.mark.ompvv
class TestOpenMPVVHost:
    """Tests for OpenMP VV host programs."""

    def test_baseline(
        self,
        run_test,
        ompt_env: dict[str, str],
        target_name: str,
        assert_regex,
    ):
        result = run_test(
            "baseline",
            target=target_name,
            env=ompt_env,
            timeout=180,
            no_check_target_arch=True,
        )

        assert_regex(result)

    def test_sampling(
        self,
        run_test,
        ompt_env: dict[str, str],
        target_name: str,
        assert_regex,
        assert_perfetto,
    ):
        result = run_test(
            "sampling",
            target=target_name,
            env=ompt_env,
            timeout=180,
            no_check_target_arch=True,
        )

        assert_regex(result)
        assert_perfetto(result)

    def test_binary_rewrite(
        self,
        run_test,
        ompt_env: dict[str, str],
        target_name: str,
        assert_regex,
    ):
        env = ompt_env.copy()
        env["ROCPROFSYS_COUT_OUTPUT"] = "ON"

        result = run_test(
            "binary_rewrite",
            target=target_name,
            rewrite_args=["-e", "-v", "2", "--instrument-loops"],
            env=env,
            timeout=180,
            no_check_target_arch=True,
        )

        assert_regex(result, pass_regex=[r"omp_parallel"])

    def test_runtime_instrument(
        self,
        run_test,
        ompt_env: dict[str, str],
        target_name: str,
        assert_regex,
    ):
        env = ompt_env.copy()
        env["ROCPROFSYS_COUT_OUTPUT"] = "ON"
        env["ROCPROFSYS_CI_SKIP_PUSH_POP_CHECK"] = "ON"

        result = run_test(
            "runtime_instrument",
            target=target_name,
            instrument_args=["-e", "-v", "1", "--label", "return", "args"],
            env=env,
            no_check_target_arch=True,
        )

        assert_regex(result, pass_regex=[r"omp_parallel"])

    def test_sys_run(
        self,
        run_test,
        ompt_env: dict[str, str],
        target_name: str,
        assert_regex,
        assert_perfetto,
    ):
        result = run_test(
            "sys_run",
            target=target_name,
            env=ompt_env,
            timeout=180,
            no_check_target_arch=True,
        )

        assert_regex(result)
        assert_perfetto(result)


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
@pytest.mark.openmp_target
@pytest.mark.ompvv
class TestOpenMPVVOffload:
    """Tests for OpenMP VV offload programs."""

    def test_baseline(
        self,
        run_test,
        openmp_target_env: dict[str, str],
        target_name: str,
        assert_regex,
    ):
        result = run_test(
            "baseline",
            target=target_name,
            env=openmp_target_env,
            timeout=300,
        )

        assert_regex(result)

    def test_sampling(
        self,
        run_test,
        openmp_target_env: dict[str, str],
        target_name: str,
        assert_regex,
    ):
        result = run_test(
            "sampling",
            target=target_name,
            env=openmp_target_env,
            timeout=300,
        )

        assert_regex(result)

    def test_binary_rewrite(
        self,
        run_test,
        openmp_target_env: dict[str, str],
        target_name: str,
        assert_regex,
    ):
        env = openmp_target_env.copy()
        env["ROCPROFSYS_COUT_OUTPUT"] = "ON"

        result = run_test(
            "binary_rewrite",
            target=target_name,
            rewrite_args=["-e", "-v", "2"],
            env=env,
            timeout=300,
        )

        assert_regex(result, pass_regex=[r"omp_offloading"])

    def test_sys_run(
        self,
        run_test,
        openmp_target_env: dict[str, str],
        target_name: str,
        assert_regex,
        assert_perfetto,
    ):
        result = run_test(
            "sys_run",
            target=target_name,
            run_args=["-e", "-v", "1", "--label", "return", "args"],
            env=openmp_target_env,
            timeout=300,
        )

        assert_regex(result)
        assert_perfetto(result)


# ============================================================================
# Test Class: Sampling Duration Tests
# ============================================================================


@pytest.mark.sampling_duration
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
        ompt_sampling_env: dict[str, str],
        run_test,
        assert_regex,
    ):
        result = run_test(
            "sampling",
            target="openmp-cg",
            env=ompt_sampling_env,
            timeout=300,
            no_check_target_arch=True,
        )

        assert_regex(result, pass_regex=self.SAMPLING_PASS_REGEX)

    def test_lu_sampling_duration(
        self,
        run_test,
        ompt_sampling_env: dict[str, str],
        assert_regex,
    ):
        """Test OpenMP LU with sampling duration limits."""
        result = run_test(
            "sampling",
            target="openmp-lu",
            env=ompt_sampling_env,
            timeout=300,
            no_check_target_arch=True,
        )

        assert_regex(result, pass_regex=self.SAMPLING_PASS_REGEX)


# ============================================================================
# Test Class: No Temporary Files Tests
# ============================================================================


@pytest.mark.no_tmp_files
class TestNoTmpFiles:
    """Tests for operation without temporary files."""

    NOTMP_SAMPLING_FILE_REGEX = [
        r"sampling_percent\.(json|txt)",
        r"sampling_cpu_clock\.(json|txt)",
        r"sampling_wall_clock\.(json|txt)",
    ]

    def test_cg_no_tmp_files(
        self,
        run_test,
        ompt_no_tmp_env: dict[str, str],
        assert_regex,
        assert_perfetto,
        assert_file_exists,
    ):
        """Test OpenMP CG without temporary files."""
        result = run_test(
            "sampling",
            target="openmp-cg",
            env=ompt_no_tmp_env,
            timeout=300,
            no_check_target_arch=True,
        )

        assert_regex(result, pass_regex=self.NOTMP_SAMPLING_FILE_REGEX)
        assert_perfetto(result)

        sampling_files = list(result.output_dir.glob("sampling_*.json")) + list(
            result.output_dir.glob("sampling_*.txt")
        )
        assert_file_exists(sampling_files)
