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
from conftest import RocprofsysTest

# OpenMP will not be traced if no GPU is available, this includes CPU-only
pytestmark = [pytest.mark.gpu, pytest.mark.openmp, pytest.mark.ci_enable]

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


class TestOpenMPCG(RocprofsysTest):
    REWRITE_ARGS = ["-e", "-v", "2", "--instrument-loops"]
    DURATION_SAMPLING_PASS_REGEX = [
        r"Sampler for thread 0 will be triggered 1000\.0x per second of CPU-time",
        r"Sampler for thread 0 will be triggered 500\.0x per second of wall-time",
        r"Sampling will be disabled after 0\.250000 seconds",
        r"Sampling duration of 0\.250000 seconds has elapsed\. Shutting down sampling",
        r"sampling_percent\.(json|txt)",
        r"sampling_cpu_clock\.(json|txt)",
        r"sampling_wall_clock\.(json|txt)",
    ]
    NOTMP_SAMPLING_FILE_REGEX = [
        r"sampling_percent\.(json|txt)",
        r"sampling_cpu_clock\.(json|txt)",
        r"sampling_wall_clock\.(json|txt)",
    ]

    @pytest.mark.parametrize("mode", ["sampling", "binary_rewrite"])
    def test(self, mode, ompt_env):
        env = ompt_env.copy()
        env["ROCPROFSYS_USE_SAMPLING"] = "OFF"
        env["ROCPROFSYS_COUT_OUTPUT"] = "ON"

        result = self.run_test(
            mode,
            "openmp-cg",
            env=env,
            rewrite_args=self.REWRITE_ARGS,
            timeout=180,
        )
        self.assert_regex(result)

    @pytest.mark.sampling_duration
    def test_sampling_duration(self, ompt_sampling_env):
        result = self.run_test(
            "sampling",
            target="openmp-cg",
            env=ompt_sampling_env,
            timeout=300,
        )
        self.assert_regex(result, pass_regex=self.DURATION_SAMPLING_PASS_REGEX)

    @pytest.mark.no_tmp_files
    def test_no_tmp_files(self, ompt_no_tmp_env):
        result = self.run_test(
            "sampling",
            target="openmp-cg",
            env=ompt_no_tmp_env,
            timeout=300,
        )
        self.assert_regex(result, pass_regex=self.NOTMP_SAMPLING_FILE_REGEX)
        self.assert_perfetto(result)


# ============================================================================
# Test Class: OpenMP LU Tests
# ============================================================================


class TestOpenMPLU(RocprofsysTest):
    REWRITE_ARGS = ["-e", "-v", "2", "--instrument-loops"]
    REWRITE_PASS_REGEX = ["\\|_omp_"]
    REWRITE_FAIL_REGEX = ["0 instrumented loops in procedure"]
    DURATION_SAMPLING_PASS_REGEX = [
        r"Sampler for thread 0 will be triggered 1000\.0x per second of CPU-time",
        r"Sampler for thread 0 will be triggered 500\.0x per second of wall-time",
        r"Sampling will be disabled after 0\.250000 seconds",
        r"Sampling duration of 0\.250000 seconds has elapsed\. Shutting down sampling",
        r"sampling_percent\.(json|txt)",
        r"sampling_cpu_clock\.(json|txt)",
        r"sampling_wall_clock\.(json|txt)",
    ]
    NOTMP_SAMPLING_FILE_REGEX = [
        r"sampling_percent\.(json|txt)",
        r"sampling_cpu_clock\.(json|txt)",
        r"sampling_wall_clock\.(json|txt)",
    ]

    @pytest.mark.parametrize("mode", ["baseline", "sampling", "binary_rewrite"])
    def test(self, mode, ompt_env):
        env = ompt_env.copy()
        env["ROCPROFSYS_USE_SAMPLING"] = "ON"
        env["ROCPROFSYS_SAMPLING_FREQ"] = "50"
        env["ROCPROFSYS_COUT_OUTPUT"] = "ON"

        result = self.run_test(
            mode,
            "openmp-lu",
            env=env,
            rewrite_args=self.REWRITE_ARGS,
            timeout=180,
        )
        self.assert_regex(
            result,
            mode,
            rewrite_pass_regex=self.REWRITE_PASS_REGEX,
            rewrite_fail_regex=self.REWRITE_FAIL_REGEX,
        )

    @pytest.mark.sampling_duration
    def test_sampling_duration(self, ompt_sampling_env):
        result = self.run_test(
            "sampling",
            target="openmp-lu",
            env=ompt_sampling_env,
            timeout=300,
        )
        self.assert_regex(result, pass_regex=self.DURATION_SAMPLING_PASS_REGEX)


# ============================================================================
# Test Class: OpenMP Target (GPU) Tests
# ============================================================================


@pytest.mark.ci_disable("all")
class TestOpenMPTarget(RocprofsysTest):
    @pytest.mark.parametrize(
        "mode",
        [
            "baseline",
            pytest.param("sampling", marks=pytest.mark.rocpd("openmp_target_env")),
            "sys_run",
        ],
    )
    def test(self, mode, openmp_target_env, openmp_target_rules):
        result = self.run_test(mode, "openmp-target", env=openmp_target_env)
        self.assert_regex(result)

        if mode == "sampling":
            self.assert_rocpd(result, rules_files=openmp_target_rules)
            self.assert_perfetto(
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
# Test Class: OpenMP-VV (Validation and Verification)
# ============================================================================


class TestOpenMPVV(RocprofsysTest):
    @pytest.mark.parametrize(
        "mode",
        ["baseline", "sampling", "binary_rewrite", "runtime_instrument", "sys_run"],
    )
    @pytest.mark.parametrize(
        "target",
        [
            "openmp-vv-host-test-parallel-for-simd-atomic",
            "openmp-vv-host-test-team-default-shared",
        ],
        ids=["parallel-for-simd-atomic", "team-default-shared"],
    )
    def test_host(self, mode, target, ompt_env):
        REWRITE_ARGS = ["-e", "-v", "2", "--instrument-loops"]
        RUNTIME_ARGS = ["-e", "-v", "1", "--label", "return", "args"]
        REWRITE_PASS_REGEX = [r"omp_parallel"]
        RUNTIME_PASS_REGEX = [r"omp_parallel"]
        SYS_RUN_PASS_REGEX = [r"omp_parallel"]

        env = ompt_env.copy()
        env["ROCPROFSYS_COUT_OUTPUT"] = "ON"
        if mode == "runtime_instrument":
            env["ROCPROFSYS_CI_SKIP_PUSH_POP_CHECK"] = "ON"

        result = self.run_test(
            mode,
            target,
            env=env,
            rewrite_args=REWRITE_ARGS,
            runtime_args=RUNTIME_ARGS,
            runtime_timeout=300,
        )
        self.assert_regex(
            result,
            mode,
            rewrite_pass_regex=REWRITE_PASS_REGEX,
            runtime_pass_regex=RUNTIME_PASS_REGEX,
            sys_run_pass_regex=SYS_RUN_PASS_REGEX,
        )

    @pytest.mark.parametrize(
        "mode",
        [
            "baseline",
            "sampling",
            "binary_rewrite",
            "sys_run",
            pytest.param(
                "runtime_instrument",
                marks=[pytest.mark.slow, pytest.mark.ci_disable("all")],
            ),
        ],
    )
    @pytest.mark.parametrize(
        "target",
        [
            "openmp-vv-offload-test-target-simd-if",
            "openmp-vv-offload-test-target-teams-distribute-parallel-for-collapse",
        ],
        ids=["target-simd-if", "target-teams-distribute-parallel-for-collapse"],
    )
    def test_offload(self, mode, target, openmp_target_env):
        REWRITE_ARGS = ["-e", "-v", "2"]
        REWRITE_PASS_REGEX = [r"omp_offloading"]
        RUNTIME_PASS_REGEX = [r"omp_offloading"]
        SYS_RUN_PASS_REGEX = [r"omp_offloading"]

        env = openmp_target_env.copy()
        env["ROCPROFSYS_COUT_OUTPUT"] = "ON"

        result = self.run_test(
            mode,
            target,
            env=env,
            rewrite_args=REWRITE_ARGS,
            timeout=300,
            check_target_arch=True,
        )
        self.assert_regex(
            result,
            mode,
            rewrite_pass_regex=REWRITE_PASS_REGEX,
            runtime_pass_regex=RUNTIME_PASS_REGEX,
            sys_run_pass_regex=SYS_RUN_PASS_REGEX,
        )
