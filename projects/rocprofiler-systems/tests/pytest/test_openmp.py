# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Tests for OpenMP integration with rocprofiler-systems.
"""

from __future__ import annotations
import pytest
from pathlib import Path
from conftest import RocprofsysTest

pytestmark = [
    pytest.mark.openmp,
    pytest.mark.ci_enable,
    pytest.mark.rocm_min_version(
        "6.4"
    ),  # Requires SDK version >= 600, 6.3 ships with 500
]

# ============================================================================
# OpenMP Fixtures
# ============================================================================


@pytest.fixture
def ompt_base_env() -> dict[str, str]:
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
def ompt_sampling_env(ompt_base_env: dict[str, str]) -> dict[str, str]:
    """Environment variables for sampling duration tests."""
    env = ompt_base_env.copy()
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
def ompt_target_env(ompt_base_env: dict[str, str]) -> dict[str, str]:
    """Environment variables for OpenMP target (GPU) tests."""
    env = ompt_base_env.copy()
    env["ROCPROFSYS_ROCM_DOMAINS"] = "hip_api,hsa_api,kernel_dispatch"
    return env


@pytest.fixture
def ompt_no_tmp_env(ompt_base_env: dict[str, str]) -> dict[str, str]:
    """Environment variables for no-tmp-files tests."""
    env = ompt_base_env.copy()
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
    BINARY_REWRITE_ARGS = ["-e", "-v", "2", "--instrument-loops"]
    DURATION_SAMPLING_PASS_REGEX = [
        r"Sampler for thread 0 will be triggered 1000\.0x per second of CPU-time",
        r"Sampler for thread 0 will be triggered 500\.0x per second of wall-time",
        r"Sampling will be disabled after 0\.250000 seconds",
        r"Sampling duration of 0\.250000 seconds (has elapsed|was interrupted by finalization)\. Shutting down sampling",
        r"sampling_percent\.(json|txt)",
        r"sampling_cpu_clock\.(json|txt)",
        r"sampling_wall_clock\.(json|txt)",
    ]
    NOTMP_SAMPLING_FILE_REGEX = [
        r"sampling_percent\.(json|txt)",
        r"sampling_cpu_clock\.(json|txt)",
        r"sampling_wall_clock\.(json|txt)",
    ]

    @pytest.mark.timeout(180)
    @pytest.mark.parametrize("mode", ["sampling", "binary_rewrite"])
    def test(self, mode, ompt_base_env):
        env = ompt_base_env.copy()
        env["ROCPROFSYS_USE_SAMPLING"] = "OFF"
        env["ROCPROFSYS_COUT_OUTPUT"] = "ON"

        result = self.run_test(
            mode,
            "openmp-cg",
            env=env,
            binary_rewrite_args=self.BINARY_REWRITE_ARGS,
        )
        self.assert_regex(result)

    @pytest.mark.timeout(300)
    @pytest.mark.sampling_duration
    def test_sampling_duration(self, ompt_sampling_env):
        result = self.run_test(
            "sampling",
            target="openmp-cg",
            env=ompt_sampling_env,
        )
        self.assert_regex(result, pass_regex=self.DURATION_SAMPLING_PASS_REGEX)

    @pytest.mark.timeout(300)
    @pytest.mark.no_tmp_files
    def test_no_tmp_files(self, ompt_no_tmp_env):
        result = self.run_test(
            "sampling",
            target="openmp-cg",
            env=ompt_no_tmp_env,
        )
        self.assert_regex(result, pass_regex=self.NOTMP_SAMPLING_FILE_REGEX)
        self.assert_perfetto(result)


# ============================================================================
# Test Class: OpenMP LU Tests
# ============================================================================


class TestOpenMPLU(RocprofsysTest):
    BINARY_REWRITE_ARGS = ["-e", "-v", "2", "--instrument-loops"]
    BINARY_REWRITE_PASS_REGEX = ["\\|_omp_"]
    BINARY_REWRITE_FAIL_REGEX = ["0 instrumented loops in procedure"]
    DURATION_SAMPLING_PASS_REGEX = [
        r"Sampler for thread 0 will be triggered 1000\.0x per second of CPU-time",
        r"Sampler for thread 0 will be triggered 500\.0x per second of wall-time",
        r"Sampling will be disabled after 0\.250000 seconds",
        r"Sampling duration of 0\.250000 seconds (has elapsed|was interrupted by finalization)\. Shutting down sampling",
        r"sampling_percent\.(json|txt)",
        r"sampling_cpu_clock\.(json|txt)",
        r"sampling_wall_clock\.(json|txt)",
    ]
    NOTMP_SAMPLING_FILE_REGEX = [
        r"sampling_percent\.(json|txt)",
        r"sampling_cpu_clock\.(json|txt)",
        r"sampling_wall_clock\.(json|txt)",
    ]

    @pytest.mark.timeout(180)
    @pytest.mark.parametrize("mode", ["baseline", "sampling", "binary_rewrite"])
    def test(self, mode, ompt_base_env):
        env = ompt_base_env.copy()
        env["ROCPROFSYS_USE_SAMPLING"] = "ON"
        env["ROCPROFSYS_SAMPLING_FREQ"] = "50"
        env["ROCPROFSYS_COUT_OUTPUT"] = "ON"

        result = self.run_test(
            mode,
            "openmp-lu",
            env=env,
            binary_rewrite_args=self.BINARY_REWRITE_ARGS,
        )
        self.assert_regex(
            result,
            mode,
            binary_rewrite_pass_regex=self.BINARY_REWRITE_PASS_REGEX,
            binary_rewrite_fail_regex=self.BINARY_REWRITE_FAIL_REGEX,
        )

    @pytest.mark.timeout(300)
    @pytest.mark.sampling_duration
    def test_sampling_duration(self, ompt_sampling_env):
        result = self.run_test(
            "sampling",
            target="openmp-lu",
            env=ompt_sampling_env,
        )
        self.assert_regex(result, pass_regex=self.DURATION_SAMPLING_PASS_REGEX)


# ============================================================================
# Test Class: OpenMP Target (GPU) Tests
# ============================================================================


@pytest.mark.ci_disable("all")  # TODO: Deprecate once TheRock switches to CTest
@pytest.mark.rocm
@pytest.mark.gpu
@pytest.mark.class_name("openmp-target")
class TestOpenMPTarget(RocprofsysTest):
    @pytest.mark.parametrize(
        "mode",
        [
            "baseline",
            pytest.param("sampling", marks=pytest.mark.rocpd("ompt_target_env")),
            "sys_run",
        ],
    )
    def test(self, mode, ompt_target_env, openmp_target_rules):
        result = self.run_test(mode, "openmp-target", env=ompt_target_env)
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
                counts=[4, 4, 4],
            )


# ============================================================================
# Test Class: OpenMP Fortran Tests
# ============================================================================


@pytest.mark.fortran
@pytest.mark.class_name("openmp-fortran")
class TestOpenMPFortran(RocprofsysTest):

    BINARY_REWRITE_ARGS = ["-e", "-v", "2", "--instrument-loops"]
    RUNTIME_INSTRUMENT_ARGS = ["-e", "-v", "2", "--label", "return", "args"]

    @pytest.mark.parametrize(
        "mode",
        ["baseline", "sampling", "binary_rewrite", "runtime_instrument", "sys_run"],
    )
    def test_host(self, mode, ompt_base_env):
        env = ompt_base_env.copy()
        env["ROCPROFSYS_COUT_OUTPUT"] = "ON"

        result = self.run_test(
            mode,
            "openmp-fortran-host",
            env=env,
            binary_rewrite_args=self.BINARY_REWRITE_ARGS,
            runtime_instrument_args=self.RUNTIME_INSTRUMENT_ARGS,
        )
        self.assert_regex(
            result,
            mode,
            binary_rewrite_pass_regex=["omp_parallel"],
            runtime_instrument_pass_regex=["omp_parallel"],
            sys_run_pass_regex=["omp_parallel"],
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
                marks=[
                    pytest.mark.slow,
                    pytest.mark.serialize,
                    pytest.mark.ci_disable(
                        "all"
                    ),  # TODO: Deprecate once TheRock switches to CTest
                ],
            ),
        ],
    )
    @pytest.mark.gpu
    def test_offload(self, mode, ompt_target_env):
        env = ompt_target_env.copy()
        env["ROCPROFSYS_COUT_OUTPUT"] = "ON"

        result = self.run_test(
            mode,
            "openmp-fortran-offload",
            env=env,
            binary_rewrite_args=self.BINARY_REWRITE_ARGS,
            runtime_instrument_args=self.RUNTIME_INSTRUMENT_ARGS,
            check_target_arch=True,
        )
        self.assert_regex(
            result,
            mode,
            binary_rewrite_pass_regex=["omp_offloading"],
            runtime_instrument_pass_regex=["omp_offloading"],
            sys_run_pass_regex=["omp_offloading"],
        )
