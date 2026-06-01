# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
HPC Example Tests.
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest
from pathlib import Path

pytestmark = [pytest.mark.hpc]


# =============================================================================
# HPC Example Fixtures
# =============================================================================


@pytest.fixture
def hpc_openmp_environment() -> dict[str, str]:
    """Environment variables for HPC OpenMP tests."""
    return {
        "ROCPROFSYS_TRACE": "ON",
        "ROCPROFSYS_PROFILE": "ON",
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
        "ROCPROFSYS_USE_OMPT": "ON",
        "ROCPROFSYS_TIMEMORY_COMPONENTS": "wall_clock,trip_count,peak_rss",
        "OMP_PROC_BIND": "spread",
        "OMP_PLACES": "threads",
        "OMP_NUM_THREADS": "2",
        "ROCPROFSYS_COUT_OUTPUT": "ON",
    }


@pytest.fixture
def hpc_hip_environment() -> dict[str, str]:
    """Environment variables for HPC HIP tests."""
    return {
        "ROCPROFSYS_TRACE": "ON",
        "ROCPROFSYS_PROFILE": "ON",
        "ROCPROFSYS_TIME_OUTPUT": "OFF",
        "ROCPROFSYS_TIMEMORY_COMPONENTS": "wall_clock,trip_count,peak_rss",
        "ROCPROFSYS_COUT_OUTPUT": "ON",
    }


@pytest.fixture
def kfd_rules(validation_rules_dir) -> list[Path]:
    """Get validation rules for KFD event tests."""
    rules_dir = validation_rules_dir / "kfd"
    return [
        rules_dir / "kfd-rules.json",
    ]


@pytest.fixture
def matrix_exponential_rules(validation_rules_dir) -> list[Path]:
    """Get validation rules for matrix exponential tests."""
    rules_dir = validation_rules_dir / "hpc" / "matrix-exponential"
    return [
        rules_dir / "sdk-metrics-rules.json",
    ]


@pytest.fixture
def split_copy_compute_hw_queues_rules(validation_rules_dir) -> list[Path]:
    """Get validation rules for split copy compute hw queues tests."""
    rules_dir = validation_rules_dir / "hpc" / "split-copy-compute"
    return [
        rules_dir / "sdk-metrics-rules.json",
    ]


# =============================================================================
# HPC Tests
# =============================================================================


@pytest.mark.gpu
class TestJacobi(RocprofsysTest):

    openmp_non_usm_run_args = ["-m", "512"]
    # With OMPX_APU_MAPS=1, it takes longer, so reduce domain size
    openmp_usm_run_args = ["-m", "64"]
    hip_run_args = ["-g", "2", "1"]

    @pytest.mark.slow
    @pytest.mark.openmp
    @pytest.mark.xnack
    @pytest.mark.rocpd("hpc_openmp_environment")
    @pytest.mark.parametrize("mode", ["sys_run"])
    def test_usm(self, mode, hpc_openmp_environment, gpu_info, kfd_rules):
        env = hpc_openmp_environment.copy()
        env["ROCPROFSYS_ROCM_DOMAINS"] = "hip_api,kernel_dispatch,memory_copy"
        env["ROCPROFSYS_TRACE_LEGACY"] = "ON"
        env["HSA_XNACK"] = "1"
        env["ROCPROFSYS_USE_AMD_SMI"] = "OFF"
        if "apu" not in gpu_info.categories:
            # Forces zero-copy behavior on non-APU GPUs
            # Without this, it would be implicit zero-copy behavior
            env["OMPX_APU_MAPS"] = "1"

        result = self.run_test(
            mode,
            target="jacobi-fortran-usm",
            env=env,
            run_args=self.openmp_usm_run_args,
            check_target_arch=True,
        )

        if "apu" in gpu_info.categories:
            # We expect no omp_target_data_op_emi (CPU and GPU share the same memory)
            self.assert_regex(
                result,
                subtest_name="USM Zero-Copy Validation (APU)",
                fail_regex=["omp_target_data_op_emi"],
            )

            self.assert_perfetto(
                result,
                subtest_name="Perfetto USM Zero-Copy Validation (APU)",
                fail_regex=["omp_target_data_op_emi"],
            )

        else:
            # We expect to see only one omp_target_data_op_emi
            # Corresponds to the Fortran array descriptor being transferred
            #   at the start of the program
            self.assert_regex(
                result,
                subtest_name="USM Zero-Copy validation (Non-APU)",
                pass_regex=[r"omp_target_data_op_emi\s*\|\s*1\s*\|\s*1\s*\|"],
            )

            self.assert_perfetto(
                result,
                subtest_name="Perfetto USM Zero-Copy validation (Non-APU)",
                categories=["rocm_ompt_api"],
                pass_regex=[r"omp_target_data_op_emi\s*\|\s*1\s*\|\s*1\s*\|"],
            )

    @pytest.mark.openmp
    @pytest.mark.roctx
    @pytest.mark.parametrize("mode", ["binary_rewrite", "sys_run"])
    def test_roctx(self, mode, hpc_openmp_environment):
        env = hpc_openmp_environment.copy()
        env["ROCPROFSYS_ROCM_DOMAINS"] = "hip_api,kernel_dispatch,roctx,memory_copy"
        env["ROCPROFSYS_TRACE_LEGACY"] = "ON"
        env["ROCPROFSYS_COUT_OUTPUT"] = "OFF"

        result = self.run_test(
            mode,
            target="jacobi-fortran-targetdata-markers",
            env=env,
            check_target_arch=True,
            run_args=self.openmp_non_usm_run_args,
        )
        self.assert_regex(result)

        self.assert_perfetto(
            result,
            subtest_name="Perfetto ROCtx marker validation",
            categories=["rocm_marker_api"],
            labels=["init", "run"],
            counts=[1, 1],
            depths=[1, 1],
        )

    @pytest.mark.hip
    @pytest.mark.mpi
    @pytest.mark.multi_gpu(2)
    @pytest.mark.rocpd("hpc_hip_environment")
    @pytest.mark.parametrize("mode", ["sys_run"])
    def test_hip(self, mode, hpc_hip_environment):
        env = hpc_hip_environment.copy()
        env["ROCPROFSYS_PERFETTO_COMBINE_TRACES"] = "ON"

        result = self.run_test(
            mode,
            target="jacobi-hip",
            env=env,
            run_args=self.hip_run_args,
            launcher="mpi",
            num_procs=2,
        )
        self.assert_regex(result)
        # Taken from the program's defines.hpp
        JACOBI_MAX_LOOPS = 1000

        # With -g 2 1, 2 MPI ranks exist. The merged perfetto trace has 2x the per-rank count
        self.assert_perfetto(
            result,
            subtest_name="Laplacian Kernel Count Validation",
            perfetto_file="merged.proto",
            categories=["rocm_hip_stream"],
            print_output=True,
            pass_regex=[rf"LocalLaplacianKernel.*\|\s+{JACOBI_MAX_LOOPS * 2}\s+\|"],
        )

        # hipHostFree is one of the last calls — tracked via rocprofiler-sdk in Perfetto
        self.assert_perfetto(
            result,
            subtest_name="hipHostFree Validation",
            perfetto_file="merged.proto",
            categories=["rocm_hip_api"],
            pass_regex=[r"hipHostFree\s*\|\s*[1-9]"],
        )


@pytest.mark.gpu
@pytest.mark.hip
@pytest.mark.openmp
@pytest.mark.roctx
@pytest.mark.class_name("matrix-exponential")
class TestMatrixExponential(RocprofsysTest):

    rocblas_gemm_kernel_prefix = ["Cijk_Ailk_Bljk"]

    @pytest.mark.rocpd("hpc_hip_environment")
    @pytest.mark.parametrize(
        "mode",
        [
            "binary_rewrite",
            "sys_run",
        ],
    )
    def test(self, mode, hpc_hip_environment, matrix_exponential_rules):
        env = hpc_hip_environment.copy()
        env["ROCPROFSYS_ROCM_DOMAINS"] = "hip_api,kernel_dispatch,roctx,memory_copy"
        env["ROCPROFSYS_USE_OMPT"] = "ON"
        env["ROCPROFSYS_TRACE_LEGACY"] = "ON"

        result = self.run_test(
            mode, target="matrix-exponential-streams-sync-hip", env=env
        )
        self.assert_regex(
            result,
            pass_regex=self.rocblas_gemm_kernel_prefix,
            subtest_name="rocBLAS GEMM Kernel Validation",
        )
        # 171 total GEMM dispatches (sum of 1 to 18 from the Taylor series loop),
        # but schedule(dynamic) distributes them non-deterministically across
        # 4 OpenMP threads, so we verify presence rather than exact per-thread counts
        self.assert_perfetto(
            result,
            subtest_name="Perfetto rocBLAS GEMM Kernel Validation",
            print_output=True,
            pass_regex=self.rocblas_gemm_kernel_prefix,
        )

        # TODO: Disabled pending investigation (AIPROFSYST-418)
        # We expect 171 GEMM dispatches, but sometimes we see less.

        # self.assert_rocpd(
        #     result,
        #     rules_files=matrix_exponential_rules,
        # )


@pytest.mark.rocm
@pytest.mark.gpu
@pytest.mark.hip
@pytest.mark.class_name("split-copy-compute-hw-queues")
class TestSplitCopyComputeHWQueues(RocprofsysTest):

    nstreams = 4

    @pytest.mark.rocpd("hpc_hip_environment")
    @pytest.mark.parametrize("mode", ["sys_run"])
    def test(self, mode, hpc_hip_environment, split_copy_compute_hw_queues_rules):
        env = hpc_hip_environment.copy()
        env["ROCPROFSYS_ROCM_DOMAINS"] = "hip_api,hsa_api,kernel_dispatch,memory_copy"
        env["ROCPROFSYS_COUT_OUTPUT"] = "OFF"
        env["ROCPROFSYS_ROCM_GROUP_BY_QUEUE"] = "ON"

        result = self.run_test(
            mode,
            target="split-copy-compute-hw-queues",
            env=env,
            run_args=[str(self.nstreams)],
            check_target_arch=True,
        )
        self.assert_regex(result)

        # We expect to see "nstreams" amount of GPU Kernel Dispatch tracks, each with a cube kernel
        self.assert_perfetto(
            result,
            subtest_name="Stream Count Validation",
            categories=["rocm_kernel_dispatch"],
            label_substrings=["cube"],
            counts=[self.nstreams],
            depths=[0],
        )

        self.assert_rocpd(
            result,
            rules_files=split_copy_compute_hw_queues_rules,
        )
