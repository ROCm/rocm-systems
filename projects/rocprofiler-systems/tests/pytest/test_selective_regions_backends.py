# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Region filter with MPI, OMPT, UCX, and Kokkos enabled."""

from __future__ import annotations

import pytest
from conftest import RocprofsysTest, check_use_perfetto
from selective_regions_conftest import merge_selective_env

pytest_plugins = ["selective_regions_conftest"]

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.selective_regions,
    pytest.mark.selective_regions_backends,
    pytest.mark.timeout(180),
    pytest.mark.rocm,
]

_REGION1_KERNELS_PASS = [
    "CodeBlock_B",
    "CodeBlock_C",
    "CodeBlock_D",
    "CodeBlock_F",
]
_REGION1_KERNELS_FAIL = ["CodeBlock_A", "CodeBlock_E", "CodeBlock_G"]


@pytest.fixture
def backend_sys_run_env(selective_region_env, sys_run_manual_env) -> dict[str, str]:
    env = merge_selective_env("sys_run", selective_region_env, sys_run_manual_env)
    env["ROCPROFSYS_SELECTED_REGIONS"] = "Region1"
    return env


@pytest.fixture
def ucx_selective_env() -> dict[str, str]:
    return {
        "ROCPROFSYS_USE_UCX": "ON",
        "ROCPROFSYS_MPI_INIT": "OFF",
        "OMPI_MCA_pml": "ucx",
        "OMPI_MCA_osc": "ucx",
        "OMPI_MCA_pml_ucx_tls": "tcp,self",
        "OMPI_MCA_pml_ucx_devices": "any",
        "OMPI_MCA_btl": "^vader,sm",
        "UCX_TLS": "tcp,self",
    }


def _assert_region1_kernels(test: RocprofsysTest, result) -> None:
    test.assert_regex(result)
    if not check_use_perfetto():
        return
    test.assert_perfetto(
        result,
        subtest_name="Region1 filtered kernels (backend)",
        categories=["rocm_hip_stream"],
        pass_regex=_REGION1_KERNELS_PASS,
        fail_regex=_REGION1_KERNELS_FAIL,
    )


@pytest.mark.class_name("selective-region-backends")
class TestSelectiveRegionsMpiBackend(RocprofsysTest):
    @pytest.mark.mpi_optional("selective_region")
    def test_mpip_region_filter(self, backend_sys_run_env):
        env = backend_sys_run_env.copy()
        env["ROCPROFSYS_USE_MPIP"] = "ON"
        result = self.run_test(
            "sys_run",
            "selective_region",
            env=env,
            launcher="mpi",
            num_procs=2,
            check_target_arch=True,
        )
        _assert_region1_kernels(self, result)


@pytest.mark.class_name("selective-region-backends")
class TestSelectiveRegionsOmptBackend(RocprofsysTest):
    @pytest.mark.openmp
    def test_ompt_hot_phase_filter(self, backend_sys_run_env):
        env = backend_sys_run_env.copy()
        env["ROCPROFSYS_USE_OMPT"] = "ON"
        env["ROCPROFSYS_SELECTED_REGIONS"] = "HotPhase"
        result = self.run_test(
            "sys_run",
            "omp_app",
            env=env,
            check_target_arch=True,
        )
        self.assert_regex(result)
        if check_use_perfetto():
            self.assert_perfetto(
                result,
                subtest_name="OMPT HotPhase kernels",
                categories=["rocm_kernel_dispatch"],
                pass_regex=["CodeBlock_Hot"],
                fail_regex=["CodeBlock_Warmup", "CodeBlock_Cooldown"],
            )


@pytest.mark.class_name("selective-region-backends")
class TestSelectiveRegionsUcxBackend(RocprofsysTest):
    @pytest.mark.ucx
    @pytest.mark.mpi_optional("selective_region")
    @pytest.mark.mpi_implementation("openmpi")
    def test_ucx_region_filter(self, backend_sys_run_env, ucx_selective_env):
        env = backend_sys_run_env.copy()
        env.update(ucx_selective_env)
        env["ROCPROFSYS_USE_MPIP"] = "ON"
        result = self.run_test(
            "sys_run",
            "selective_region",
            env=env,
            launcher="mpi",
            num_procs=2,
            check_target_arch=True,
        )
        _assert_region1_kernels(self, result)


@pytest.mark.class_name("selective-region-backends")
class TestSelectiveRegionsKokkosBackend(RocprofsysTest):
    def test_kokkosp_region_filter(self, backend_sys_run_env):
        env = backend_sys_run_env.copy()
        env["ROCPROFSYS_USE_KOKKOSP"] = "ON"
        env["ROCPROFSYS_KOKKOSP_PREFIX"] = "[kokkos]"
        result = self.run_test(
            "sys_run",
            "selective_region",
            env=env,
            check_target_arch=True,
        )
        _assert_region1_kernels(self, result)
