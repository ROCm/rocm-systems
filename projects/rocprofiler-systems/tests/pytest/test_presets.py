# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Preset, domain flag, export config, and help tests.
Mirrors rocprof-sys-preset-tests.cmake for pytest execution.
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest

pytestmark = [pytest.mark.presets]

PRESETS = [
    "balanced",
    "profile-only",
    "detailed",
    "trace-hpc",
    "sys-trace",
    "runtime-trace",
    "trace-gpu",
    "trace-openmp",
    "profile-mpi",
    "trace-hw-counters",
]

# workload-trace requires rocPD and a valid GPU — tested separately
ROCPD_PRESETS = ["workload-trace"]

# ============================================================================
# Preset Tests (rocprof-sys-sample)
# ============================================================================


class TestSamplePresets(RocprofsysTest):
    @pytest.mark.sampling
    @pytest.mark.parametrize("preset", PRESETS)
    def test_preset(self, preset):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-sample",
            run_args=[f"--preset={preset}", "-v", "2", "--", "ls"],
            timeout=60,
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[f"Preset:        {preset}"],
        )

    @pytest.mark.sampling
    @pytest.mark.gpu
    @pytest.mark.rocpd
    @pytest.mark.parametrize("preset", ROCPD_PRESETS)
    def test_preset_rocpd(self, preset):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-sample",
            run_args=[f"--preset={preset}", "-v", "2", "--", "ls"],
            timeout=60,
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[f"Preset:        {preset}"],
        )


# ============================================================================
# Preset Tests (rocprof-sys-run)
# ============================================================================


class TestRunPresets(RocprofsysTest):
    @pytest.mark.sys_run
    @pytest.mark.parametrize("preset", PRESETS)
    def test_preset(self, preset):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-run",
            run_args=[f"--preset={preset}", "-v", "2", "--", "ls"],
            timeout=60,
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[f"Preset:        {preset}"],
        )

    @pytest.mark.sys_run
    @pytest.mark.gpu
    @pytest.mark.rocpd
    @pytest.mark.parametrize("preset", ROCPD_PRESETS)
    def test_preset_rocpd(self, preset):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-run",
            run_args=[f"--preset={preset}", "-v", "2", "--", "ls"],
            timeout=60,
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[f"Preset:        {preset}"],
        )


# ============================================================================
# Domain Flag Tests (rocprof-sys-sample)
# ============================================================================


class TestSampleDomainFlags(RocprofsysTest):
    @pytest.mark.sampling
    def test_gpu(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-sample",
            run_args=["--gpu", "-v", "2", "--", "ls"],
            timeout=60,
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["ROCPROFSYS_USE_AMD_SMI=true"])

    @pytest.mark.sampling
    def test_gpu_metrics(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-sample",
            run_args=["--gpu=temp,power", "-v", "2", "--", "ls"],
            timeout=60,
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["ROCPROFSYS_AMD_SMI_METRICS=temp,power"])

    @pytest.mark.sampling
    def test_rocm(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-sample",
            run_args=["--rocm=hip,kernel", "-v", "2", "--", "ls"],
            timeout=60,
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=["ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch"],
        )

    @pytest.mark.sampling
    def test_cpu(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-sample",
            run_args=["--cpu=50", "-v", "2", "--", "ls"],
            timeout=60,
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["ROCPROFSYS_SAMPLING_FREQ=50"])

    @pytest.mark.sampling
    def test_parallel(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-sample",
            run_args=["--parallel=mpi,openmp", "-v", "2", "--", "ls"],
            timeout=60,
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["ROCPROFSYS_USE_MPIP=true"])

    @pytest.mark.sampling
    def test_preset_plus_domain(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-sample",
            run_args=[
                "--preset=balanced",
                "--gpu=temp,power",
                "-v",
                "2",
                "--",
                "ls",
            ],
            timeout=60,
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["ROCPROFSYS_AMD_SMI_METRICS=temp,power"])


# ============================================================================
# Domain Flag Tests (rocprof-sys-run)
# ============================================================================


class TestRunDomainFlags(RocprofsysTest):
    @pytest.mark.sys_run
    def test_gpu(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-run",
            run_args=["--gpu", "-v", "2", "--", "ls"],
            timeout=60,
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["ROCPROFSYS_USE_AMD_SMI=true"])

    @pytest.mark.sys_run
    def test_gpu_metrics(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-run",
            run_args=["--gpu=temp,power", "-v", "2", "--", "ls"],
            timeout=60,
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["ROCPROFSYS_AMD_SMI_METRICS=temp,power"])

    @pytest.mark.sys_run
    def test_rocm(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-run",
            run_args=["--rocm=hip,kernel", "-v", "2", "--", "ls"],
            timeout=60,
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=["ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch"],
        )

    @pytest.mark.sys_run
    def test_cpu(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-run",
            run_args=["--cpu=50", "-v", "2", "--", "ls"],
            timeout=60,
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["ROCPROFSYS_SAMPLING_FREQ=50"])

    @pytest.mark.sys_run
    def test_parallel(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-run",
            run_args=["--parallel=mpi,openmp", "-v", "2", "--", "ls"],
            timeout=60,
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["ROCPROFSYS_USE_MPIP=true"])

    @pytest.mark.sys_run
    def test_preset_plus_domain(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-run",
            run_args=[
                "--preset=balanced",
                "--gpu=temp,power",
                "-v",
                "2",
                "--",
                "ls",
            ],
            timeout=60,
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["ROCPROFSYS_AMD_SMI_METRICS=temp,power"])


# ============================================================================
# Export Config Tests
# ============================================================================


class TestExportConfig(RocprofsysTest):
    @pytest.mark.sys_run
    def test_export_run(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-run",
            run_args=["--preset=balanced", "--export-config"],
            timeout=30,
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=['"name": "balanced"'])

    @pytest.mark.sampling
    def test_export_sample(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-sample",
            run_args=["--preset=balanced", "--export-config"],
            timeout=30,
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=['"name": "balanced"'])


# ============================================================================
# List Presets and Explain Tests
# ============================================================================


class TestPresetDiscovery(RocprofsysTest):
    @pytest.mark.sys_run
    def test_list_presets_run(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-run",
            run_args=["--list-presets"],
            timeout=30,
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["Available Presets:"])

    @pytest.mark.sampling
    def test_list_presets_sample(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-sample",
            run_args=["--list-presets"],
            timeout=30,
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["Available Presets:"])

    @pytest.mark.sys_run
    def test_explain_preset_run(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-run",
            run_args=["--explain=balanced"],
            timeout=30,
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["Preset: balanced"])

    @pytest.mark.sampling
    def test_explain_preset_sample(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-sample",
            run_args=["--explain=balanced"],
            timeout=30,
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["Preset: balanced"])
