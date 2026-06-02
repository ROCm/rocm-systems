# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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


@pytest.mark.sampling
@pytest.mark.class_name("sample-presets")
class TestSamplePresets(RocprofsysTest):
    @pytest.mark.timeout(60)
    @pytest.mark.parametrize("preset", PRESETS)
    def test_preset(self, preset):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-sample",
            run_args=[f"--preset={preset}", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[f"Preset:        {preset}"],
        )

    @pytest.mark.timeout(60)
    @pytest.mark.gpu
    @pytest.mark.rocpd
    @pytest.mark.parametrize("preset", ROCPD_PRESETS)
    def test_preset_rocpd(self, preset):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-sample",
            run_args=[f"--preset={preset}", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[f"Preset:        {preset}"],
        )


# ============================================================================
# Preset Tests (rocprof-sys-run)
# ============================================================================


@pytest.mark.timeout(60)
@pytest.mark.sys_run
@pytest.mark.class_name("run-presets")
class TestRunPresets(RocprofsysTest):
    @pytest.mark.parametrize("preset", PRESETS)
    def test_preset(self, preset):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-run",
            run_args=[f"--preset={preset}", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[f"Preset:        {preset}"],
        )

    @pytest.mark.gpu
    @pytest.mark.rocpd
    @pytest.mark.parametrize("preset", ROCPD_PRESETS)
    def test_preset_rocpd(self, preset):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-run",
            run_args=[f"--preset={preset}", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[f"Preset:        {preset}"],
        )


# ============================================================================
# Domain Flag Tests (rocprof-sys-sample)
# ============================================================================


@pytest.mark.timeout(60)
@pytest.mark.sampling
@pytest.mark.class_name("sample-domain-flags")
class TestSampleDomainFlags(RocprofsysTest):
    def test_gpu(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-sample",
            run_args=["--gpu", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["ROCPROFSYS_USE_AMD_SMI=true"])

    def test_gpu_metrics(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-sample",
            run_args=["--gpu=temp,power", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["ROCPROFSYS_AMD_SMI_METRICS=temp,power"])

    def test_rocm(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-sample",
            run_args=["--rocm=hip,kernel", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=["ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch"],
        )

    def test_cpu(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-sample",
            run_args=["--cpu=50", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["ROCPROFSYS_SAMPLING_FREQ=50"])

    def test_parallel(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-sample",
            run_args=["--parallel=mpi,openmp", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["ROCPROFSYS_USE_MPIP=true"])

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
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["ROCPROFSYS_AMD_SMI_METRICS=temp,power"])


# ============================================================================
# Domain Flag Tests (rocprof-sys-run)
# ============================================================================


@pytest.mark.timeout(60)
@pytest.mark.sys_run
@pytest.mark.class_name("run-domain-flags")
class TestRunDomainFlags(RocprofsysTest):
    def test_gpu(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-run",
            run_args=["--gpu", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["ROCPROFSYS_USE_AMD_SMI=true"])

    def test_gpu_metrics(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-run",
            run_args=["--gpu=temp,power", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["ROCPROFSYS_AMD_SMI_METRICS=temp,power"])

    def test_rocm(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-run",
            run_args=["--rocm=hip,kernel", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=["ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch"],
        )

    def test_cpu(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-run",
            run_args=["--cpu=50", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["ROCPROFSYS_SAMPLING_FREQ=50"])

    def test_parallel(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-run",
            run_args=["--parallel=mpi,openmp", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["ROCPROFSYS_USE_MPIP=true"])

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
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["ROCPROFSYS_AMD_SMI_METRICS=temp,power"])


# ============================================================================
# Sampling target flags (--gpus / --cpus / --ai-nics) stand alone:
# they must not require --device / --host on the command line, so presets
# that already enable the matching backend (e.g. trace-hpc, trace-gpu set
# ROCPROFSYS_USE_AMD_SMI=true) can compose with them.
# ============================================================================


@pytest.mark.timeout(60)
@pytest.mark.sys_run
@pytest.mark.class_name("run-sampling-target-flags")
class TestRunSamplingTargetFlags(RocprofsysTest):
    def test_gpus_without_device(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-run",
            run_args=["--gpus=0", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["ROCPROFSYS_SAMPLING_GPUS=0"])

    def test_cpus_without_host(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-run",
            run_args=["--cpus=0-3", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["ROCPROFSYS_SAMPLING_CPUS=0-3"])

    def test_ai_nics_without_device(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-run",
            run_args=["--ai-nics=nic0", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["ROCPROFSYS_SAMPLING_AINICS=nic0"])

    @pytest.mark.parametrize("preset", ["trace-hpc", "trace-gpu"])
    def test_preset_plus_gpus(self, preset):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-run",
            run_args=[f"--preset={preset}", "--gpus=0", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[
                "ROCPROFSYS_SAMPLING_GPUS=0",
                "ROCPROFSYS_USE_AMD_SMI=true",
            ],
        )


@pytest.mark.timeout(60)
@pytest.mark.sampling
@pytest.mark.class_name("sample-sampling-target-flags")
class TestSampleSamplingTargetFlags(RocprofsysTest):
    def test_gpus_without_device(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-sample",
            run_args=["--gpus=0", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["ROCPROFSYS_SAMPLING_GPUS=0"])

    def test_cpus_without_host(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-sample",
            run_args=["--cpus=0-3", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["ROCPROFSYS_SAMPLING_CPUS=0-3"])

    def test_ai_nics_without_device(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-sample",
            run_args=["--ai-nics=nic0", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["ROCPROFSYS_SAMPLING_AINICS=nic0"])

    @pytest.mark.parametrize("preset", ["trace-hpc", "trace-gpu"])
    def test_preset_plus_gpus(self, preset):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-sample",
            run_args=[f"--preset={preset}", "--gpus=0", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[
                "ROCPROFSYS_SAMPLING_GPUS=0",
                "ROCPROFSYS_USE_AMD_SMI=true",
            ],
        )


# ============================================================================
# Export Config Tests
# ============================================================================


@pytest.mark.timeout(30)
@pytest.mark.class_name("export-config")
class TestExportConfig(RocprofsysTest):
    @pytest.mark.sys_run
    def test_export_run(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-run",
            run_args=["--preset=balanced", "--export-config"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=['"name": "balanced"'])

    @pytest.mark.sampling
    def test_export_sample(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-sample",
            run_args=["--preset=balanced", "--export-config"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=['"name": "balanced"'])


# ============================================================================
# List Presets and Explain Tests
# ============================================================================


@pytest.mark.timeout(30)
@pytest.mark.class_name("preset-discovery")
class TestPresetDiscovery(RocprofsysTest):
    @pytest.mark.sys_run
    def test_list_presets_run(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-run",
            run_args=["--list-presets"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["Available Presets:"])

    @pytest.mark.sampling
    def test_list_presets_sample(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-sample",
            run_args=["--list-presets"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["Available Presets:"])

    @pytest.mark.sys_run
    def test_explain_preset_run(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-run",
            run_args=["--explain=balanced"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["Preset: balanced"])

    @pytest.mark.sampling
    def test_explain_preset_sample(self):
        result = self.run_test(
            "baseline",
            target="rocprof-sys-sample",
            run_args=["--explain=balanced"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["Preset: balanced"])
