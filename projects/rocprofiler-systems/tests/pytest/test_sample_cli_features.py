# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
CLI + GPU artifact tests for rocprof-sys-sample flags whose env/config
equivalents live elsewhere:

  --selected-regions  -> test_selective_regions.py
      that module sets ROCPROFSYS_SELECTED_REGIONS; here we pass
      --selected-regions on the sample command line and check the trace.

  --sample-realtime   -> test_transpose.py
      that module uses ROCPROFSYS_SAMPLING_REALTIME env vars; here we pass
      --sample-realtime and confirm realtime sampling (no "Defaulting to cputime").

  -G / --gpu-events   -> test_transpose.py
      that module sets ROCPROFSYS_ROCM_EVENTS; here we pass -G and validate
      counter tracks in Perfetto and ROCpd on transpose.

  --gpu=temp,power    -> test_roctx.py, test_presets.py
      test_presets only checks -v2 config echo with ls; test_roctx drives AMD-SMI
      via env on roctx. Here we pass --gpu=temp,power and check Temperature/Power
      in the trace and ROCpd on a real GPU workload.

Flags go in sampling_args (before --). Each case checks Perfetto, ROCpd,
and/or timemory output, not just that the flag parses or sets an env var.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import RocprofsysTest

pytestmark = [
    pytest.mark.sample_cli_features,
    pytest.mark.gpu,
    pytest.mark.timeout(120),
]

_TRANSPOSE_ARGS = ["2", "100", "50"]


@pytest.fixture
def selective_region_env() -> dict[str, str]:
    return {
        "ROCPROFSYS_ROCM_DOMAINS": "hip_runtime_api,marker_api,kernel_dispatch,marker_core_range_api",
    }


@pytest.fixture
def rocpd_env() -> dict[str, str]:
    return {}


@pytest.fixture
def hw_counter_rules(validation_rules_dir: Path) -> list[Path]:
    return [
        validation_rules_dir / "default-rules.json",
        validation_rules_dir / "transpose" / "hw-counter-rules.json",
    ]


@pytest.fixture
def amd_smi_rules(validation_rules_dir: Path) -> list[Path]:
    rules_dir = validation_rules_dir / "roctx"
    return [
        validation_rules_dir / "default-rules.json",
        rules_dir / "amd-smi-rules.json",
    ]


_SELECTED_REGION_CASES = [
    pytest.param(
        "Region1",
        ["CodeBlock_B", "CodeBlock_C", "CodeBlock_D", "CodeBlock_F"],
        ["CodeBlock_A", "CodeBlock_E", "CodeBlock_G"],
        ["Region1", "Region2"],
        ["Region3"],
        id="region1",
    ),
    pytest.param(
        "Region2,Region3",
        ["CodeBlock_C", "CodeBlock_E"],
        ["CodeBlock_A", "CodeBlock_B", "CodeBlock_D", "CodeBlock_F", "CodeBlock_G"],
        ["Region2", "Region3"],
        ["Region1"],
        id="region2-3",
    ),
]


@pytest.mark.sampling
@pytest.mark.class_name("sample-selected-regions")
class TestSampleSelectedRegions(RocprofsysTest):
    """--selected-regions filters tracing to named roctx regions."""

    @pytest.mark.parametrize(
        "regions, kernels_present, kernels_absent, markers_present, markers_absent",
        _SELECTED_REGION_CASES,
    )
    def test(
        self,
        regions,
        kernels_present,
        kernels_absent,
        markers_present,
        markers_absent,
        selective_region_env,
    ):
        result = self.run_test(
            "sampling",
            target="selective_region",
            sampling_args=[f"--selected-regions={regions}"],
            env=selective_region_env,
            check_target_arch=True,
        )
        self.assert_regex(result)
        self.assert_perfetto(
            result,
            subtest_name="Filtered kernels",
            categories=["rocm_hip_stream"],
            pass_regex=kernels_present,
            fail_regex=kernels_absent,
        )
        self.assert_perfetto(
            result,
            subtest_name="Filtered markers",
            categories=["rocm_marker_api"],
            pass_regex=markers_present,
            fail_regex=markers_absent,
        )

    def test_unknown_region(self, selective_region_env):
        """A region name that matches nothing still runs but traces no region kernels."""
        result = self.run_test(
            "sampling",
            target="selective_region",
            sampling_args=["--selected-regions=NoSuch"],
            env=selective_region_env,
            check_target_arch=True,
        )
        self.assert_regex(result)
        self.assert_perfetto(
            result,
            subtest_name="No region kernels traced",
            categories=["rocm_hip_stream"],
            fail_regex=["CodeBlock_B", "CodeBlock_C", "CodeBlock_E", "CodeBlock_F"],
        )
        self.assert_perfetto(
            result,
            subtest_name="No regions traced",
            categories=["rocm_marker_api"],
            fail_regex=["Region1", "Region2", "Region3"],
        )


@pytest.mark.sampling
@pytest.mark.class_name("sample-realtime")
class TestSampleRealtime(RocprofsysTest):
    """--sample-realtime drives real-clock timer sampling on a GPU workload."""

    def test(self):
        result = self.run_test(
            "sampling",
            target="transpose",
            sampling_args=["--sample-realtime", "300", "0.05"],
            run_args=_TRANSPOSE_ARGS,
            check_target_arch=True,
        )
        # Absence of this warning means --sample-realtime was honored (see tool_runner).
        self.assert_regex(result, fail_regex=["Defaulting to cputime"])
        self.assert_perfetto(result, categories=["hip_runtime_api"])
        self.assert_timemory(result, file_name="wall_clock.json", metric="wall_clock")


@pytest.mark.sampling
@pytest.mark.class_name("sample-amd-smi")
class TestSampleAmdSmiMetrics(RocprofsysTest):
    """--gpu=temp,power records AMD-SMI GPU metrics on a GPU workload."""

    @pytest.mark.rocpd("rocpd_env")
    def test(self, rocpd_env, amd_smi_rules):
        result = self.run_test(
            "sampling",
            target="roctx",
            sampling_args=[
                "--gpu=temp,power",
                "--process-freq",
                "1000",
                "--process-wait",
                "0.0",
                "--process-duration",
                "10",
            ],
            env=rocpd_env,
            check_target_arch=True,
        )
        self.assert_regex(result)
        self.assert_perfetto(
            result,
            subtest_name="Perfetto AMD-SMI metric validation",
            counter_names=["Temperature", "Power"],
        )
        self.assert_rocpd(
            result,
            subtest_name="ROCpd AMD-SMI metric validation",
            rules_files=amd_smi_rules,
        )
        self.assert_timemory(result, file_name="wall_clock.json", metric="wall_clock")


@pytest.mark.sampling
@pytest.mark.rocprofiler
@pytest.mark.class_name("sample-gpu-events")
class TestSampleGpuEvents(RocprofsysTest):
    """-G/--gpu-events records GPU hardware counters on a GPU workload."""

    @pytest.mark.rocpd("rocpd_env")
    def test(self, rocpd_env, gpu_info, hw_counter_rules):
        result = self.run_test(
            "sampling",
            target="transpose",
            sampling_args=["-G", gpu_info.rocm_events_for_test],
            run_args=_TRANSPOSE_ARGS,
            env=rocpd_env,
            check_target_arch=True,
        )
        self.assert_regex(result)
        self.assert_perfetto(
            result,
            subtest_name="Perfetto counter validation",
            counter_names=gpu_info.counter_names,
            check_counter_pairing=True,
        )
        self.assert_rocpd(
            result,
            subtest_name="RocPD HW counter validation",
            rules_files=hw_counter_rules,
        )
