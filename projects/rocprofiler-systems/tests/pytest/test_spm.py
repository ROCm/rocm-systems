# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""End-to-end tests for ROCm SPM Perfetto output."""

from __future__ import annotations

import pytest

from conftest import RocprofsysTest

pytestmark = [
    pytest.mark.spm,
    pytest.mark.spm_available,
    pytest.mark.gpu,
    pytest.mark.transpose,
]


@pytest.fixture
def spm_perfetto_env() -> dict[str, str]:
    """Environment for a bounded SPM Perfetto validation run."""
    return {
        "ROCPROFSYS_TRACE": "ON",
        "ROCPROFSYS_PROFILE": "OFF",
        "ROCPROFSYS_USE_SAMPLING": "OFF",
        "ROCPROFSYS_USE_PROCESS_SAMPLING": "OFF",
        "ROCPROFSYS_USE_KOKKOSP": "OFF",
        "ROCPROFILER_SPM_BETA_ENABLED": "ON",
        "ROCPROFSYS_ROCM_SPM_EVENTS": "SQ_WAVES",
        # Matches the documented example interval and is an exact multiple of
        # the 32-cycle hardware granularity.
        "ROCPROFSYS_ROCM_SPM_SAMPLE_INTERVAL": "8192",
    }


@pytest.mark.timeout(240)
@pytest.mark.class_name("spm-perfetto")
class TestSPMPerfetto(RocprofsysTest):
    """Validate that SPM emits SQ_WAVES samples to Perfetto when supported."""

    def test_sq_waves_trace(self, rocprof_config, gpu_info, spm_perfetto_env):
        if rocprof_config.capabilities.is_ci and any(
            arch.startswith("gfx95") for arch in gpu_info.architectures
        ):
            pytest.skip(
                "SDK SPM collection currently times out on gfx95 CI runners "
                "with passthrough virtualization"
            )

        result = self.run_test(
            "sys_run",
            "transpose",
            env=spm_perfetto_env,
            check_target_arch=True,
        )
        self.assert_regex(result)

        self.assert_perfetto(
            result,
            subtest_name="Perfetto SPM SQ_WAVES counter validation",
            counter_names=["GPU SPM SQ_WAVES"],
            print_output=True,
        )
