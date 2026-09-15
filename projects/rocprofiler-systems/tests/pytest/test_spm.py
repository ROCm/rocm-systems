# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""End-to-end tests for ROCm SPM Perfetto output."""

from __future__ import annotations

import pytest

from conftest import RocprofsysTest

pytestmark = [
    pytest.mark.spm,
    pytest.mark.gpu,
    pytest.mark.transpose,
]


@pytest.fixture(name="spm_perfetto_env")
def _spm_perfetto_env() -> dict[str, str]:
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
    """Validate SPM output and initialization-failure behavior."""

    @pytest.mark.parametrize(
        ("env_overrides", "expected_error"),
        [
            pytest.param(
                {"ROCPROFSYS_ROCM_SPM_SAMPLE_INTERVAL": "0"},
                "SPM counter collection requires a positive sample interval",
                id="missing-positive-interval",
            ),
            pytest.param(
                {"ROCPROFSYS_ROCM_EVENTS": "SQ_WAVES:device=0"},
                "SPM and ROCPROFSYS_ROCM_EVENTS cannot be enabled together",
                id="dispatch-counters",
                marks=pytest.mark.spm_available,
            ),
            pytest.param(
                {"ROCPROFSYS_GPU_PERF_COUNTERS": "SQ_WAVES"},
                "mutually exclusive with ROCPROFSYS_GPU_PERF_COUNTERS",
                id="gpu-device-counters",
                marks=pytest.mark.spm_available,
            ),
        ],
    )
    def test_invalid_configuration_exits_before_workload(
        self,
        spm_perfetto_env: dict[str, str],
        env_overrides: dict[str, str],
        expected_error: str,
    ) -> None:
        """Require fatal setup errors to reach both sinks without partial output."""
        log_file = self.test_output_dir / "fatal-initialization.log"
        environment = {
            **spm_perfetto_env,
            **env_overrides,
            "ROCPROFSYS_LOG_FILE": str(log_file),
        }

        result = self.run_test(
            "sys_run",
            "transpose",
            env=environment,
            check_target_arch=True,
            fail_on_pass=True,
        )

        assert result.returncode == 1
        assert expected_error in result.test_output
        assert "Runtime of transpose" not in result.test_output
        log_files = tuple(self.test_output_dir.glob("fatal-initialization_*.log"))
        assert len(log_files) == 1
        assert expected_error in log_files[0].read_text(encoding="utf-8")
        assert not tuple(result.output_dir.rglob("*.proto"))

    @pytest.mark.spm_available
    def test_sq_waves_trace(self, rocprof_config, gpu_info, spm_perfetto_env):
        """Collect SQ_WAVES and validate the resulting Perfetto counters."""
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
