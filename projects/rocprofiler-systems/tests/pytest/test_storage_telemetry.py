# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Tests for hipFile GPU-direct storage I/O telemetry.

Runs the hipfile-test workload under rocprof-sys with ROCPROFSYS_USE_HIPFILE=ON
and validates that the hipFile per-GPU I/O counters appear in the ROCPD database
and Perfetto trace, as implemented by the hipFile PMC collector.

Every hipFile track is GPU-indexed. hipFile also maintains process-scoped
counters (file and buffer registrations), but those are not GPU measurements and
are deliberately not collected, so no assertion here refers to them.

The test is skipped automatically when rocprof-sys was not built with hipFile
support (ROCPROFSYS_USE_HIPFILE=OFF), because the hipfile-test binary is only
built in that configuration.
"""

from __future__ import annotations

import pytest
from pathlib import Path

from conftest import RocprofsysTest

pytestmark = [pytest.mark.gpu]


@pytest.fixture
def hipfile_env() -> dict[str, str]:
    """Environment enabling hipFile telemetry via process sampling."""
    return {
        "ROCPROFSYS_USE_HIPFILE": "ON",
        "ROCPROFSYS_USE_PROCESS_SAMPLING": "ON",
        # Sample frequently so the short-lived workload is captured many times.
        "ROCPROFSYS_PROCESS_SAMPLING_FREQ": "100",
        "ROCPROFSYS_ROCM_DOMAINS": "hip_runtime_api",
    }


@pytest.fixture
def hipfile_fallback_env(hipfile_env: dict[str, str]) -> dict[str, str]:
    """hipFile telemetry with the fastpath forced off."""
    return {**hipfile_env, "HIPFILE_FORCE_COMPAT_MODE": "1"}


@pytest.fixture
def hipfile_rules(validation_rules_dir: Path) -> list[Path]:
    """Validation rules for hipFile ROCPD output."""
    rules_dir = validation_rules_dir / "storage-telemetry"
    return [
        rules_dir / "validation-rules.json",
        rules_dir / "storage-telemetry-rules.json",
    ]


@pytest.mark.class_name("storage-telemetry")
class TestStorageTelemetry(RocprofsysTest):
    """Tests for hipFile I/O telemetry (Perfetto and ROCPD)."""

    @pytest.mark.timeout(180)
    @pytest.mark.parametrize(
        "mode", [pytest.param("sys_run", marks=pytest.mark.rocpd("hipfile_env"))]
    )
    def test_telemetry(self, mode, hipfile_env, hipfile_rules):
        """Run hipfile-test and validate hipFile counters in ROCPD + Perfetto."""
        workload_file = self.test_output_dir / "hipfile-test.bin"
        result = self.run_test(
            mode,
            "hipfile-test",
            env=hipfile_env,
            run_args=[str(workload_file), "0", "5"],
        )
        self.assert_regex(result)

        if mode == "sys_run":
            self.assert_rocpd(result, rules_files=hipfile_rules)
            self.assert_perfetto(
                result,
                counter_names=[
                    "hipFile GPU0 Read Bytes",
                    "hipFile GPU0 Write Bytes",
                    "hipFile GPU0 Read Ops",
                    "hipFile GPU0 Write Ops",
                    "hipFile GPU0 Read Bandwidth",
                    "hipFile GPU0 Write Bandwidth",
                ],
                subtest_name="Perfetto hipFile counter validation",
            )

    @pytest.mark.timeout(180)
    @pytest.mark.rocpd("hipfile_fallback_env")
    def test_telemetry_fallback_path(self, hipfile_fallback_env, hipfile_rules):
        """
        Force hipFile's POSIX fallback and assert the split reflects it.

        This is the only end-to-end assertion that can deterministically pin which
        path an operation took: the fastpath needs ext4-with-ordered-journaling or
        xfs plus O_DIRECT, none of which a test can guarantee on arbitrary CI
        storage. Compat mode is guaranteed in the other direction, so it is what
        proves the fastpath and fallback counters are not wired to the same source.
        The fastpath side stays covered by the collector unit tests.
        """
        workload_file = self.test_output_dir / "hipfile-test-fallback.bin"
        result = self.run_test(
            "sys_run",
            "hipfile-test",
            env=hipfile_fallback_env,
            run_args=[str(workload_file), "0", "5"],
        )
        self.assert_regex(result)

        self.assert_rocpd(
            result,
            rules_files=hipfile_rules,
            subtest_name="ROCPD hipFile fallback-path validation",
        )
        self.assert_perfetto(
            result,
            counter_names=[
                "hipFile GPU0 Fallback Reads",
                "hipFile GPU0 Fallback Writes",
            ],
            subtest_name="Perfetto hipFile fallback counter validation",
        )
