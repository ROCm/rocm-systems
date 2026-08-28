# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Tests for the hipfile examples: API tracing (`hipfile-trace`) and GPU-direct
storage I/O telemetry (`hipfile-io`).
"""

from __future__ import annotations

import subprocess

import pytest
from pathlib import Path

from conftest import RocprofsysTest
from rocprofsys import RocprofsysConfig

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.hipfile,
]

# Settings registered only when hipFile support is compiled in
# (ROCPROFSYS_HIPFILE_SUPPORT; see cmake/Packages.cmake and the guarded
# ROCPROFSYS_CONFIG_SETTING blocks in config.cpp).
HIPFILE_SETTINGS = [
    "ROCPROFSYS_USE_HIPFILE",
    "ROCPROFSYS_HIPFILE_METRICS",
]


# =============================================================================
# hipFILE API tracing (`hipfile-trace`)
# =============================================================================


@pytest.fixture
def hipfile_env() -> dict[str, str]:
    """Environment variables for hipFILE trace tests."""
    return {
        "ROCPROFSYS_ROCM_DOMAINS": "hip_runtime_api,kernel_dispatch,memory_copy,hipfile_api",
        "ROCPROFSYS_SAMPLING_CPUS": "none",
    }


@pytest.fixture
def hipfile_rules(validation_rules_dir) -> list[Path]:
    """Get validation rules for hipFILE trace tests."""
    rules_dir = validation_rules_dir / "hipfile"
    return [
        rules_dir / "validation-rules.json",
        rules_dir / "sdk-metrics-rules.json",
    ]


@pytest.mark.timeout(120)
@pytest.mark.rocm
# hipFILE callback tracing domain requires rocprofiler-sdk >= 1.3.5
@pytest.mark.rocprofiler_sdk_min_version("1.3.5")
@pytest.mark.parametrize(
    "mode",
    [
        pytest.param("sampling", marks=pytest.mark.rocpd("hipfile_env")),
        "sys_run",
    ],
)
@pytest.mark.class_name("hipfile")
class TestHipFile(RocprofsysTest):
    def test(self, mode, hipfile_env, hipfile_rules):
        result = self.run_test(
            mode,
            "hipfile-trace",
            env=hipfile_env,
        )
        self.assert_regex(result)

        if mode == "sampling":
            self.assert_perfetto(
                result,
                categories=["rocm_hipfile_api"],
                labels=["hipFileGetVersion"],
                counts=[1],
                depths=[1],
            )
            self.assert_rocpd(result, rules_files=hipfile_rules)


# =============================================================================
# hipFile I/O telemetry (`hipfile-io`)
# =============================================================================


@pytest.fixture
def hipfile_telemetry_env() -> dict[str, str]:
    """Environment enabling hipFile telemetry via process sampling."""
    return {
        "ROCPROFSYS_USE_HIPFILE": "ON",
        "ROCPROFSYS_USE_PROCESS_SAMPLING": "ON",
        # Sample frequently so the short-lived workload is captured many times.
        "ROCPROFSYS_PROCESS_SAMPLING_FREQ": "100",
        "ROCPROFSYS_ROCM_DOMAINS": "hip_runtime_api",
        # The validation rules cover the whole metric surface, including the operation
        # counters that the default selection leaves off.
        "ROCPROFSYS_HIPFILE_METRICS": "all",
    }


@pytest.fixture
def hipfile_fallback_env(hipfile_telemetry_env: dict[str, str]) -> dict[str, str]:
    """
    hipFile telemetry with the fastpath forced off.

    HIPFILE_FORCE_COMPAT_MODE is hipFile's own variable, read by libhipfile rather
    than by rocprofiler-systems. It is not part of the ROCPROFSYS_* surface, does not
    appear in rocprof-sys-avail, and its spelling and semantics are owned upstream, so
    a hipFile release could change it without any signal on this side. It routes every
    operation through the POSIX fallback backend, which is the only way a test can pin
    down which path an operation took, since the fastpath additionally requires xfs or
    ext4 with ordered journaling plus O_DIRECT.
    """
    return {**hipfile_telemetry_env, "HIPFILE_FORCE_COMPAT_MODE": "true"}


@pytest.fixture
def hipfile_default_metrics_env(hipfile_telemetry_env: dict[str, str]) -> dict[str, str]:
    """hipFile telemetry with ROCPROFSYS_HIPFILE_METRICS left unset."""
    env = dict(hipfile_telemetry_env)
    env.pop("ROCPROFSYS_HIPFILE_METRICS", None)
    return env


@pytest.fixture
def hipfile_telemetry_rules(validation_rules_dir: Path) -> list[Path]:
    """Validation rules for hipFile ROCPD output."""
    rules_dir = validation_rules_dir / "hipfile-telemetry"
    return [
        rules_dir / "validation-rules.json",
        rules_dir / "hipfile-telemetry-rules.json",
    ]


@pytest.fixture
def require_hipfile_io(rocprof_config: RocprofsysConfig) -> None:
    """Skip when the hipfile-io example binary was not built."""
    try:
        rocprof_config.get_target_executable("hipfile-io")
    except FileNotFoundError:
        pytest.skip(
            "hipfile-io example not found — hipFile runtime not available at configure time"
        )


@pytest.fixture
def require_hipfile_collector(rocprof_config: RocprofsysConfig) -> None:
    """Skip runtime telemetry tests when the hipFile collector was not compiled in."""
    result = subprocess.run(
        [str(rocprof_config.rocprofsys_avail), "--settings"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode != 0:
        pytest.skip(f"rocprof-sys-avail failed: {result.stderr}")

    missing = [s for s in HIPFILE_SETTINGS if s not in result.stdout]
    if missing:
        pytest.skip(
            "hipFile collector not compiled in (ROCPROFSYS_BUILD_HIPFILE=OFF, "
            "or AUTO with no new enough package) — "
            f"missing settings: {missing}"
        )


@pytest.mark.usefixtures("require_hipfile_io", "require_hipfile_collector")
@pytest.mark.class_name("hipfile-io")
class TestHipFileTelemetry(RocprofsysTest):
    """Tests for hipFile I/O telemetry (Perfetto and ROCPD).

    ``hipfile-io`` can exist without the in-tree collector (older/basic hipFile
    at configure time). Both fixtures are required so that configuration skips
    rather than failing on missing ``ROCPROFSYS_*HIPFILE*`` settings.
    """

    @pytest.mark.timeout(30)
    def test_settings_present(self, rocprof_config: RocprofsysConfig):
        """hipFile settings must be listed by ``rocprof-sys-avail --settings``.

        These settings are only registered when hipFile support is compiled in.
        No GPU-direct storage hardware is required. The class-level collector
        fixture skips when they are absent; this asserts both names appear.
        """
        result = subprocess.run(
            [str(rocprof_config.rocprofsys_avail), "--settings"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        assert result.returncode == 0, f"rocprof-sys-avail failed: {result.stderr}"

        settings = result.stdout
        missing = [s for s in HIPFILE_SETTINGS if s not in settings]
        assert not missing, (
            "hipFile settings not reported by rocprof-sys-avail --settings — "
            "was hipFile support compiled in?\n"
            f"Missing: {missing}"
        )

    @pytest.mark.timeout(180)
    @pytest.mark.rocpd("hipfile_telemetry_env")
    def test_telemetry(self, hipfile_telemetry_env, hipfile_telemetry_rules):
        """Run hipfile-io and validate hipFile counters in ROCPD + Perfetto."""
        workload_file = self.test_output_dir / "hipfile-io.bin"
        result = self.run_test(
            "sys_run",
            "hipfile-io",
            env=hipfile_telemetry_env,
            run_args=[str(workload_file), "0", "5"],
        )
        self.assert_regex(result)

        self.assert_rocpd(result, rules_files=hipfile_telemetry_rules)
        self.assert_perfetto(
            result,
            counter_names=[
                "GPU [0] Storage Read Bytes (S)",
                "GPU [0] Storage Write Bytes (S)",
                "GPU [0] Storage Read Ops (S)",
                "GPU [0] Storage Write Ops (S)",
                "GPU [0] Storage Read Bandwidth (S)",
                "GPU [0] Storage Write Bandwidth (S)",
            ],
            subtest_name="Perfetto hipFile counter validation",
        )

    @pytest.mark.timeout(180)
    @pytest.mark.rocpd("hipfile_default_metrics_env")
    def test_default_metric_selection(self, hipfile_default_metrics_env):
        """
        Leave ROCPROFSYS_HIPFILE_METRICS unset and check what a user gets for free.

        The default registered for the setting is ``fastpath, fallback, bandwidth, bytes,
        errors``, so those ten tracks must be present without any opt-in. Ops and
        unaligned are not part of that default; this test does not assert their
        absence.

        The fastpath, fallback, and error counters are asserted on presence rather than value:
        which path pair is non-zero is a property of the storage under the test, not of the
        collector. Requiring a value from either one would make this test pass or fail based
        on the filesystem backing the build directory.
        """
        workload_file = self.test_output_dir / "hipfile-io-defaults.bin"
        result = self.run_test(
            "sys_run",
            "hipfile-io",
            env=hipfile_default_metrics_env,
            run_args=[str(workload_file), "0", "5"],
        )
        self.assert_regex(result)

        self.assert_perfetto(
            result,
            counter_names=[
                "GPU [0] Storage Read Bytes (S)",
                "GPU [0] Storage Write Bytes (S)",
                "GPU [0] Storage Read Bandwidth (S)",
                "GPU [0] Storage Write Bandwidth (S)",
            ],
            counter_names_present=[
                "GPU [0] Storage Fastpath Reads (S)",
                "GPU [0] Storage Fastpath Writes (S)",
                "GPU [0] Storage Fallback Reads (S)",
                "GPU [0] Storage Fallback Writes (S)",
                "GPU [0] Storage Read Errors (S)",
                "GPU [0] Storage Write Errors (S)",
            ],
            subtest_name="Perfetto hipFile default metric selection",
        )

    @pytest.mark.timeout(180)
    @pytest.mark.rocpd("hipfile_fallback_env")
    def test_telemetry_fallback_path(self, hipfile_fallback_env, hipfile_telemetry_rules):
        """
        Force hipFile's POSIX fallback and assert the split reflects it.

        This is the only end-to-end assertion that can deterministically pin which
        path an operation took: the fastpath needs ext4-with-ordered-journaling or
        xfs plus O_DIRECT, none of which a test can guarantee on arbitrary CI
        storage. Compat mode is guaranteed in the other direction.

        The two directions are asserted together on purpose. Requiring only that the
        fallback counters move would still pass if both counters were wired to the
        same source; pairing it with fastpath held at exactly zero is what shows they
        are read from different fields. Zero is only safe to assert here, because
        compat mode makes it independent of the filesystem under the test.
        """
        workload_file = self.test_output_dir / "hipfile-io-fallback.bin"
        result = self.run_test(
            "sys_run",
            "hipfile-io",
            env=hipfile_fallback_env,
            run_args=[str(workload_file), "0", "5"],
        )
        self.assert_regex(result)

        self.assert_rocpd(
            result,
            rules_files=hipfile_telemetry_rules,
            subtest_name="ROCPD hipFile fallback-path validation",
        )
        self.assert_perfetto(
            result,
            counter_names=[
                "GPU [0] Storage Fallback Reads (S)",
                "GPU [0] Storage Fallback Writes (S)",
            ],
            counter_names_zero=[
                "GPU [0] Storage Fastpath Reads (S)",
                "GPU [0] Storage Fastpath Writes (S)",
            ],
            subtest_name="Perfetto hipFile fallback counter validation",
        )
