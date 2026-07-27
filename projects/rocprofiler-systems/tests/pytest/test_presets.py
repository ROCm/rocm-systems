# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Preset, domain flag, and export-config CLI tests for the rocprof-sys-sample and rocprof-sys-run launchers.

These tests are parametrized over both exes, so each preset/domain/export-config
case is exercised against both front-ends.
"""

from __future__ import annotations
import json
import pytest
from conftest import RocprofsysTest

pytestmark = [pytest.mark.presets]

# Minimal but schema-valid preset used to exercise loading a preset from an
# explicit JSON file path.
CUSTOM_PRESET = {
    "metadata": {
        "name": "cli-test-custom",
        "description": "Custom sampling preset for CLI tests",
        "category": "general",
    },
    "sampling": {"enabled": True, "frequency_hz": {"value": 100}},
    "profiling": {"enabled": True},
}

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

# Verbosity convention for these tests:
#   -v 2  surfaces the resolved ROCPROFSYS_* env vars in output (needed when a
#         test asserts on a specific env var like ROCPROFSYS_USE_AMD_SMI).
#   -v 1  is enough for the pre-execution banner, deprecation warnings, and
#         composition notes (which print regardless of env echoing).

# Both launchers share the same preset/domain/flag surface; the embedded mark
# selects the matching runner category so a single test body covers run + sample.
TARGETS = [
    pytest.param("rocprof-sys-run", marks=pytest.mark.sys_run, id="run"),
    pytest.param("rocprof-sys-sample", marks=pytest.mark.sampling, id="sample"),
]


def _assert_baseline_output(
    test: RocprofsysTest,
    *,
    target: str,
    run_args: list[str],
    pass_regex: list[str],
) -> None:
    """Launch ``target`` in baseline mode and assert its output matches ``pass_regex``.

    Every test in this module shares one contract, run a launcher, require the
    command to be found, and match expected lines, so it lives in one place.
    """
    result = test.run_test(
        "baseline",
        target=target,
        run_args=run_args,
        fail_on_not_found=True,
    )
    test.assert_regex(result, pass_regex=pass_regex)


# ============================================================================
# Preset Tests (rocprof-sys-run + rocprof-sys-sample)
# ============================================================================
@pytest.mark.timeout(60)
@pytest.mark.class_name("presets")
@pytest.mark.parametrize("target", TARGETS)
class TestPresets(RocprofsysTest):
    @pytest.mark.parametrize("preset", PRESETS)
    def test(self, target, preset):
        _assert_baseline_output(
            self,
            target=target,
            run_args=[f"--preset={preset}", "-v", "2", "--", "ls"],
            pass_regex=[rf"Preset:\s+{preset}"],
        )

    @pytest.mark.timeout(60)
    @pytest.mark.gpu
    @pytest.mark.rocpd
    @pytest.mark.parametrize("preset", ROCPD_PRESETS)
    def test_rocpd(self, target, preset):
        _assert_baseline_output(
            self,
            target=target,
            run_args=[f"--preset={preset}", "-v", "2", "--", "ls"],
            pass_regex=[rf"Preset:\s+{preset}"],
        )

    def test_banner_reflects_sections(self, target):
        """The pre-execution banner reports tracing/profiling state from the preset.

        profile-only disables tracing but keeps (flat) profiling.
        """
        _assert_baseline_output(
            self,
            target=target,
            run_args=["--preset=profile-only", "-v", "1", "--", "ls"],
            pass_regex=[
                r"Tracing:\s+OFF",
                r"Profiling:\s+ON",
            ],
        )


# ============================================================================
# Legacy preset flag translation --preset=<name>
# ============================================================================


@pytest.mark.timeout(30)
@pytest.mark.class_name("legacy-presets")
@pytest.mark.parametrize("target", TARGETS)
class TestLegacyPresetFlags(RocprofsysTest):
    """Old-style flags like --balanced still work, but are now just shortcuts.

    Running --balanced is the same as running --preset=balanced. The tool
    accepts the old flag, prints a deprecation warning telling you to switch
    to --preset=<name>, and then applies that preset anyway.
    """

    @pytest.mark.parametrize("preset", PRESETS)
    def test_flag_translates_to_preset(self, target, preset):
        _assert_baseline_output(
            self,
            target=target,
            run_args=[f"--{preset}", "-v", "1", "--", "ls"],
            pass_regex=[
                rf"'--{preset}' is deprecated",
                rf"Use '--preset={preset}'",
                rf"Preset:\s+{preset}",
            ],
        )


# ============================================================================
# Domain Flag Tests (rocprof-sys-run + rocprof-sys-sample)
# ============================================================================


@pytest.mark.timeout(60)
@pytest.mark.class_name("domain-flags")
@pytest.mark.parametrize("target", TARGETS)
class TestDomainFlags(RocprofsysTest):
    def test_gpu(self, target):
        _assert_baseline_output(
            self,
            target=target,
            run_args=["--gpu", "-v", "2", "--", "ls"],
            pass_regex=["ROCPROFSYS_USE_AMD_SMI=true"],
        )

    def test_gpu_metrics(self, target):
        _assert_baseline_output(
            self,
            target=target,
            run_args=["--gpu=temp,power", "-v", "2", "--", "ls"],
            pass_regex=["ROCPROFSYS_AMD_SMI_METRICS=temp,power"],
        )

    def test_rocm(self, target):
        _assert_baseline_output(
            self,
            target=target,
            run_args=["--rocm=hip,kernel", "-v", "2", "--", "ls"],
            pass_regex=["ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch"],
        )

    def test_cpu(self, target):
        _assert_baseline_output(
            self,
            target=target,
            run_args=["--cpu=50", "-v", "2", "--", "ls"],
            pass_regex=["ROCPROFSYS_SAMPLING_FREQ=50"],
        )

    def test_parallel(self, target):
        _assert_baseline_output(
            self,
            target=target,
            run_args=["--parallel=mpi,openmp", "-v", "2", "--", "ls"],
            pass_regex=["ROCPROFSYS_USE_MPIP=true"],
        )

    def test_preset_plus_gpu(self, target):
        _assert_baseline_output(
            self,
            target=target,
            run_args=[
                "--preset=balanced",
                "--gpu=temp,power",
                "-v",
                "2",
                "--",
                "ls",
            ],
            pass_regex=["ROCPROFSYS_AMD_SMI_METRICS=temp,power"],
        )


# ============================================================================
# preset + domain flag interactions
# ============================================================================


@pytest.mark.timeout(60)
@pytest.mark.class_name("composition-notes")
@pytest.mark.parametrize("target", TARGETS)
class TestDomainFlagOverrideNotes(RocprofsysTest):
    """Advisory notes emitted by validate_domain_flags() when presets and domain
    flags are combined. These are guidance hints printed to stderr, not errors —
    the workload still runs."""

    def test_cpu_with_no_sampling_preset(self, target):
        """--cpu with a preset that disables CPU sampling warns it will be overridden."""
        _assert_baseline_output(
            self,
            target=target,
            run_args=["--preset=trace-hpc", "--cpu=100", "-v", "1", "--", "ls"],
            pass_regex=[
                r"--cpu flag used with 'trace-hpc' preset which disables CPU sampling",
                r"will override the preset's sampling settings",
            ],
        )

    def test_multiple_domains_without_preset(self, target):
        """Three or more domain flags without a preset suggests using one."""
        _assert_baseline_output(
            self,
            target=target,
            run_args=["--gpu", "--rocm", "--cpu", "-v", "1", "--", "ls"],
            pass_regex=[
                r"Multiple domain flags specified",
                r"--preset=detailed",
            ],
        )

    def test_rocm_without_gpu(self, target):
        """--rocm without --gpu suggests adding --gpu for GPU metrics."""
        _assert_baseline_output(
            self,
            target=target,
            run_args=["--rocm=hip", "-v", "1", "--", "ls"],
            pass_regex=[
                r"--rocm enables ROCm API tracing",
                r"adding --gpu for GPU metrics",
            ],
        )

    def test_parallel_without_rocm(self, target):
        """--parallel without --rocm suggests adding --rocm for collective tracing."""
        _assert_baseline_output(
            self,
            target=target,
            run_args=["--parallel=mpi", "-v", "1", "--", "ls"],
            pass_regex=[
                r"--parallel enables MPI/OpenMP profiling",
                r"adding --rocm for GPU collective tracing",
            ],
        )


# ============================================================================
# Preset-from-file Tests (rocprof-sys-run + rocprof-sys-sample)
# ============================================================================


@pytest.mark.timeout(60)
@pytest.mark.class_name("preset-file")
@pytest.mark.parametrize("target", TARGETS)
class TestPresetFile(RocprofsysTest):
    """--preset accepting an explicit JSON file path (valid / malformed / missing).

    Both launchers share the same preset-loading path, so each case is exercised
    against rocprof-sys-run and rocprof-sys-sample.
    """

    def test_custom_json(self, target, tmp_path):
        """A valid custom JSON preset loads and runs the workload."""
        preset_file = tmp_path / "custom.json"
        preset_file.write_text(json.dumps(CUSTOM_PRESET))
        _assert_baseline_output(
            self,
            target=target,
            run_args=[f"--preset={preset_file}", "-v", "2", "--", "ls"],
            pass_regex=[
                r"Preset:\s+\S*custom\.json",
                r"Custom sampling preset for CLI tests",
            ],
        )

    def test_without_metadata_name_uses_filepath(self, target, tmp_path):
        """A valid preset file lacking metadata.name loads; the name falls back to the path."""
        preset_file = tmp_path / "noname.json"
        preset_file.write_text(
            json.dumps(
                {
                    "sampling": {"enabled": True, "frequency_hz": {"value": 100}},
                    "profiling": {"enabled": True},
                }
            )
        )
        result = self.run_test(
            "baseline",
            target=target,
            run_args=[f"--preset={preset_file}", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[r"Preset:\s+\S*noname\.json"],
            fail_regex=[r"Could not load preset", r"Failed to parse preset"],
        )

    def test_malformed_json_warns(self, target, tmp_path):
        """A malformed preset file warns and degrades gracefully (still runs)."""
        preset_file = tmp_path / "malformed.json"
        preset_file.write_text("{ this is not valid json ")
        _assert_baseline_output(
            self,
            target=target,
            run_args=[f"--preset={preset_file}", "-v", "2", "--", "ls"],
            pass_regex=[
                r"Failed to parse preset",
                r"Could not load preset",
            ],
        )

    def test_missing_json_warns(self, target, tmp_path):
        """A non-existent preset file warns and degrades gracefully (still runs)."""
        preset_file = tmp_path / "does_not_exist.json"
        _assert_baseline_output(
            self,
            target=target,
            run_args=[f"--preset={preset_file}", "-v", "2", "--", "ls"],
            pass_regex=[
                r"Could not load preset",
                r"Check preset name or file path",
            ],
        )

    def test_unknown_name_warns(self, target):
        """An unknown built-in preset name warns and degrades gracefully (still runs)."""
        _assert_baseline_output(
            self,
            target=target,
            run_args=["--preset=unknown-name", "-v", "2", "--", "ls"],
            pass_regex=[
                r"Could not load preset 'unknown-name'",
                r"Check preset name or file path",
            ],
        )

    def test_absolute_path(self, target, tmp_path):
        """A valid preset given as an absolute file path loads and runs."""
        preset_file = tmp_path / "abs.json"
        preset_file.write_text(json.dumps(CUSTOM_PRESET))
        _assert_baseline_output(
            self,
            target=target,
            run_args=[f"--preset={preset_file.resolve()}", "-v", "2", "--", "ls"],
            pass_regex=[
                r"Preset:\s+/\S*abs\.json",
                r"Custom sampling preset for CLI tests",
            ],
        )

    def test_path_traversal_rejected(self, target):
        """A preset path containing '..' is rejected (path-traversal guard); app still runs."""
        _assert_baseline_output(
            self,
            target=target,
            run_args=["--preset=../etc/passwd.json", "-v", "2", "--", "ls"],
            pass_regex=[
                r"contains '\.\.'\. Ignoring",
                r"Could not load preset",
            ],
        )


# ============================================================================
# Argument-parsing edge cases for --preset
# ============================================================================


@pytest.mark.timeout(30)
@pytest.mark.class_name("preset-arg-edge-case")
@pytest.mark.parametrize("target", TARGETS)
class TestPresetArgEdgeCases(RocprofsysTest):
    """How --preset handles empty, whitespace, and duplicate values."""

    def test_empty_is_noop(self, target):
        """An empty --preset= value is ignored: no preset loaded, no warning, app runs."""
        # Use a runtime-computed sentinel: the launcher's "Executing '...'" line
        # echoes the unexpanded command (literal "$((2+3))"), so only the
        # actually-executed shell can produce "NOOP_5_OK". This prevents the
        # assertion from matching the launcher's command echo instead of real output.
        result = self.run_test(
            "baseline",
            target=target,
            run_args=["--preset=", "--", "sh", "-c", "echo NOOP_$((2+3))_OK"],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[r"NOOP_5_OK"],
            fail_regex=[r"Could not load preset", r"Failed to parse preset"],
        )

    def test_whitespace_warns(self, target):
        """A whitespace-only value is treated as a bad name and degrades gracefully."""
        _assert_baseline_output(
            self,
            target=target,
            run_args=["--preset= ", "-v", "1", "--", "ls"],
            pass_regex=[
                r"Could not load preset",
                r"Check preset name or file path",
            ],
        )

    def test_duplicate_uses_first(self, target):
        """When --preset is given twice the first value wins (no hard error)."""
        _assert_baseline_output(
            self,
            target=target,
            run_args=[
                "--preset=balanced",
                "--preset=detailed",
                "-v",
                "1",
                "--",
                "ls",
            ],
            pass_regex=[r"Preset:\s+balanced"],
        )


# ============================================================================
# Export Config Tests
# ============================================================================
@pytest.mark.timeout(30)
@pytest.mark.class_name("export-config")
@pytest.mark.parametrize("target", TARGETS)
class TestExportConfig(RocprofsysTest):
    """--export-config behavior, exercised against rocprof-sys-run and rocprof-sys-sample."""

    def test_stdout(self, target):
        """--export-config prints the resolved preset config as JSON to stdout."""
        _assert_baseline_output(
            self,
            target=target,
            run_args=["--preset=balanced", "--export-config"],
            pass_regex=['"name": "balanced"'],
        )

    def test_to_file(self, target, tmp_path):
        """--export-config=FILE writes parseable JSON to the given path."""
        out_file = tmp_path / "exported.json"
        result = self.run_test(
            "baseline",
            target=target,
            run_args=[f"--export-config={out_file}", "--preset=balanced", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=[r"Configuration exported to:"])

        assert out_file.is_file(), f"export-config did not create {out_file}"
        data = json.loads(out_file.read_text())
        assert data.get("metadata", {}).get("name") == "balanced"

    def test_does_not_run_app(self, target):
        """--export-config prints config and exits without launching the target."""
        result = self.run_test(
            "baseline",
            target=target,
            run_args=[
                "--preset=balanced",
                "--export-config",
                "--",
                "/usr/bin/echo",
                "ROCPROFSYS_APP_DID_RUN",
            ],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=['"name": "balanced"'],
            fail_regex=[r"ROCPROFSYS_APP_DID_RUN"],
        )


# ============================================================================
# Sampling target flags (--gpus / --cpus / --ai-nics) stand alone:
# they must not require --device / --host on the command line, so presets
# that already enable the matching backend (e.g. trace-hpc, trace-gpu set
# ROCPROFSYS_USE_AMD_SMI=true) can compose with them.
# ============================================================================


@pytest.mark.timeout(60)
@pytest.mark.class_name("sampling-target-flags")
@pytest.mark.parametrize("target", TARGETS)
class TestSamplingTargetFlags(RocprofsysTest):
    def test_gpus_without_device(self, target):
        result = self.run_test(
            "baseline",
            target=target,
            run_args=["--gpus=0", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["ROCPROFSYS_SAMPLING_GPUS=0"])

    def test_cpus_without_host(self, target):
        result = self.run_test(
            "baseline",
            target=target,
            run_args=["--cpus=0-3", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["ROCPROFSYS_SAMPLING_CPUS=0-3"])

    def test_ai_nics_without_device(self, target):
        result = self.run_test(
            "baseline",
            target=target,
            run_args=["--ai-nics=nic0", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=["ROCPROFSYS_SAMPLING_AINICS=nic0"])

    @pytest.mark.parametrize("preset", ["trace-hpc", "trace-gpu"])
    def test_preset_plus_gpus(self, target, preset):
        result = self.run_test(
            "baseline",
            target=target,
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
