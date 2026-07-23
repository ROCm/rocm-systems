# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Regression tests for the shared tool_runner used by rocprof-sys-run and
rocprof-sys-sample. Each test pins behavior that previously broke or that the
unification refactor consolidated; failures here indicate a regression in
tool_runner, argparse env-update semantics, or the conflict-detection path.

The classes further down add plain CLI flag -> env var checks for config/output,
trace/profile, host/device, and wait/duration/periods flags. Both binaries parse
args through the same tool_runner path, so each case is parametrized over both
instead of written twice.
"""

from __future__ import annotations
import pytest
from conftest import RocprofsysTest

pytestmark = [pytest.mark.presets]

TARGETS = [
    pytest.param("rocprof-sys-run", marks=pytest.mark.sys_run, id="run"),
    pytest.param("rocprof-sys-sample", marks=pytest.mark.sampling, id="sample"),
]


# ============================================================================
# update_env(REPLACE) must not leave duplicate KEY= entries
# ----------------------------------------------------------------------------
# Regression for the bug fixed in this refactor: a shell-exported value plus a
# preset that REPLACEs the same key used to produce "KEY=preset_value:shell_value"
# after consolidate_env_entries joined the two surviving entries. The original
# reproducer was ROCPROFSYS_TRACE=true + --preset=profile-only, which silently
# turned tracing back on under "profile only".
# ============================================================================


@pytest.mark.timeout(30)
@pytest.mark.class_name("tool-runner-replace-env")
class TestReplaceEnvNoDuplicates(RocprofsysTest):
    @pytest.mark.parametrize("target", TARGETS)
    def test_preset_replaces_shell_env(self, target):
        result = self.run_test(
            "baseline",
            target=target,
            env={"ROCPROFSYS_TRACE": "true"},
            run_args=["--preset=profile-only", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[r"ROCPROFSYS_TRACE=false(?![^\n]*:)"],
            fail_regex=[r"ROCPROFSYS_TRACE=\S*:\S*"],
        )


# ============================================================================
# --profile and --flat-profile must not be accepted together
# ----------------------------------------------------------------------------
# argparse declares the conflict via .conflicts({"flat-profile"}); tool_runner's
# apply_post_parse keeps a defensive throw as a second line of defense. Either
# layer must reject the combination with a non-zero exit code and a message
# mentioning the conflict.
# ============================================================================


@pytest.mark.timeout(30)
@pytest.mark.class_name("tool-runner-profile-conflict")
class TestProfileFlatProfileConflict(RocprofsysTest):
    @pytest.mark.parametrize("target", TARGETS)
    def test_profile_and_flat_profile_rejected(self, target):
        result = self.run_test(
            "baseline",
            target=target,
            run_args=["--profile", "--flat-profile", "--", "ls"],
            fail_on_not_found=True,
            fail_on_pass=True,
        )
        self.assert_regex(
            result,
            pass_regex=[r"--profile.*conflicts.*--flat-profile"],
        )


# ============================================================================
# --output-format selects backends authoritatively
# ----------------------------------------------------------------------------
# --output-format names the exact set of outputs to produce: listed formats are
# enabled and every unlisted backend is explicitly disabled. A lone "rocpd" must
# not leave tracing on through the perfetto/profile default derivation, where
# ROCPROFSYS_TRACE otherwise defaults to the negation of ROCPROFSYS_PROFILE.
# ============================================================================


@pytest.mark.timeout(30)
@pytest.mark.class_name("tool-runner-output-format")
class TestOutputFormatSelection(RocprofsysTest):
    @pytest.mark.parametrize("target", TARGETS)
    def test_proto_rocpd_enables_both(self, target):
        result = self.run_test(
            "baseline",
            target=target,
            run_args=["--output-format", "proto", "rocpd", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[
                r"ROCPROFSYS_TRACE=true",
                r"ROCPROFSYS_USE_ROCPD=true",
                r"ROCPROFSYS_PROFILE=false",
            ],
        )

    @pytest.mark.parametrize("target", TARGETS)
    def test_rocpd_only_disables_perfetto(self, target):
        result = self.run_test(
            "baseline",
            target=target,
            run_args=["--output-format", "rocpd", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[
                r"ROCPROFSYS_USE_ROCPD=true",
                r"ROCPROFSYS_TRACE=false",
                r"ROCPROFSYS_PROFILE=false",
            ],
        )

    @pytest.mark.parametrize(
        "legacy_args",
        [
            ["--trace"],
            ["--profile"],
            ["--flat-profile"],
            ["--profile-format", "text"],
        ],
    )
    @pytest.mark.parametrize("target", TARGETS)
    def test_conflicts_with_legacy_flags(self, target, legacy_args):
        result = self.run_test(
            "baseline",
            target=target,
            run_args=["--output-format", "rocpd", *legacy_args, "--", "ls"],
            fail_on_not_found=True,
            fail_on_pass=True,
        )
        self.assert_regex(
            result,
            pass_regex=[r"--output-format.*conflicts.*"],
        )


# =============================================================================
# Config file + output path/prefix
# =============================================================================


@pytest.mark.timeout(30)
@pytest.mark.class_name("tool-runner-config-output")
@pytest.mark.parametrize("target", TARGETS)
class TestConfigOutput(RocprofsysTest):
    """-c/--config and -o/--output actually load from and write to the given paths."""

    def test(self, target, test_output_dir):
        config_dir = test_output_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        empty_cfg = config_dir / "empty.cfg"
        empty_cfg.write_text("#\n# empty config file\n#\n")

        output_prefix = "tool-runner-config-output-"
        result = self.run_test(
            "baseline",
            target=target,
            run_args=[
                "-c",
                str(empty_cfg),
                "-o",
                str(test_output_dir),
                output_prefix,
                "--",
                "ls",
            ],
            fail_on_not_found=True,
        )
        self.assert_regex(result)

        metadata_file = result.get_output_file(f"{output_prefix}metadata*.json")
        if metadata_file is None:
            pytest.fail(
                f"-c {empty_cfg} -o {test_output_dir} {output_prefix!r} did not "
                f"produce a metadata file under {test_output_dir}"
            )
        self.assert_file_exists(metadata_file, description="Sample metadata output")


# =============================================================================
# Trace / profile flags
# =============================================================================

TRACE_PROFILE_FLAG_CASES = [
    pytest.param(["-T"], [r"ROCPROFSYS_TRACE=true"], id="trace"),
    pytest.param(["-L"], [r"ROCPROFSYS_TRACE_LEGACY=true"], id="trace_legacy"),
    pytest.param(["-P"], [r"ROCPROFSYS_PROFILE=true"], id="profile"),
    pytest.param(
        ["-F"],
        [r"ROCPROFSYS_PROFILE=true", r"ROCPROFSYS_FLAT_PROFILE=true"],
        id="flat_profile",
    ),
]


@pytest.mark.timeout(30)
@pytest.mark.class_name("tool-runner-trace-profile-flags")
@pytest.mark.parametrize("target", TARGETS)
class TestTraceProfileFlags(RocprofsysTest):
    """-T/-L/-P/-F individually set their corresponding env vars."""

    @pytest.mark.parametrize("flag_args, pass_regex", TRACE_PROFILE_FLAG_CASES)
    def test(self, target, flag_args, pass_regex):
        result = self.run_test(
            "baseline",
            target=target,
            run_args=[*flag_args, "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=pass_regex)


# =============================================================================
# Host / device flags
# =============================================================================

HOST_DEVICE_FLAG_CASES = [
    pytest.param(
        ["-H"],
        [
            r"ROCPROFSYS_USE_PROCESS_SAMPLING=true",
            r"ROCPROFSYS_CPU_FREQ_ENABLED=true",
            r"ROCPROFSYS_USE_AMD_SMI=false",
        ],
        id="host_only",
    ),
    pytest.param(
        ["-D"],
        [
            r"ROCPROFSYS_USE_PROCESS_SAMPLING=true",
            r"ROCPROFSYS_USE_AMD_SMI=true",
            r"ROCPROFSYS_CPU_FREQ_ENABLED=false",
        ],
        id="device_only",
    ),
    pytest.param(
        ["-H", "-D"],
        [
            r"ROCPROFSYS_USE_PROCESS_SAMPLING=true",
            r"ROCPROFSYS_CPU_FREQ_ENABLED=true",
            r"ROCPROFSYS_USE_AMD_SMI=true",
        ],
        id="host_and_device",
    ),
]


@pytest.mark.timeout(30)
@pytest.mark.class_name("tool-runner-host-device-flags")
@pytest.mark.parametrize("target", TARGETS)
class TestHostDeviceFlags(RocprofsysTest):
    """-H and -D cross-set process-sampling env vars based on argparse.cpp's
    host/device interaction logic (each reads the other's current state).
    """

    @pytest.mark.parametrize("flag_args, pass_regex", HOST_DEVICE_FLAG_CASES)
    def test(self, target, flag_args, pass_regex):
        result = self.run_test(
            "baseline",
            target=target,
            run_args=[*flag_args, "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=pass_regex)


# =============================================================================
# Wait / duration short flags
# =============================================================================


@pytest.mark.timeout(30)
@pytest.mark.class_name("tool-runner-wait-duration")
@pytest.mark.parametrize("target", TARGETS)
class TestWaitDuration(RocprofsysTest):
    """-w/-d (short forms) fan out to the trace and sampling delay/duration
    env vars.
    """

    def test_wait_sets_delay_envs(self, target):
        result = self.run_test(
            "baseline",
            target=target,
            run_args=["-w", "1.5", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[
                r"ROCPROFSYS_TRACE_DELAY=1\.500000",
                r"ROCPROFSYS_SAMPLING_DELAY=1\.500000",
            ],
        )

    def test_duration_sets_duration_envs(self, target):
        result = self.run_test(
            "baseline",
            target=target,
            run_args=["-d", "2.5", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[
                r"ROCPROFSYS_TRACE_DURATION=2\.500000",
                r"ROCPROFSYS_SAMPLING_DURATION=2\.500000",
            ],
        )


# =============================================================================
# --periods
# =============================================================================


@pytest.mark.timeout(30)
@pytest.mark.class_name("tool-runner-trace-periods")
@pytest.mark.parametrize("target", TARGETS)
class TestTracePeriods(RocprofsysTest):
    """--periods sets ROCPROFSYS_TRACE_PERIODS; repeated occurrences are
    space-joined rather than overwriting each other.
    """

    def test_single_period_sets_env(self, target):
        result = self.run_test(
            "baseline",
            target=target,
            run_args=["--periods", "0:2", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=[r"ROCPROFSYS_TRACE_PERIODS=0:2"])

    def test_repeated_periods_are_space_joined(self, target):
        result = self.run_test(
            "baseline",
            target=target,
            run_args=["--periods", "0:2", "--periods", "3:2", "-v", "2", "--", "ls"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=[r"ROCPROFSYS_TRACE_PERIODS=0:2 3:2"])
