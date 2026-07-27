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
import json
import shutil
import pytest
from conftest import RocprofsysTest

pytestmark = [pytest.mark.presets]

TARGETS = [
    pytest.param("rocprof-sys-run", marks=pytest.mark.sys_run, id="run"),
    pytest.param("rocprof-sys-sample", marks=pytest.mark.sampling, id="sample"),
]

# Shared tail for the CLI-flag tests below: "-v 2" so the tool echoes its
# resolved env, and "-- ls" as a minimal child process to instrument.
VERBOSE_LS_ARGS = ["-v", "2", "--", "ls"]


def _parse_metadata_settings(metadata_file) -> dict:
    """Read a metadata-*.json's settings block into {ROCPROFSYS_KEY: resolved value}.

    metadata.json is the tool's own record of what a setting actually resolved
    to, so checking it catches regressions in storage/serialization that a
    -v 2 log echo wouldn't necessarily show.
    """
    settings = json.loads(metadata_file.read_text())["rocprofiler-systems"]["metadata"][
        "settings"
    ]
    return {
        key: entry["value"]
        for key, entry in settings.items()
        if isinstance(entry, dict) and "value" in entry
    }


def _resolved_settings(result) -> dict:
    """Find the run's metadata-*.json at the default output location and parse it."""
    metadata_file = result.get_output_file("metadata*.json")
    if metadata_file is None:
        pytest.fail(f"No metadata*.json found under {result.output_dir}")
    return _parse_metadata_settings(metadata_file)


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
            run_args=["--preset=profile-only", *VERBOSE_LS_ARGS],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[r"ROCPROFSYS_TRACE=false(?![^\n]*:)"],
            fail_regex=[r"ROCPROFSYS_TRACE=\S*:\S*"],
        )

        settings = _resolved_settings(result)
        assert settings["ROCPROFSYS_TRACE"] is False, (
            f"metadata.json still resolved ROCPROFSYS_TRACE to "
            f"{settings['ROCPROFSYS_TRACE']!r}"
        )
        assert result.perfetto_file is None, (
            f"profile-only should disable tracing, but found a perfetto trace "
            f"at {result.perfetto_file}"
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
            run_args=["--output-format", "proto", "rocpd", *VERBOSE_LS_ARGS],
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

        settings = _resolved_settings(result)
        assert settings["ROCPROFSYS_TRACE"] is True
        assert settings["ROCPROFSYS_USE_ROCPD"] is True
        assert settings["ROCPROFSYS_PROFILE"] is False
        assert result.perfetto_file is not None, "expected a perfetto trace file"
        assert result.rocpd_files, "expected at least one rocpd database file"

    @pytest.mark.parametrize("target", TARGETS)
    def test_rocpd_only_disables_perfetto(self, target):
        result = self.run_test(
            "baseline",
            target=target,
            run_args=["--output-format", "rocpd", *VERBOSE_LS_ARGS],
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

        settings = _resolved_settings(result)
        assert settings["ROCPROFSYS_USE_ROCPD"] is True
        assert settings["ROCPROFSYS_TRACE"] is False
        assert settings["ROCPROFSYS_PROFILE"] is False
        assert result.perfetto_file is None, (
            f"rocpd-only should not produce a perfetto trace, found "
            f"{result.perfetto_file}"
        )
        assert result.rocpd_files, "expected at least one rocpd database file"

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


@pytest.fixture
def instrumentable_child(rocprof_config) -> str:
    """Child executable that actually has instrumentable functions.

    "ls" doesn't work here — the runtime never gets far enough into init to
    validate the config and reject a bad setting (same reason test_config.py's
    config_target fixture avoids it). Falls back to "ls" if parallel-overhead
    isn't built, at which point the malformed-config test just won't catch
    anything meaningful.
    """
    try:
        return str(rocprof_config.get_target_executable("parallel-overhead"))
    except FileNotFoundError:
        return shutil.which("ls") or "ls"


@pytest.mark.timeout(30)
@pytest.mark.class_name("tool-runner-config-output")
@pytest.mark.parametrize("target", TARGETS)
class TestConfigOutput(RocprofsysTest):
    """-c/--config and -o/--output actually load from and write to the given paths."""

    def test_values_reach_env_and_metadata_file(self, target, test_output_dir):
        config_dir = test_output_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        empty_cfg = config_dir / "empty.cfg"
        empty_cfg.write_text("#\n# empty config file\n#\n")

        # Has to be a subdir, not test_output_dir itself — the harness already
        # points ROCPROFSYS_OUTPUT_PATH there, so reusing it wouldn't actually
        # prove -o overrides the default.
        output_dir = test_output_dir / "custom_output"
        output_prefix = "tool-runner-config-output-"
        result = self.run_test(
            "baseline",
            target=target,
            run_args=[
                "-c",
                str(empty_cfg),
                "-o",
                str(output_dir),
                output_prefix,
                *VERBOSE_LS_ARGS,
            ],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[
                f"ROCPROFSYS_CONFIG_FILE={empty_cfg}",
                f"ROCPROFSYS_OUTPUT_PATH={output_dir}",
                f"ROCPROFSYS_OUTPUT_PREFIX={output_prefix}",
            ],
        )

        metadata_file = result.get_output_file(
            f"custom_output/{output_prefix}metadata*.json"
        )
        if metadata_file is None:
            pytest.fail(
                f"-c {empty_cfg} -o {output_dir} {output_prefix!r} did not "
                f"produce a metadata file under {output_dir}"
            )
        self.assert_file_exists(metadata_file, description="Sample metadata output")

        settings = _parse_metadata_settings(metadata_file)
        assert settings["ROCPROFSYS_CONFIG_FILE"] == str(empty_cfg)
        assert settings["ROCPROFSYS_OUTPUT_PATH"] == str(output_dir)
        assert settings["ROCPROFSYS_OUTPUT_PREFIX"] == output_prefix

    def test_malformed_file_is_rejected(
        self, target, create_config_file, instrumentable_child
    ):
        """-c should feed a bad config into the same validation path as
        ROCPROFSYS_CONFIG_FILE. test_config.py already covers that env var in
        depth but only through rocprof-sys-run; here we just want to confirm
        the -c flag itself hits the same path, on both binaries.
        """
        malformed_cfg = create_config_file(
            {"ROCPROFSYS_TRACE_DURATION": "not-a-number"},
            "malformed.cfg",
            skip_filter=True,
        )
        result = self.run_test(
            "baseline",
            target=target,
            run_args=["-c", str(malformed_cfg), "--", instrumentable_child],
            fail_on_pass=True,
        )
        self.assert_regex(
            result,
            pass_regex=[r"[Ii]nvalid value.*ROCPROFSYS_TRACE_DURATION"],
            use_abort_fail_regex=False,
        )


# =============================================================================
# Trace/profile and host/device flags
# ----------------------------------------------------------------------------
# -T/-L/-P/-F and -H/-D all fan a single flag out to a handful of env vars, so
# they share one parametrized class instead of two near-identical ones.
# =============================================================================

CLI_FLAG_ENV_CASES = [
    pytest.param(["-T"], {"ROCPROFSYS_TRACE": True}, "trace", id="trace"),
    pytest.param(["-L"], {"ROCPROFSYS_TRACE_LEGACY": True}, "trace", id="trace_legacy"),
    pytest.param(["-P"], {"ROCPROFSYS_PROFILE": True}, "profile", id="profile"),
    pytest.param(
        ["-F"],
        {"ROCPROFSYS_PROFILE": True, "ROCPROFSYS_FLAT_PROFILE": True},
        "profile",
        id="flat_profile",
    ),
    pytest.param(
        ["-H"],
        {
            "ROCPROFSYS_USE_PROCESS_SAMPLING": True,
            "ROCPROFSYS_CPU_FREQ_ENABLED": True,
            "ROCPROFSYS_USE_AMD_SMI": False,
        },
        None,
        id="host_only",
    ),
    pytest.param(
        ["-D"],
        {
            "ROCPROFSYS_USE_PROCESS_SAMPLING": True,
            "ROCPROFSYS_USE_AMD_SMI": True,
            "ROCPROFSYS_CPU_FREQ_ENABLED": False,
        },
        None,
        id="device_only",
    ),
    pytest.param(
        ["-H", "-D"],
        {
            "ROCPROFSYS_USE_PROCESS_SAMPLING": True,
            "ROCPROFSYS_CPU_FREQ_ENABLED": True,
            "ROCPROFSYS_USE_AMD_SMI": True,
        },
        None,
        id="host_and_device",
    ),
]


@pytest.mark.timeout(30)
@pytest.mark.class_name("tool-runner-cli-flag-env-mapping")
@pytest.mark.parametrize("target", TARGETS)
class TestCliFlagEnvMapping(RocprofsysTest):
    """Each flag sets its documented env var(s) — checked against both the
    -v 2 echo and metadata.json — and produces whatever artifact it implies.
    """

    @pytest.mark.parametrize(
        "flag_args, expected_settings, artifact_kind", CLI_FLAG_ENV_CASES
    )
    def test_sets_expected_settings_and_artifacts(
        self, target, flag_args, expected_settings, artifact_kind
    ):
        result = self.run_test(
            "baseline",
            target=target,
            run_args=[*flag_args, *VERBOSE_LS_ARGS],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[
                f"{key}={str(value).lower()}" for key, value in expected_settings.items()
            ],
        )

        settings = _resolved_settings(result)
        for key, expected in expected_settings.items():
            assert settings[key] == expected, (
                f"{flag_args}: expected {key}={expected!r} in metadata.json, "
                f"got {settings[key]!r}"
            )

        if artifact_kind == "trace":
            assert result.perfetto_file is not None, (
                f"{flag_args} should produce a perfetto trace under "
                f"{result.output_dir}"
            )
        elif artifact_kind == "profile":
            assert result.timemory_files, (
                f"{flag_args} should produce timemory profile output under "
                f"{result.output_dir}"
            )


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
            run_args=["-w", "1.5", *VERBOSE_LS_ARGS],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[
                r"ROCPROFSYS_TRACE_DELAY=1\.500000",
                r"ROCPROFSYS_SAMPLING_DELAY=1\.500000",
            ],
        )

        settings = _resolved_settings(result)
        assert settings["ROCPROFSYS_TRACE_DELAY"] == 1.5
        assert settings["ROCPROFSYS_SAMPLING_DELAY"] == 1.5

    def test_duration_sets_duration_envs(self, target):
        result = self.run_test(
            "baseline",
            target=target,
            run_args=["-d", "2.5", *VERBOSE_LS_ARGS],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[
                r"ROCPROFSYS_TRACE_DURATION=2\.500000",
                r"ROCPROFSYS_SAMPLING_DURATION=2\.500000",
            ],
        )

        settings = _resolved_settings(result)
        assert settings["ROCPROFSYS_TRACE_DURATION"] == 2.5
        assert settings["ROCPROFSYS_SAMPLING_DURATION"] == 2.5


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
            run_args=["--periods", "0:2", *VERBOSE_LS_ARGS],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=[r"ROCPROFSYS_TRACE_PERIODS=0:2"])
        assert _resolved_settings(result)["ROCPROFSYS_TRACE_PERIODS"] == "0:2"

    def test_repeated_periods_are_space_joined(self, target):
        result = self.run_test(
            "baseline",
            target=target,
            run_args=["--periods", "0:2", "--periods", "3:2", *VERBOSE_LS_ARGS],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=[r"ROCPROFSYS_TRACE_PERIODS=0:2 3:2"])
        assert _resolved_settings(result)["ROCPROFSYS_TRACE_PERIODS"] == "0:2 3:2"
