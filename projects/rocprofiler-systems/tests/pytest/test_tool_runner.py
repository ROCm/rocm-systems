# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Regression tests for the tool_runner shared by rocprof-sys-run and
rocprof-sys-sample. Both binaries parse args through the same path, so we run
these through rocprof-sys-run alone. A failure here points at tool_runner,
argparse env handling, or conflict detection.
"""

from __future__ import annotations
import json
import pytest
from conftest import RocprofsysTest

pytestmark = [pytest.mark.tool_runner]

# rocprof-sys-run and rocprof-sys-sample share the same arg-parsing backend
# (rocprof-sys-run -S is equivalent to rocprof-sys-sample), so running the flags
# through rocprof-sys-run alone covers the parsing path for both.
TARGETS = [
    pytest.param("rocprof-sys-run", marks=pytest.mark.sys_run, id="run"),
]

# <nfib> <nthreads> <nitr>. Sized to run for a couple of seconds so sampling
# has time to collect a bunch of fib() frames, while still keeping tests quick.
PARALLEL_OVERHEAD_ARGS = ["20", "2", "100000"]

# <cpu-threads> <kernel-iterations> <sync-every-N-iterations>. Enough kernel
# launches and runtime for the -D device sampler to poll GPU metrics at least
# once, while still finishing quickly.
TRANSPOSE_ARGS = ["4", "300", "50"]

# parallel-overhead can finish before the default 0.5s sampling delay elapses,
# which leaves nothing for sampling to record. Start sampling immediately so the
# workload's fib() frames actually get captured regardless of how fast the host
# runs the workload.
SAMPLING_NO_DELAY = {"ROCPROFSYS_SAMPLING_DELAY": "0.1"}


@pytest.fixture
def cpu_workload(rocprof_config) -> str:
    """CPU workload with real, instrumentable functions. Used by most tests here."""
    try:
        return str(rocprof_config.get_target_executable("parallel-overhead"))
    except FileNotFoundError:
        pytest.skip("parallel-overhead example not built")


@pytest.fixture
def gpu_workload(rocprof_config) -> str:
    """HIP workload that dispatches real GPU kernels."""
    try:
        return str(rocprof_config.get_target_executable("transpose"))
    except FileNotFoundError:
        pytest.skip("transpose example not built")


def parallel_overhead_args(workload: str) -> list[str]:
    """Trailing "-- parallel-overhead <args>" so the flags under test act on
    real work. `workload` is the resolved parallel-overhead executable path."""
    return ["--", workload, *PARALLEL_OVERHEAD_ARGS]


def _parse_metadata_settings(metadata_file) -> dict:
    """Parse a metadata-*.json settings block into {KEY: resolved value}.

    metadata.json records what each setting actually resolved to, so it catches
    storage/serialization bugs the stdout echo might miss.
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


def _profile_has_label(result, needle: str) -> bool:
    """True if a function named `needle` was profiled in sampling_wall_clock.json.

    We read the profile's function list instead of grepping raw bytes. Sample
    counts vary run to run, so we only check the function is there.
    """
    profile = result.output_dir / "sampling_wall_clock.json"
    if not profile.exists():
        pytest.fail(f"No sampling_wall_clock.json found under {result.output_dir}")
    graph = json.loads(profile.read_text())["timemory"]["sampling_wall_clock"]["ranks"][
        0
    ]["graph"]
    return any(needle in node["prefix"] for node in graph)


# ============================================================================
# A preset must fully override a matching env var, not merge with it.
# Old bug: `ROCPROFSYS_TRACE=true` + `--preset=profile-only` merged into
# `ROCPROFSYS_TRACE=true:false`, which silently re-enabled tracing.
# ============================================================================


@pytest.mark.timeout(30)
@pytest.mark.class_name("tool-runner-replace-env")
class TestReplaceEnvNoDuplicates(RocprofsysTest):
    @pytest.mark.parametrize("target", TARGETS)
    def test_preset_overrides_shell_value(self, target, cpu_workload):
        result = self.run_test(
            "baseline",
            target=target,
            env={"ROCPROFSYS_TRACE": "true"},
            run_args=[
                "--preset=profile-only",
                *parallel_overhead_args(cpu_workload),
            ],
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
# --profile and --flat-profile conflict and must be rejected.
# argparse declares it via .conflicts({"flat-profile"}), and tool_runner keeps a
# defensive check too. Either way: non-zero exit and a message naming both flags.
# ============================================================================


@pytest.mark.timeout(30)
@pytest.mark.class_name("tool-runner-profile-conflict")
class TestProfileFlatProfileConflict(RocprofsysTest):
    @pytest.mark.parametrize("target", TARGETS)
    def test_profile_and_flat_profile_rejected(self, target, cpu_workload):
        result = self.run_test(
            "baseline",
            target=target,
            run_args=["--profile", "--flat-profile", "--", cpu_workload],
            fail_on_not_found=True,
            fail_on_pass=True,
        )
        self.assert_regex(
            result,
            pass_regex=[r"--profile.*conflicts.*--flat-profile"],
            use_abort_fail_regex=False,
        )


# ============================================================================
# --output-format lists exactly which outputs to produce: named ones on, the
# rest off. A lone "rocpd" must not leave tracing on via the default where
# ROCPROFSYS_TRACE falls back to the negation of ROCPROFSYS_PROFILE.
# ============================================================================


@pytest.mark.timeout(30)
@pytest.mark.class_name("tool-runner-output-format")
class TestOutputFormatSelection(RocprofsysTest):
    # rocpd output needs a ROCm agent (GPU) and the rocpd DB backend (ROCm >= 7.0);
    # on a CPU-only host it aborts with "Agent not found", so gate the rocpd cases.
    @pytest.mark.gpu
    @pytest.mark.rocm_min_version("7.0")
    @pytest.mark.parametrize("target", TARGETS)
    def test_proto_rocpd_enables_both(self, target, cpu_workload):
        result = self.run_test(
            "baseline",
            target=target,
            env={"ROCPROFSYS_TRACE": "OFF", **SAMPLING_NO_DELAY},
            run_args=[
                "--output-format",
                "proto",
                "rocpd",
                *parallel_overhead_args(cpu_workload),
            ],
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
        self.assert_perfetto(
            result,
            label_substrings=["fib"],
            subtest_name="proto trace records workload fib frames",
        )

    @pytest.mark.gpu
    @pytest.mark.rocm_min_version("7.0")
    @pytest.mark.parametrize("target", TARGETS)
    def test_rocpd_only_disables_perfetto(self, target, cpu_workload):
        result = self.run_test(
            "baseline",
            target=target,
            run_args=[
                "--output-format",
                "rocpd",
                *parallel_overhead_args(cpu_workload),
            ],
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

    @pytest.mark.gpu
    @pytest.mark.rocm_min_version("7.0")
    @pytest.mark.parametrize("target", TARGETS)
    def test_default_produces_rocpd_only(self, target, cpu_workload):
        """No --output-format flag: library default is rocpd on, Perfetto off.

        Uses no_base_env=True so the framework's ROCPROFSYS_TRACE=ON base env
        does not interfere with the library default.  ROCPROFSYS_TIME_OUTPUT=OFF
        and ROCPROFSYS_FILE_OUTPUT=ON are set explicitly because the base env
        normally provides them and the output helpers (rocpd_files, get_output_file)
        expect flat output directly in output_dir, not a timestamped subdirectory.
        """
        result = self.run_test(
            "baseline",
            target=target,
            no_base_env=True,
            env={
                **SAMPLING_NO_DELAY,
                "ROCPROFSYS_USE_AMD_SMI": "OFF",
                "ROCPROFSYS_TIME_OUTPUT": "OFF",
                "ROCPROFSYS_FILE_OUTPUT": "ON",
            },
            run_args=parallel_overhead_args(cpu_workload),
            fail_on_not_found=True,
        )

        # ROCPROFSYS_USE_ROCPD/TRACE/PROFILE are library defaults here (not
        # explicit env vars), so they do not appear in the env dump.  Verify
        # the resolved values from the metadata JSON instead.
        settings = _resolved_settings(result)
        assert (
            settings["ROCPROFSYS_USE_ROCPD"] is True
        ), "expected ROCPROFSYS_USE_ROCPD to default to true"
        assert (
            settings["ROCPROFSYS_TRACE"] is False
        ), "expected ROCPROFSYS_TRACE to default to false"
        assert (
            settings["ROCPROFSYS_PROFILE"] is False
        ), "expected ROCPROFSYS_PROFILE to default to false"
        assert result.rocpd_files, "default run should produce a rocpd database"
        assert result.perfetto_file is None, (
            f"default run should not produce a perfetto trace, found "
            f"{result.perfetto_file}"
        )

    @pytest.mark.parametrize("target", TARGETS)
    def test_proto_only_disables_rocpd(self, target, cpu_workload):
        """--output-format proto enables Perfetto and disables the rocpd default."""
        result = self.run_test(
            "baseline",
            target=target,
            env=SAMPLING_NO_DELAY,
            run_args=[
                "--output-format",
                "proto",
                *parallel_overhead_args(cpu_workload),
            ],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[
                r"ROCPROFSYS_TRACE=true",
                r"ROCPROFSYS_USE_ROCPD=false",
                r"ROCPROFSYS_PROFILE=false",
            ],
        )

        settings = _resolved_settings(result)
        assert settings["ROCPROFSYS_TRACE"] is True
        assert settings["ROCPROFSYS_USE_ROCPD"] is False
        assert settings["ROCPROFSYS_PROFILE"] is False
        assert result.perfetto_file is not None, "expected a perfetto trace file"
        assert not result.rocpd_files, (
            f"proto-only should not produce a rocpd database, found "
            f"{result.rocpd_files}"
        )

    @pytest.mark.parametrize(
        "format_token",
        [
            pytest.param("pftrace", id="pftrace"),
            pytest.param("proto", id="proto_alias"),
        ],
    )
    @pytest.mark.parametrize("target", TARGETS)
    def test_pftrace_and_proto_alias_enable_perfetto(
        self, target, format_token, cpu_workload
    ):
        """'proto' is a deprecated alias for 'pftrace'; both must behave identically."""
        result = self.run_test(
            "baseline",
            target=target,
            env=SAMPLING_NO_DELAY,
            run_args=[
                "--output-format",
                format_token,
                *parallel_overhead_args(cpu_workload),
            ],
            fail_on_not_found=True,
        )
        self.assert_regex(
            result,
            pass_regex=[
                r"ROCPROFSYS_TRACE=true",
                r"ROCPROFSYS_PROFILE=false",
            ],
        )

        settings = _resolved_settings(result)
        assert settings["ROCPROFSYS_TRACE"] is True
        assert settings["ROCPROFSYS_PROFILE"] is False
        assert result.perfetto_file is not None, "expected a perfetto trace file"
        assert result.perfetto_file.suffix == ".pftrace", (
            f"expected a .pftrace trace regardless of the token used, found "
            f"{result.perfetto_file}"
        )

    @pytest.mark.parametrize(
        "old_style_args",
        [
            pytest.param(["--trace"], id="trace"),
            pytest.param(["--profile"], id="profile"),
            pytest.param(["--flat-profile"], id="flat_profile"),
            pytest.param(["--profile-format", "text"], id="profile_format"),
        ],
    )
    @pytest.mark.parametrize("target", TARGETS)
    def test_conflicts_with_old_style_flags(self, target, old_style_args, cpu_workload):
        result = self.run_test(
            "baseline",
            target=target,
            run_args=["--output-format", "rocpd", *old_style_args, "--", cpu_workload],
            fail_on_not_found=True,
            fail_on_pass=True,
        )
        self.assert_regex(
            result,
            pass_regex=[r"--output-format.*conflicts.*"],
            use_abort_fail_regex=False,
        )


# =============================================================================
# Config file + output path/prefix
# =============================================================================


@pytest.mark.timeout(30)
@pytest.mark.class_name("tool-runner-config-output")
@pytest.mark.parametrize("target", TARGETS)
class TestConfigOutput(RocprofsysTest):
    """-c/--config and -o/--output actually load from and write to the given paths."""

    def test_values_reach_env_and_metadata_file(
        self, target, test_output_dir, cpu_workload, create_config_file
    ):
        empty_cfg = create_config_file({}, "empty.cfg", skip_filter=True)

        # Use a subdir, not test_output_dir itself: the harness already points
        # ROCPROFSYS_OUTPUT_PATH there, so reusing it wouldn't prove -o works.
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
                *parallel_overhead_args(cpu_workload),
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


# =============================================================================
# Trace/profile and host/device flags
# -T/-P/-F and -H/-D each fan one flag out to a few env vars, so they share
# one parametrized class instead of several near-identical ones.
# =============================================================================

CLI_FLAG_ENV_CASES = [
    pytest.param(["-T"], {"ROCPROFSYS_TRACE": True}, "trace", id="trace"),
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
        marks=pytest.mark.gpu,
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
        marks=pytest.mark.gpu,
        id="host_and_device",
    ),
]


@pytest.mark.timeout(30)
@pytest.mark.class_name("tool-runner-cli-flag-env-mapping")
@pytest.mark.parametrize("target", TARGETS)
class TestCliFlagEnvMapping(RocprofsysTest):
    """Each flag sets its documented env var(s), verified via the stdout echo
    and metadata.json, and produces the artifact it implies.
    """

    BASE_INJECTED = ("ROCPROFSYS_TRACE", "ROCPROFSYS_PROFILE")

    @pytest.mark.parametrize(
        "flag_args, expected_settings, artifact_kind", CLI_FLAG_ENV_CASES
    )
    def test_sets_expected_settings_and_artifacts(
        self, target, flag_args, expected_settings, artifact_kind, cpu_workload
    ):
        seed_env = {
            key: "OFF"
            for key, value in expected_settings.items()
            if key in self.BASE_INJECTED and value is True
        }
        seed_env.update(SAMPLING_NO_DELAY)
        result = self.run_test(
            "baseline",
            target=target,
            env=seed_env,
            run_args=[*flag_args, *parallel_overhead_args(cpu_workload)],
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
            # A real trace has the workload's fib() frames as slices, not just
            # start/exit. Validate through trace_processor rather than raw bytes.
            self.assert_perfetto(
                result,
                label_substrings=["fib"],
                subtest_name=f"{flag_args}: trace records workload fib frames",
            )
        elif artifact_kind == "profile":
            profile_file = result.output_dir / "sampling_wall_clock.json"
            assert profile_file.is_file(), f"{flag_args} should produce {profile_file}"
            # Same check for the profile: fib() must be a recorded sample entry.
            assert _profile_has_label(result, "fib"), (
                f"{flag_args}: no fib() frames in sampling profile under "
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

    def test_wait_sets_delay_envs(self, target, cpu_workload):
        result = self.run_test(
            "baseline",
            target=target,
            run_args=["-w", "1.5", *parallel_overhead_args(cpu_workload)],
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

    def test_duration_sets_duration_envs(self, target, cpu_workload):
        result = self.run_test(
            "baseline",
            target=target,
            run_args=["-d", "2.5", *parallel_overhead_args(cpu_workload)],
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

    def test_single_period_sets_env(self, target, cpu_workload):
        result = self.run_test(
            "baseline",
            target=target,
            run_args=["--periods", "0:2", *parallel_overhead_args(cpu_workload)],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=[r"ROCPROFSYS_TRACE_PERIODS=0:2"])
        assert _resolved_settings(result)["ROCPROFSYS_TRACE_PERIODS"] == "0:2"

    def test_repeated_periods_are_space_joined(self, target, cpu_workload):
        result = self.run_test(
            "baseline",
            target=target,
            run_args=[
                "--periods",
                "0:2",
                "--periods",
                "3:2",
                *parallel_overhead_args(cpu_workload),
            ],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=[r"ROCPROFSYS_TRACE_PERIODS=0:2 3:2"])
        assert _resolved_settings(result)["ROCPROFSYS_TRACE_PERIODS"] == "0:2 3:2"


# =============================================================================
# -D device sampling (GPU)
# The CPU cases above only prove -D sets ROCPROFSYS_USE_AMD_SMI; on a CPU-only
# workload there's nothing to sample. This runs -D against a real HIP workload
# and checks the device sampler actually wrote GPU counters. gpu/rocm marked,
# so it skips when there's no GPU.
# =============================================================================


@pytest.mark.gpu
@pytest.mark.timeout(120)
@pytest.mark.class_name("tool-runner-device-sampling")
@pytest.mark.parametrize("target", TARGETS)
class TestDeviceSamplingArtifacts(RocprofsysTest):
    """-D on a real GPU workload records device metric counters, not just the
    ROCPROFSYS_USE_AMD_SMI env var.
    """

    def test_records_gpu_counters(self, target, gpu_workload):
        result = self.run_test(
            "baseline",
            target=target,
            # Trace HIP/kernels so we can see the workload ran, and list the
            # AMD-SMI metrics -D should collect.
            env={
                "ROCPROFSYS_ROCM_DOMAINS": "hip_runtime_api,kernel_dispatch",
                "ROCPROFSYS_AMD_SMI_METRICS": "busy,temp,power,mem_usage",
            },
            run_args=["-D", "--", gpu_workload, *TRANSPOSE_ARGS],
            fail_on_not_found=True,
        )

        # -D must set the device-sampling env var.
        self.assert_regex(result, pass_regex=[r"ROCPROFSYS_USE_AMD_SMI=true"])
        assert _resolved_settings(result)["ROCPROFSYS_USE_AMD_SMI"] is True

        assert (
            result.perfetto_file is not None
        ), f"-D should produce a perfetto trace under {result.output_dir}"
        trace = result.perfetto_file.read_bytes()

        # The transpose kernel shows the GPU workload actually ran.
        assert (
            b"transpose" in trace
        ), f"-D trace at {result.perfetto_file} has no transpose kernel"
        # GPU counters like "GPU [0] GFX Busy (S)" only appear when AMD-SMI
        # sampling ran, so this is the real proof -D worked, not just the env var.
        assert (
            b"GFX Busy" in trace
        ), f"-D trace at {result.perfetto_file} has no GPU device counters"
