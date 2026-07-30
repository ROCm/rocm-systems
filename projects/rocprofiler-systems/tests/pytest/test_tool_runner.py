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

# <nfib> <nthreads> <nitr>. Big enough to run for about a second and get
# sampled a bunch, small enough to keep tests quick.
PARALLEL_OVERHEAD_ARGS = ["18", "2", "100000"]

# <matrix-size> <iterations> <num-inputs>. Enough kernel launches and runtime
# for the -D device sampler to poll GPU metrics at least once, while still
# finishing quickly.
TRANSPOSE_ARGS = ["4", "300", "50"]


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


def workload_args(workload: str) -> list[str]:
    """Trailing "-- <workload>" so the flags under test act on real work."""
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
                *workload_args(cpu_workload),
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
            env={"ROCPROFSYS_TRACE": "OFF"},
            run_args=[
                "--output-format",
                "proto",
                "rocpd",
                *workload_args(cpu_workload),
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
                *workload_args(cpu_workload),
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
    def test_conflicts_with_legacy_flags(self, target, legacy_args, cpu_workload):
        result = self.run_test(
            "baseline",
            target=target,
            run_args=["--output-format", "rocpd", *legacy_args, "--", cpu_workload],
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
                *workload_args(cpu_workload),
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
# -T/-L/-P/-F and -H/-D each fan one flag out to a few env vars, so they share
# one parametrized class instead of several near-identical ones.
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
        result = self.run_test(
            "baseline",
            target=target,
            env=seed_env,
            run_args=[*flag_args, *workload_args(cpu_workload)],
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
            if key == "ROCPROFSYS_USE_AMD_SMI":
                continue
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
            assert result.timemory_files, (
                f"{flag_args} should produce timemory profile output under "
                f"{result.output_dir}"
            )
            # Same check for the profile: fib() must be a recorded sample entry.
            assert _profile_has_label(result, "fib(long)"), (
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
            run_args=["-w", "1.5", *workload_args(cpu_workload)],
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
            run_args=["-d", "2.5", *workload_args(cpu_workload)],
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
            run_args=["--periods", "0:2", *workload_args(cpu_workload)],
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
                *workload_args(cpu_workload),
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
