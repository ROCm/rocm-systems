# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Tests rocprof-sys binaries
"""

from __future__ import annotations
import json
import re
import pytest
import os
from conftest import RocprofsysTest

pytestmark = [
    pytest.mark.rocprof_binary,
    pytest.mark.ci_enable,  # TODO: Deprecate once TheRock switches to CTest
]

# ============================================================================
# Avail format consistency data
# ============================================================================

# Env vars intentionally excluded from the JSON preset schema.
# These are internal/session-specific settings documented in
# json_config.cpp::env_vars_to_json_schema(). If this set needs
# updating, that comment should be updated too.
EXCLUDED_FROM_JSON_SCHEMA: frozenset[str] = frozenset(
    {
        "ROCPROFSYS_CI",
        "ROCPROFSYS_CONFIG_FILE",
        "ROCPROFSYS_ENABLED",
        "ROCPROFSYS_OUTPUT_PREFIX",
        "ROCPROFSYS_SUPPRESS_CONFIG",
        "ROCPROFSYS_SUPPRESS_PARSING",
        "ROCPROFSYS_TMPDIR",
    }
)

# Mapping from ROCPROFSYS_* env var names to their expected JSON schema
# path. Used to verify that the JSON config export covers the same
# settings as the TXT config export.
ENV_VAR_TO_JSON_PATH: dict[str, str] = {
    # --- Tracing ---
    "ROCPROFSYS_TRACE": "tracing.enabled",
    "ROCPROFSYS_TRACE_LEGACY": "tracing.legacy",
    "ROCPROFSYS_PERFETTO_BUFFER_SIZE_KB": "tracing.buffer_size_kb",
    "ROCPROFSYS_PERFETTO_FILL_POLICY": "tracing.fill_policy",
    "ROCPROFSYS_PERFETTO_BACKEND": "tracing.backend",
    "ROCPROFSYS_PERFETTO_FLUSH_PERIOD_MS": "tracing.flush_period_ms",
    "ROCPROFSYS_SELECTED_REGIONS": "tracing.region",
    # --- Profiling ---
    "ROCPROFSYS_PROFILE": "profiling.enabled",
    "ROCPROFSYS_FLAT_PROFILE": "profiling.flat_profile",
    # --- Sampling ---
    "ROCPROFSYS_USE_SAMPLING": "sampling.enabled",
    "ROCPROFSYS_SAMPLING_FREQ": "sampling.frequency_hz",
    "ROCPROFSYS_SAMPLING_DELAY": "sampling.delay_sec",
    "ROCPROFSYS_SAMPLING_DURATION": "sampling.duration_sec",
    "ROCPROFSYS_SAMPLING_CPUS": "sampling.cpus",
    "ROCPROFSYS_SAMPLING_GPUS": "sampling.gpus",
    "ROCPROFSYS_SAMPLING_AINICS": "sampling.ainics",
    "ROCPROFSYS_SAMPLING_OVERFLOW_EVENT": "sampling.overflow_event",
    # --- Domains: GPU ---
    "ROCPROFSYS_USE_AMD_SMI": "domains.gpu.enabled",
    "ROCPROFSYS_AMD_SMI_METRICS": "domains.gpu.metrics",
    "ROCPROFSYS_USE_AINIC": "domains.gpu.ainic",
    "ROCPROFSYS_USE_PROCESS_SAMPLING": "domains.gpu.process_sampling",
    "ROCPROFSYS_PROCESS_SAMPLING_FREQ": "domains.gpu.process_sampling_freq",
    "ROCPROFSYS_PROCESS_SAMPLING_DURATION": "domains.gpu.process_sampling_duration",
    "ROCPROFSYS_CPU_FREQ_ENABLED": "domains.cpu.cpu_freq_enabled",
    "ROCPROFSYS_CPU_METRICS": "domains.cpu.metrics",
    # --- Domains: ROCm ---
    "ROCPROFSYS_ROCM_DOMAINS": "domains.rocm.api_domains",
    "ROCPROFSYS_ROCM_GROUP_BY_QUEUE": "domains.rocm.group_by_queue",
    # --- Domains: Parallel ---
    "ROCPROFSYS_USE_MPIP": "domains.parallel.runtimes.mpi",
    "ROCPROFSYS_USE_OMPT": "domains.parallel.runtimes.openmp",
    "ROCPROFSYS_USE_KOKKOSP": "domains.parallel.runtimes.kokkos",
    "ROCPROFSYS_USE_RCCLP": "domains.parallel.runtimes.rccl",
    "ROCPROFSYS_USE_SHMEM": "domains.parallel.runtimes.shmem",
    "ROCPROFSYS_USE_UCX": "domains.parallel.runtimes.ucx",
    # --- Output ---
    "ROCPROFSYS_OUTPUT_PATH": "output.path",
    "ROCPROFSYS_TIME_OUTPUT": "output.time_output",
    "ROCPROFSYS_FILE_OUTPUT": "output.file_output",
    "ROCPROFSYS_USE_ROCPD": "output.rocpd_output",
    "ROCPROFSYS_USE_PID": "output.use_pid",
    # --- Hardware counters ---
    "ROCPROFSYS_ROCM_EVENTS": "hardware_counters.rocm_events",
    "ROCPROFSYS_PAPI_EVENTS": "hardware_counters.papi_events",
    "ROCPROFSYS_PAPI_MULTIPLEXING": "hardware_counters.papi_multiplexing",
    # --- Causal ---
    "ROCPROFSYS_USE_CAUSAL": "causal.enabled",
    "ROCPROFSYS_CAUSAL_MODE": "causal.mode",
    "ROCPROFSYS_CAUSAL_BACKEND": "causal.backend",
    "ROCPROFSYS_CAUSAL_BINARY_SCOPE": "causal.binary_scope",
    "ROCPROFSYS_CAUSAL_BINARY_EXCLUDE": "causal.binary_exclude",
    "ROCPROFSYS_CAUSAL_FUNCTION_SCOPE": "causal.function_scope",
    "ROCPROFSYS_CAUSAL_FUNCTION_EXCLUDE": "causal.function_exclude",
    "ROCPROFSYS_CAUSAL_SOURCE_SCOPE": "causal.source_scope",
    "ROCPROFSYS_CAUSAL_SOURCE_EXCLUDE": "causal.source_exclude",
    "ROCPROFSYS_CAUSAL_DELAY": "causal.delay_sec",
    "ROCPROFSYS_CAUSAL_DURATION": "causal.duration_sec",
    "ROCPROFSYS_CAUSAL_RANDOM_SEED": "causal.random_seed",
    # --- Advanced ---
    "ROCPROFSYS_VERBOSE": "advanced.verbose",
    "ROCPROFSYS_DEBUG": "advanced.debug",
    "ROCPROFSYS_MAX_DEPTH": "advanced.max_depth",
    "ROCPROFSYS_TRACE_DELAY": "advanced.trace_delay_sec",
    "ROCPROFSYS_TRACE_DURATION": "advanced.trace_duration_sec",
    "ROCPROFSYS_CPU_AFFINITY": "advanced.cpu_affinity",
    "ROCPROFSYS_COLLAPSE_THREADS": "advanced.collapse_threads",
    "ROCPROFSYS_TIMEMORY_COMPONENTS": "advanced.timemory_components",
    "ROCPROFSYS_NETWORK_INTERFACE": "advanced.network_interface",
    "ROCPROFSYS_TRACE_PERIODS": "advanced.trace_periods",
    "ROCPROFSYS_TRACE_PERIOD_CLOCK_ID": "advanced.trace_period_clock_id",
}


def flatten_json_keys(
    obj: dict | list | str | int | float | bool, prefix: str = ""
) -> set[str]:
    """Recursively collect all leaf paths in a JSON object."""
    keys: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys |= flatten_json_keys(v, f"{prefix}.{k}" if prefix else k)
    else:
        keys.add(prefix)
    return keys


def validate_format_consistency(txt_vars: set[str], json_paths: set[str]) -> list[str]:
    """Check that every TXT env var has a JSON schema mapping.

    Returns a list of error messages for vars without coverage.
    """
    missing = []
    for var in sorted(txt_vars):
        if var in EXCLUDED_FROM_JSON_SCHEMA:
            continue
        if var in ENV_VAR_TO_JSON_PATH:
            json_key = ENV_VAR_TO_JSON_PATH[var]
            if not any(json_key in path for path in json_paths):
                missing.append(f"{var} -> expected JSON path '{json_key}' not found")
        else:
            missing.append(f"{var} -> no JSON schema mapping defined")
    return missing


# ============================================================================
# Helper functions
# ============================================================================


def get_ls_command() -> tuple[str, list[str]]:
    """Get ls binary name and args (handles RedHat coreutils wrapper).

    Returns:
        Tuple of (binary_name, args_list)
    """
    if os.path.exists("/usr/bin/coreutils"):
        return "coreutils", ["--coreutils-prog=ls"]
    return "ls", []


# ============================================================================
# rocprof-sys-instrument tests
# ============================================================================


@pytest.mark.instrument
@pytest.mark.class_name("rocprofiler-systems-instrument")
class TestRocprofilerSystemsInstrument(RocprofsysTest):
    """Tests for rocprof-sys-instrument binary."""

    target = "rocprof-sys-instrument"

    @pytest.mark.timeout(45)
    def test_help(self):
        pass_regex = [
            r"\[rocprof-sys-instrument\] Usage:[\s\S]*"
            r"\[DEBUG OPTIONS\][\s\S]*"
            r"\[MODE OPTIONS\][\s\S]*"
            r"\[LIBRARY OPTIONS\][\s\S]*"
            r"\[SYMBOL SELECTION OPTIONS\][\s\S]*"
            r"\[RUNTIME OPTIONS\][\s\S]*"
            r"\[GRANULARITY OPTIONS\][\s\S]*"
            r"\[DYNINST OPTIONS\]"
        ]

        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["--help"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=pass_regex)

    @pytest.mark.timeout(240)
    def test_simulate_ls(self):
        ls_name, ls_args = get_ls_command()

        test_args = [
            "--simulate",
            "--dump-info",
            "--print-format",
            "json",
            "txt",
            "xml",
            "-v",
            "2",
            "--all-functions",
            "--",
            ls_name,
            *ls_args,
        ]

        expected_files = [
            "available.json",
            "available.txt",
            "available.xml",
            "excluded.json",
            "excluded.txt",
            "excluded.xml",
            "instrumented.json",
            "instrumented.txt",
            "instrumented.xml",
            "overlapping.json",
            "overlapping.txt",
            "overlapping.xml",
        ]

        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=test_args,
            fail_on_not_found=True,
        )

        self.assert_regex(result)
        expected_files_paths = [
            result.output_dir / "instrumentation" / f for f in expected_files
        ]
        self.assert_file_exists(expected_files_paths)

    @pytest.mark.timeout(120)
    def test_simulate_lib(self, rocprof_config):
        user_lib = rocprof_config.rocprofsys_lib_dir / "librocprof-sys-user.so"
        if not user_lib.exists():
            pytest.fail("librocprof-sys-user.so not found")

        pass_regex = [
            r"\[rocprof-sys\]\[exe\] Runtime instrumentation is not possible![\s\S]*"
            r"\[rocprof-sys\]\[exe\] Switching to binary rewrite mode and assuming '--simulate --all-functions'"
        ]

        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["--print-available", "functions", "-v", "2", "--", str(user_lib)],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=pass_regex)

    @pytest.mark.timeout(120)
    def test_simulate_lib_basename(self, rocprof_config, test_output_dir):
        """Test instrument with library basename.

        This MUST be run from a tmp directory, NOT from the actual lib directory.
        Running from the lib directory causes Dyninst to modify the library in-place,
        contaminating it with instrumentation markers. This breaks all subsequent
        binary rewrite tests with "unable to reinstrument previously instrumented
        binary" errors.
        """
        lib_basename = "librocprof-sys-user.so"
        user_lib = rocprof_config.rocprofsys_lib_dir / lib_basename
        if not user_lib.exists():
            pytest.skip(f"{lib_basename} not built")

        tmp_dir = test_output_dir / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        output_lib = test_output_dir / lib_basename

        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=[
                "--print-available",
                "functions",
                "-v",
                "2",
                "-o",
                str(output_lib),
                "--",
                lib_basename,
            ],
            working_directory=tmp_dir,
            fail_on_not_found=True,
        )
        self.assert_regex(result)

    @pytest.mark.timeout(120)
    def test_write_log(self):
        """Test instrument writing to log file."""
        ls_name, ls_args = get_ls_command()

        pass_regex = [r"Opening .*/instrumentation/user\.log"]

        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=[
                "--print-instrumented",
                "functions",
                "-v",
                "1",
                "--log-file",
                "user.log",
                "--",
                ls_name,
                *ls_args,
            ],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=pass_regex)
        self.assert_file_exists(result.output_dir / "instrumentation" / "user.log")


# ============================================================================
# rocprof-sys-avail tests
# ============================================================================


@pytest.mark.avail
@pytest.mark.class_name("rocprofiler-systems-avail")
class TestRocprofilerSystemsAvail(RocprofsysTest):
    """Tests for rocprof-sys-avail binary."""

    target = "rocprof-sys-avail"

    @pytest.mark.timeout(45)
    def test_help(self):
        pass_regex = [
            r"\[rocprof-sys-avail\] Usage:[\s\S]*"
            r"\[DEBUG OPTIONS\][\s\S]*"
            r"\[INFO OPTIONS\][\s\S]*"
            r"\[FILTER OPTIONS\][\s\S]*"
            r"\[COLUMN OPTIONS\][\s\S]*"
            r"\[DISPLAY OPTIONS\][\s\S]*"
            r"\[OUTPUT OPTIONS\][\s\S]*"
        ]

        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["--help"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=pass_regex)

    @pytest.mark.timeout(45)
    def test_all(self):
        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["--all"],
            fail_on_not_found=True,
        )
        self.assert_regex(result)

    @pytest.mark.timeout(45)
    def test_all_expand_keys(self):
        fail_regex = [r"%[a-zA-Z_]%"]

        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["--all", "--expand-keys"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, fail_regex=fail_regex)

    @pytest.mark.timeout(45)
    def test_all_only_available_alphabetical(self, test_output_dir):
        log_file = (
            test_output_dir / "rocprof-sys-avail-all-only-available-alphabetical.log"
        )

        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=[
                "--all",
                "--available",
                "--alphabetical",
                "--debug",
                "--output",
                str(log_file),
            ],
            fail_on_not_found=True,
        )
        self.assert_regex(result)
        self.assert_file_exists(log_file)

    @pytest.mark.timeout(45)
    def test_all_csv(self):
        pass_regex = [
            r"COMPONENT#AVAILABLE#VALUE_TYPE#STRING_IDS#FILENAME#DESCRIPTION#CATEGORY#[\s\S]*"
            r"ENVIRONMENT VARIABLE#VALUE#DATA TYPE#DESCRIPTION#CATEGORIES#[\s\S]*"
            r"HARDWARE COUNTER#DEVICE#AVAILABLE#DESCRIPTION#"
        ]

        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["--all", "--csv", "--csv-separator", "#"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=pass_regex)

    @pytest.mark.timeout(45)
    def test_filter_wall_clock_available(self):
        pass_regex = [
            r"\|[-]+\|[\s\S]*"
            r"\|[ ]+COMPONENT[ ]+\|[\s\S]*"
            r"\|[-]+\|[\s\S]*"
            r"\| (wall_clock)[ ]+\|[\s\S]*"
            r"\|[-]+\|"
        ]

        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["-r", "wall_clock", "-C", "--available"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=pass_regex)

    @pytest.mark.timeout(45)
    def test_category_filter_rocprofiler_systems(self):
        pass_regex = [r"ROCPROFSYS_(SETTINGS_DESC|OUTPUT_FILE|OUTPUT_PREFIX)"]
        fail_regex = [
            r"ROCPROFSYS_(ADD_SECONDARY|SCIENTIFIC|PRECISION|MEMORY_PRECISION|TIMING_PRECISION)",
        ]

        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["--categories", "settings::rocprofsys", "--brief"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=pass_regex, fail_regex=fail_regex)

    @pytest.mark.timeout(45)
    def test_category_filter_timemory(self):
        pass_regex = [
            r"ROCPROFSYS_(ADD_SECONDARY|SCIENTIFIC|PRECISION|MEMORY_PRECISION|TIMING_PRECISION)"
        ]
        fail_regex = [r"ROCPROFSYS_(SETTINGS_DESC|OUTPUT_FILE)"]

        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["--categories", "settings::timemory", "--brief", "--advanced"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=pass_regex, fail_regex=fail_regex)

    @pytest.mark.timeout(45)
    def test_regex_negation(self):
        pass_regex = [
            r"ENVIRONMENT VARIABLE,[\s\S]*"
            r"ROCPROFSYS_CI_SKIP_PUSH_POP_CHECK,[\s\S]*"
            r"ROCPROFSYS_THREAD_POOL_SIZE,[\s\S]*"
            r"ROCPROFSYS_USE_PID,"
        ]
        fail_regex = [r"ROCPROFSYS_TRACE"]

        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=[
                "-R",
                "rocprofsys",
                "~timemory",
                "-r",
                "_P",
                "~PERFETTO",
                "~PROCESS_SAMPLING",
                "~KOKKOSP",
                "~PAGE",
                "--csv",
                "--brief",
                "--advanced",
            ],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=pass_regex, fail_regex=fail_regex)

    @pytest.mark.timeout(45)
    def test_write_config(self, test_output_dir):
        config_base = test_output_dir / "rocprof-sys-test"

        avail_cfg_path = test_output_dir / "rocprof-sys-"
        avail_cfg_path = str(avail_cfg_path).replace("+", r"\+")

        pass_regex = [
            rf"Outputting JSON configuration file '{avail_cfg_path}test\.json'"
            r"[\s\S]*"
            rf"Outputting XML configuration file '{avail_cfg_path}test\.xml'"
            r"[\s\S]*"
            rf"Outputting text configuration file '{avail_cfg_path}test\.cfg'"
        ]

        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=[
                "-G",
                str(config_base) + ".cfg",
                "-F",
                "txt",
                "json",
                "xml",
                "--force",
                "--all",
                "-c",
                "rocprofsys",
            ],
            fail_on_not_found=True,
        )

        self.assert_regex(result, pass_regex=pass_regex)

        config_files = [
            test_output_dir / f"rocprof-sys-test.{ext}" for ext in ["cfg", "json", "xml"]
        ]
        self.assert_file_exists(
            config_files, subtest_name="Config file existence validation"
        )

    @pytest.mark.timeout(45)
    def test_write_config_tweak(self, test_output_dir):
        config_base = test_output_dir / "rocprof-sys-tweak"

        env_overrides = {
            "ROCPROFSYS_TRACE": "OFF",
            "ROCPROFSYS_PROFILE": "ON",
            "ROCPROFSYS_USE_SAMPLING": "OFF",
            "ROCPROFSYS_TIME_OUTPUT": "OFF",
        }

        avail_cfg_path = test_output_dir / "rocprof-sys-"
        avail_cfg_path = str(avail_cfg_path).replace("+", r"\+")

        pass_regex = [
            rf"Outputting JSON configuration file '{avail_cfg_path}tweak\.json'"
            r"[\s\S]*"
            rf"Outputting XML configuration file '{avail_cfg_path}tweak\.xml'"
            r"[\s\S]*"
            rf"Outputting text configuration file '{avail_cfg_path}tweak\.cfg'"
        ]

        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=[
                "-G",
                str(config_base) + ".cfg",
                "-F",
                "txt",
                "json",
                "xml",
                "--force",
            ],
            fail_on_not_found=True,
            env=env_overrides,
        )
        self.assert_regex(result, pass_regex=pass_regex)

        config_files = [
            test_output_dir / f"rocprof-sys-tweak.{ext}" for ext in ["cfg", "json", "xml"]
        ]
        self.assert_file_exists(
            config_files, subtest_name="Config file existence validation"
        )

    @pytest.mark.timeout(45)
    def test_format_consistency(self, test_output_dir):
        """Validate that JSON and TXT config formats cover the same env vars.

        Generates both JSON (hierarchical schema) and TXT (flat key=value)
        configs, then verifies that every ROCPROFSYS_* env var in the TXT
        output has a corresponding mapping in the JSON schema — except for
        documented internal/session-specific vars in EXCLUDED_FROM_JSON_SCHEMA.
        """
        config_base = test_output_dir / "format-check"

        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=[
                "-G",
                str(config_base) + ".cfg",
                "-F",
                "txt",
                "json",
                "--force",
            ],
            fail_on_not_found=True,
        )
        self.assert_regex(result)

        txt_file = test_output_dir / "format-check.cfg"
        json_file = test_output_dir / "format-check.json"

        self.assert_file_exists(
            [txt_file, json_file],
            subtest_name="Config files generated",
        )

        # Parse TXT: extract all ROCPROFSYS_* env var names
        txt_vars = set()
        with open(txt_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                match = re.match(r"(ROCPROFSYS_\w+)\s*=", line)
                if match:
                    txt_vars.add(match.group(1))

        # Parse JSON: flatten to find all leaf paths
        with open(json_file) as f:
            json_paths = flatten_json_keys(json.load(f))

        missing = validate_format_consistency(txt_vars, json_paths)
        if missing:
            pytest.fail(
                f"TXT config has {len(missing)} env var(s) without JSON "
                f"schema coverage:\n" + "\n".join(f"  {m}" for m in missing)
            )

    @pytest.mark.timeout(45)
    def test_list_keys(self):
        pass_regex = [r"Output Keys:[\s\S]*%argv%[\s\S]*%argv_hash%"]

        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["--list-keys", "--expand-keys"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=pass_regex)

    @pytest.mark.timeout(45)
    def test_list_keys_markdown(self):
        pass_regex = [r"`%argv%`[\s\S]*`%argv_hash%`"]

        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["--list-keys", "--expand-keys", "--markdown"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=pass_regex)

    @pytest.mark.timeout(45)
    def test_list_categories(self):
        pass_regex = [r" component::[\s\S]* hw_counters::[\s\S]* settings::"]

        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["--list-categories"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=pass_regex)

    @pytest.mark.timeout(45)
    def test_core_categories(self):
        pass_regex = [
            r"ROCPROFSYS_CONFIG_FILE[\s\S]*ROCPROFSYS_ENABLED[\s\S]*"
            r"ROCPROFSYS_SUPPRESS_CONFIG[\s\S]*ROCPROFSYS_SUPPRESS_PARSING[\s\S]*ROCPROFSYS_VERBOSE"
        ]

        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["-c", "core"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=pass_regex)

    @pytest.mark.timeout(45)
    def test_settings_no_gpu(self):
        """Test that settings query works without GPU initialization.

        This test validates the lightweight initialization mode that allows
        rocprof-sys-avail to query settings without initializing the ROCm runtime,
        reducing startup time and allowing operation in environments without GPU/ROCm.
        """
        pass_regex = [r"ROCPROFSYS_TRACE"]

        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["--settings", "--brief"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=pass_regex)

    @pytest.mark.timeout(45)
    def test_components_no_gpu(self):
        """Test that component query works without GPU initialization.

        This test validates that timemory component introspection works without
        requiring GPU/ROCm access, using the lightweight initialization path.
        """
        pass_regex = [r"COMPONENT"]

        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["--components", "--brief"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=pass_regex)

    @pytest.mark.timeout(45)
    def test_settings_description_no_gpu(self):
        """Test that settings with descriptions works without GPU initialization.

        This test validates that detailed settings queries (with descriptions)
        work without GPU/ROCm initialization.
        """
        pass_regex = [r"ROCPROFSYS_OUTPUT_PATH"]

        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["--settings", "--description", "--brief"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=pass_regex)

    @pytest.mark.timeout(45)
    @pytest.mark.gpu
    def test_settings_rocm_available(self, rocprof_config):
        """Test that ROCm-specific settings are present.

        This test validates that settings like ROCPROFSYS_ROCM_DOMAINS and
        ROCPROFSYS_AMD_SMI_METRICS are registered and visible.
        """

        # These settings should always be present when ROCm is enabled
        pass_regex = [
            r"ROCPROFSYS_AMD_SMI_METRICS",
            r"ROCPROFSYS_ROCM_DOMAINS",
        ]

        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["--settings", "--brief"],
            fail_on_not_found=True,
        )
        self.assert_regex(result, pass_regex=pass_regex)


# ============================================================================
# rocprof-sys-run tests
# ============================================================================


@pytest.mark.sys_run
@pytest.mark.class_name("rocprofiler-systems-run")
class TestRocprofilerSystemsRun(RocprofsysTest):
    """Tests for rocprof-sys-run binary."""

    target = "rocprof-sys-run"

    @pytest.mark.timeout(45)
    def test_help(self):
        """Test rocprof-sys-run --help output."""
        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=["--help"],
            fail_on_not_found=True,
        )

        self.assert_regex(result)

    @pytest.mark.timeout(45)
    def test_args(self, test_output_dir):
        """Test rocprof-sys-run with comprehensive arguments."""
        import shutil

        # Check if sleep command exists
        sleep_cmd = shutil.which("sleep")
        if not sleep_cmd:
            pytest.skip("sleep command not found")

        # Create empty config file
        config_dir = test_output_dir / "config"
        config_dir.mkdir(parents=True, exist_ok=True)
        empty_cfg = config_dir / "empty.cfg"
        empty_cfg.write_text("#\n# empty config file\n#\n")

        tmpdir = test_output_dir / "tmpdir"
        tmpdir = tmpdir.resolve()
        tmpdir.mkdir(parents=True, exist_ok=True)

        args = [
            "--monochrome",
            "--debug=false",
            "-v",
            "1",
            "-c",
            str(empty_cfg),
            "-o",
            str(test_output_dir),
            "run-args-output/",
            "-TPHD",
            "-S",
            "cputime",
            "realtime",
            "--trace-wait=1.0e-12",
            "--trace-duration=5.0",
            "--wait=1.0",
            "--duration=3.0",
            "--trace-file=perfetto-run-args-trace.proto",
            "--trace-buffer-size=100",
            "--trace-fill-policy=ring_buffer",
            "--profile-format",
            "console",
            "json",
            "text",
            "--process-freq",
            "1000",
            "--process-wait",
            "0.0",
            "--process-duration",
            "10",
            "--cpus",
            "0-4",
            "--gpus",
            "0",
            "-f",
            "1000",
            "--sampling-wait",
            "1.0",
            "--sampling-duration",
            "10",
            "-t",
            "0-3",
            "--sample-cputime",
            "1000",
            "1.0",
            "0-3",
            "--sample-realtime",
            "10",
            "0.5",
            "0-3",
            "-I",
            "all",
            "-E",
            "mutex-locks",
            "rw-locks",
            "spin-locks",
            "-C",
            "perf::INSTRUCTIONS",
            "--inlines",
            "--hsa-interrupt",
            "0",
            "--use-causal=false",
            "--use-kokkosp",
            "--num-threads-hint=4",
            "--sampling-allocator-size=32",
            "--ci",
            "--dl-verbose=3",
            "--perfetto-annotations=off",
            "--kokkosp-kernel-logger",
            "--kokkosp-name-length-max=1024",
            '--kokkosp-prefix="[kokkos]"',
            "--tmpdir",
            str(tmpdir),
            "--perfetto-backend",
            "inprocess",
            "--use-pid",
            "false",
            "--time-output",
            "off",
            "--thread-pool-size",
            "0",
            "--timemory-components",
            "wall_clock",
            "cpu_clock",
            "peak_rss",
            "page_rss",
            "--fork",
            "--",
            sleep_cmd,
            "5",
        ]

        result = self.run_test(
            "baseline",
            target=self.target,
            run_args=args,
            fail_on_not_found=True,
        )
        self.assert_regex(result)
