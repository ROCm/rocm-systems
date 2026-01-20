# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Tests rocprof-sys binaries
"""

from __future__ import annotations
import pytest
from pathlib import Path
import os

pytestmark = [pytest.mark.rocprof_binary]


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


class TestInstrumentBinary:
    """Tests for rocprof-sys-instrument binary."""

    target = "rocprof-sys-instrument"

    def test_help(
        self,
        run_test,
        assert_regex,
    ):
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

        result = run_test(
            "baseline",
            target=self.target,
            run_args=["--help"],
            timeout=45,
            fail_on_not_found=True,
        )

        assert_regex(result, pass_regex=pass_regex)

    def test_simulate_ls(
        self,
        run_test,
        assert_regex,
        assert_file_exists,
    ):
        ls_name, ls_args = get_ls_command()

        test_args = [
            "--simulate",
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

        result = run_test(
            "baseline",
            target=self.target,
            run_args=test_args,
            timeout=240,
            fail_on_not_found=True,
        )

        assert_regex(result)
        expected_files_paths = [
            result.output_dir / "instrumentation" / f for f in expected_files
        ]
        assert_file_exists(expected_files_paths)

    def test_simulate_lib(
        self,
        rocprof_config,
        run_test,
        assert_regex,
    ):
        user_lib = rocprof_config.rocprofsys_lib_dir / "librocprof-sys-user.so"
        if not user_lib.exists():
            pytest.fail("librocprof-sys-user.so not found")

        pass_regex = [
            r"\[rocprof-sys\]\[exe\] Runtime instrumentation is not possible![\s\S]*"
            r"\[rocprof-sys\]\[exe\] Switching to binary rewrite mode and assuming '--simulate --all-functions'"
        ]

        result = run_test(
            "baseline",
            target=self.target,
            run_args=["--print-available", "functions", "-v", "2", "--", str(user_lib)],
            timeout=120,
            fail_on_not_found=True,
        )

        assert_regex(result, pass_regex=pass_regex)

    def test_simulate_lib_basename(
        self,
        rocprof_config,
        test_output_dir,
        run_test,
        assert_regex,
    ):
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

        result = run_test(
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
            timeout=120,
            working_directory=tmp_dir,
            fail_on_not_found=True,
        )

        assert_regex(result)

    def test_write_log(
        self,
        run_test,
        assert_regex,
        assert_file_exists,
    ):
        """Test instrument writing to log file."""
        ls_name, ls_args = get_ls_command()

        pass_regex = [r"Opening .*/instrumentation/user\.log"]

        result = run_test(
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
            timeout=120,
            fail_on_not_found=True,
        )

        assert_regex(result, pass_regex=pass_regex)
        assert_file_exists(result.output_dir / "instrumentation" / "user.log")


# ============================================================================
# rocprof-sys-avail tests
# ============================================================================


class TestAvailBinary:
    """Tests for rocprof-sys-avail binary."""

    target = "rocprof-sys-avail"

    def test_help(
        self,
        run_test,
        assert_regex,
    ):
        pass_regex = [
            r"\[rocprof-sys-avail\] Usage:[\s\S]*"
            r"\[DEBUG OPTIONS\][\s\S]*"
            r"\[INFO OPTIONS\][\s\S]*"
            r"\[FILTER OPTIONS\][\s\S]*"
            r"\[COLUMN OPTIONS\][\s\S]*"
            r"\[DISPLAY OPTIONS\][\s\S]*"
            r"\[OUTPUT OPTIONS\][\s\S]*"
        ]

        result = run_test(
            "baseline",
            target=self.target,
            run_args=["--help"],
            timeout=45,
            fail_on_not_found=True,
        )

        assert_regex(result, pass_regex=pass_regex)

    def test_all(
        self,
        run_test,
        assert_regex,
    ):
        result = run_test(
            "baseline",
            target=self.target,
            run_args=["--all"],
            timeout=45,
            fail_on_not_found=True,
        )

        assert_regex(result)

    def test_all_expand_keys(
        self,
        run_test,
        assert_regex,
    ):
        fail_regex = [r"%[a-zA-Z_]%"]

        result = run_test(
            "baseline",
            target=self.target,
            run_args=["--all", "--expand-keys"],
            timeout=45,
            fail_on_not_found=True,
        )

        assert_regex(result, fail_regex=fail_regex)

    def test_all_only_available_alphabetical(
        self,
        run_test,
        test_output_dir,
        assert_regex,
        assert_file_exists,
    ):
        log_file = (
            test_output_dir / "rocprof-sys-avail-all-only-available-alphabetical.log"
        )

        result = run_test(
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
            timeout=45,
            fail_on_not_found=True,
        )

        assert_regex(result)
        assert_file_exists(log_file)

    def test_all_csv(
        self,
        run_test,
        assert_regex,
    ):
        pass_regex = [
            r"COMPONENT#AVAILABLE#VALUE_TYPE#STRING_IDS#FILENAME#DESCRIPTION#CATEGORY#[\s\S]*"
            r"ENVIRONMENT VARIABLE#VALUE#DATA TYPE#DESCRIPTION#CATEGORIES#[\s\S]*"
            r"HARDWARE COUNTER#DEVICE#AVAILABLE#DESCRIPTION#"
        ]

        result = run_test(
            "baseline",
            target=self.target,
            run_args=["--all", "--csv", "--csv-separator", "#"],
            timeout=45,
            fail_on_not_found=True,
        )

        assert_regex(result, pass_regex=pass_regex)

    def test_filter_wall_clock_available(
        self,
        run_test,
        assert_regex,
    ):
        pass_regex = [
            r"\|[-]+\|[\s\S]*"
            r"\|[ ]+COMPONENT[ ]+\|[\s\S]*"
            r"\|[-]+\|[\s\S]*"
            r"\| (wall_clock)[ ]+\|[\s\S]*"
            r"\|[-]+\|"
        ]

        result = run_test(
            "baseline",
            target=self.target,
            run_args=["-r", "wall_clock", "-C", "--available"],
            timeout=45,
            fail_on_not_found=True,
        )

        assert_regex(result, pass_regex=pass_regex)

    def test_category_filter_rocprofiler_systems(
        self,
        run_test,
        assert_regex,
    ):
        pass_regex = [r"ROCPROFSYS_(SETTINGS_DESC|OUTPUT_FILE|OUTPUT_PREFIX)"]
        fail_regex = [
            r"ROCPROFSYS_(ADD_SECONDARY|SCIENTIFIC|PRECISION|MEMORY_PRECISION|TIMING_PRECISION)",
        ]

        result = run_test(
            "baseline",
            target=self.target,
            run_args=["--categories", "settings::rocprofsys", "--brief"],
            timeout=45,
            fail_on_not_found=True,
        )

        assert_regex(result, pass_regex=pass_regex, fail_regex=fail_regex)

    def test_category_filter_timemory(
        self,
        run_test,
        assert_regex,
    ):
        pass_regex = [
            r"ROCPROFSYS_(ADD_SECONDARY|SCIENTIFIC|PRECISION|MEMORY_PRECISION|TIMING_PRECISION)"
        ]
        fail_regex = [r"ROCPROFSYS_(SETTINGS_DESC|OUTPUT_FILE)"]

        result = run_test(
            "baseline",
            target=self.target,
            run_args=["--categories", "settings::timemory", "--brief", "--advanced"],
            timeout=45,
            fail_on_not_found=True,
        )

        assert_regex(result, pass_regex=pass_regex, fail_regex=fail_regex)

    def test_regex_negation(
        self,
        run_test,
        assert_regex,
    ):
        pass_regex = [
            r"ENVIRONMENT VARIABLE,[\s\S]*"
            r"ROCPROFSYS_CI_SKIP_PUSH_POP_CHECK,[\s\S]*"
            r"ROCPROFSYS_THREAD_POOL_SIZE,[\s\S]*"
            r"ROCPROFSYS_USE_PID,"
        ]
        fail_regex = [r"ROCPROFSYS_TRACE"]

        result = run_test(
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
            timeout=45,
            fail_on_not_found=True,
        )

        assert_regex(result, pass_regex=pass_regex, fail_regex=fail_regex)

    def test_write_config(
        self,
        run_test,
        test_output_dir,
        assert_regex,
        assert_file_exists,
    ):
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

        result = run_test(
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
            timeout=45,
            fail_on_not_found=True,
        )

        assert_regex(result, pass_regex=pass_regex)

        config_files = [
            test_output_dir / f"rocprof-sys-test.{ext}" for ext in ["cfg", "json", "xml"]
        ]
        assert_file_exists(config_files, subtest_name="Config file existence validation")

    def test_write_config_tweak(
        self,
        run_test,
        test_output_dir,
        assert_regex,
        assert_file_exists,
    ):
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

        result = run_test(
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
            timeout=45,
            fail_on_not_found=True,
            env=env_overrides,
        )
        assert_regex(result, pass_regex=pass_regex)

        config_files = [
            test_output_dir / f"rocprof-sys-tweak.{ext}" for ext in ["cfg", "json", "xml"]
        ]
        assert_file_exists(config_files, subtest_name="Config file existence validation")

    def test_list_keys(
        self,
        run_test,
        assert_regex,
    ):
        pass_regex = [r"Output Keys:[\s\S]*%argv%[\s\S]*%argv_hash%"]

        result = run_test(
            "baseline",
            target=self.target,
            run_args=["--list-keys", "--expand-keys"],
            timeout=45,
            fail_on_not_found=True,
        )

        assert_regex(result, pass_regex=pass_regex)

    def test_list_keys_markdown(
        self,
        run_test,
        assert_regex,
    ):
        pass_regex = [r"`%argv%`[\s\S]*`%argv_hash%`"]

        result = run_test(
            "baseline",
            target=self.target,
            run_args=["--list-keys", "--expand-keys", "--markdown"],
            timeout=45,
            fail_on_not_found=True,
        )

        assert_regex(result, pass_regex=pass_regex)

    def test_list_categories(
        self,
        run_test,
        assert_regex,
    ):
        pass_regex = [r" component::[\s\S]* hw_counters::[\s\S]* settings::"]

        result = run_test(
            "baseline",
            target=self.target,
            run_args=["--list-categories"],
            timeout=45,
            fail_on_not_found=True,
        )

        assert_regex(result, pass_regex=pass_regex)

    def test_core_categories(
        self,
        run_test,
        assert_regex,
    ):
        pass_regex = [
            r"ROCPROFSYS_CONFIG_FILE[\s\S]*ROCPROFSYS_ENABLED[\s\S]*"
            r"ROCPROFSYS_SUPPRESS_CONFIG[\s\S]*ROCPROFSYS_SUPPRESS_PARSING[\s\S]*ROCPROFSYS_VERBOSE"
        ]

        result = run_test(
            "baseline",
            target=self.target,
            run_args=["-c", "core"],
            timeout=45,
            fail_on_not_found=True,
        )

        assert_regex(result, pass_regex=pass_regex)


# ============================================================================
# rocprof-sys-run tests
# ============================================================================


class TestRunBinary:
    """Tests for rocprof-sys-run binary."""

    target = "rocprof-sys-run"

    def test_help(
        self,
        run_test,
        assert_regex,
    ):
        """Test rocprof-sys-run --help output."""
        result = run_test(
            "baseline",
            target=self.target,
            run_args=["--help"],
            timeout=45,
            fail_on_not_found=True,
        )

        assert_regex(result)

    def test_args(
        self,
        test_output_dir,
        run_test,
        assert_regex,
    ):
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

        result = run_test(
            "baseline",
            target=self.target,
            run_args=args,
            timeout=45,
            fail_on_not_found=True,
        )

        assert_regex(result)
