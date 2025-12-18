# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""
Tests rocprof-sys binaries
"""

from __future__ import annotations

import sys
from pathlib import Path
import os
import re

sys.path.insert(0, str(Path(__file__).parent))

import pytest

from rocprofsys import (
    RocprofsysConfig,
    SysBinaryRunner,
)

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
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
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

        try:
            runner = SysBinaryRunner(
                config=rocprof_config,
                target=self.target,
                output_dir=test_output_dir,
                args=["--help"],
                timeout=45,
                pass_regex=pass_regex,
            )
        except FileNotFoundError:
            pytest.skip(f"{self.target} not built")

        result = runner.run()
        assert result.success, f"Instrument help failed: {result.failure_reason}"

    def test_simulate_ls(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
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

        try:
            runner = SysBinaryRunner(
                config=rocprof_config,
                target=self.target,
                output_dir=test_output_dir,
                args=test_args,
                timeout=240
            )
        except FileNotFoundError:
            pytest.skip(f"{self.target} not built")

        result = runner.run()
        assert result.success, f"Instrument simulate ls failed: {result.failure_reason}"
        for file in expected_files:
            result.assert_file_exists(f"instrumentation/{file}")

    def test_simulate_lib(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
    ):
        user_lib = rocprof_config.rocprofsys_lib_dir / "librocprof-sys-user.so"
        if not user_lib.exists():
            pytest.skip("librocprof-sys-user.so not built")

        pass_regex = [
            r"\[rocprof-sys\]\[exe\] Runtime instrumentation is not possible![\s\S]*"
            r"\[rocprof-sys\]\[exe\] Switching to binary rewrite mode and assuming '--simulate --all-functions'"
        ]

        try:
            runner = SysBinaryRunner(
                config=rocprof_config,
                target=self.target,
                output_dir=test_output_dir,
                args=["--print-available", "functions", "-v", "2", "--", str(user_lib)],
                timeout=120,
                pass_regex=pass_regex,
            )
        except FileNotFoundError:
            pytest.skip(f"{self.target} not built")

        result = runner.run()
        assert result.success, f"Instrument simulate lib failed: {result.failure_reason}"

    def test_simulate_lib_basename(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
    ):
        """Test instrument with library basename (run from lib directory)."""
        lib_basename = "librocprof-sys-user.so"
        user_lib = rocprof_config.rocprofsys_lib_dir / lib_basename
        if not user_lib.exists():
            pytest.skip(f"{lib_basename} not built")

        output_lib = test_output_dir / lib_basename

        try:
            runner = SysBinaryRunner(
                config=rocprof_config,
                target=self.target,
                output_dir=test_output_dir,
                args=[
                    "--print-available", "functions",
                    "-v", "2",
                    "-o", str(output_lib),
                    "--", lib_basename,
                ],
                timeout=120,
                # Run from lib directory so basename can be found
                working_directory=rocprof_config.rocprofsys_lib_dir,
            )
        except FileNotFoundError:
            pytest.skip(f"{self.target} not built")

        result = runner.run()
        assert result.success, f"Instrument simulate lib basename failed: {result.failure_reason}"

    def test_write_log(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
    ):
        """Test instrument writing to log file."""
        ls_name, ls_args = get_ls_command()

        pass_regex = [r"Opening .*/instrumentation/user\.log"]

        try:
            runner = SysBinaryRunner(
                config=rocprof_config,
                target=self.target,
                output_dir=test_output_dir,
                args=[
                    "--print-instrumented", "functions",
                    "-v", "1",
                    "--log-file", "user.log",
                    "--", ls_name, *ls_args,
                ],
                timeout=120,
                pass_regex=pass_regex,
            )
        except FileNotFoundError:
            pytest.skip(f"{self.target} not built")

        result = runner.run()
        assert result.success, f"Instrument write log failed: {result.failure_reason}"
        result.assert_file_exists("instrumentation/user.log")


# ============================================================================
# rocprof-sys-avail tests
# ============================================================================

class TestAvailBinary:
    """Tests for rocprof-sys-avail binary."""
    target = "rocprof-sys-avail"

    def test_help(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
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

        try:
            runner = SysBinaryRunner(
                config=rocprof_config,
                target=self.target,
                output_dir=test_output_dir,
                args=["--help"],
                timeout=45,
                pass_regex=pass_regex,
            )
        except FileNotFoundError:
            pytest.skip(f"{self.target} not built")

        result = runner.run()
        assert result.success, f"Avail help failed: {result.failure_reason}"

    def test_all(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
    ):
        try:
            runner = SysBinaryRunner(
                config=rocprof_config,
                target=self.target,
                output_dir=test_output_dir,
                args=["--all"],
                timeout=45,
            )
        except FileNotFoundError:
            pytest.skip(f"{self.target} not built")

        result = runner.run()
        assert result.success, f"Avail all failed: {result.failure_reason}"

    def test_all_expand_keys(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
    ):
        fail_regex = [r"%[a-zA-Z_]%|ROCPROFSYS_ABORT_FAIL_REGEX"]
        try:
            runner = SysBinaryRunner(
                config=rocprof_config,
                target=self.target,
                output_dir=test_output_dir,
                args=["--all", "--expand-keys"],
                timeout=45,
                fail_regex=fail_regex,
            )
        except FileNotFoundError:
            pytest.skip(f"{self.target} not built")

        result = runner.run()
        assert result.success, f"Avail all expand keys failed: {result.failure_reason}"

    # TOCHECK everything below this point
    def test_all_only_available_alphabetical(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
    ):
        log_file = test_output_dir / "rocprof-sys-avail-all-only-available-alphabetical.log"

        try:
            runner = SysBinaryRunner(
                config=rocprof_config,
                target=self.target,
                output_dir=test_output_dir,
                args=[
                    "--all", "--available", "--alphabetical", "--debug",
                    "--output", str(log_file),
                ],
                timeout=45,
            )
        except FileNotFoundError:
            pytest.skip(f"{self.target} not built")

        result = runner.run()
        assert result.success, f"Avail all only available alphabetical failed: {result.failure_reason}"
        assert log_file.exists(), f"Log file not created: {log_file}"

    def test_all_csv(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
    ):
        pass_regex = [
            r"COMPONENT#AVAILABLE#VALUE_TYPE#STRING_IDS#FILENAME#DESCRIPTION#CATEGORY#[\s\S]*"
            r"ENVIRONMENT VARIABLE#VALUE#DATA TYPE#DESCRIPTION#CATEGORIES#[\s\S]*"
            r"HARDWARE COUNTER#DEVICE#AVAILABLE#DESCRIPTION#"
        ]

        try:
            runner = SysBinaryRunner(
                config=rocprof_config,
                target=self.target,
                output_dir=test_output_dir,
                args=["--all", "--csv", "--csv-separator", "#"],
                timeout=45,
                pass_regex=pass_regex,
            )
        except FileNotFoundError:
            pytest.skip(f"{self.target} not built")

        result = runner.run()
        assert result.success, f"Avail all csv failed: {result.failure_reason}"

    def test_filter_wall_clock_available(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
    ):
        pass_regex = [
            r"\|[-]+\|[\s\S]*"
            r"\|[ ]+COMPONENT[ ]+\|[\s\S]*"
            r"\|[-]+\|[\s\S]*"
            r"\| (wall_clock)[ ]+\|[\s\S]*"
            r"\|[-]+\|"
        ]

        try:
            runner = SysBinaryRunner(
                config=rocprof_config,
                target=self.target,
                output_dir=test_output_dir,
                args=["-r", "wall_clock", "-C", "--available"],
                timeout=45,
                pass_regex=pass_regex,
            )
        except FileNotFoundError:
            pytest.skip(f"{self.target} not built")

        result = runner.run()
        assert result.success, f"Avail filter wall_clock available failed: {result.failure_reason}"

    def test_category_filter_rocprofiler_systems(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
    ):
        pass_regex = [r"ROCPROFSYS_(SETTINGS_DESC|OUTPUT_FILE|OUTPUT_PREFIX)"]
        fail_regex = [
            r"ROCPROFSYS_(ADD_SECONDARY|SCIENTIFIC|PRECISION|MEMORY_PRECISION|TIMING_PRECISION)",
            "|ROCPROFSYS_ABORT_FAIL_REGEX",
        ]

        try:
            runner = SysBinaryRunner(
                config=rocprof_config,
                target=self.target,
                output_dir=test_output_dir,
                args=["--categories", "settings::rocprofsys", "--brief"],
                timeout=45,
                pass_regex=pass_regex,
                fail_regex=fail_regex,
            )
        except FileNotFoundError:
            pytest.skip(f"{self.target} not built")

        result = runner.run()
        assert result.success, f"Avail category filter rocprofiler-systems failed: {result.failure_reason}"

    def test_category_filter_timemory(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
    ):
        pass_regex = [r"ROCPROFSYS_(ADD_SECONDARY|SCIENTIFIC|PRECISION|MEMORY_PRECISION|TIMING_PRECISION)"]
        fail_regex = [
            r"ROCPROFSYS_(SETTINGS_DESC|OUTPUT_FILE)",
            "|ROCPROFSYS_ABORT_FAIL_REGEX",
        ]

        try:
            runner = SysBinaryRunner(
                config=rocprof_config,
                target=self.target,
                output_dir=test_output_dir,
                args=["--categories", "settings::timemory", "--brief", "--advanced"],
                timeout=45,
                pass_regex=pass_regex,
                fail_regex=fail_regex,
            )
        except FileNotFoundError:
            pytest.skip(f"{self.target} not built")

        result = runner.run()
        assert result.success, f"Avail category filter timemory failed: {result.failure_reason}"

    def test_regex_negation(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
    ):
        pass_regex = [
            r"ENVIRONMENT VARIABLE,[\s\S]*"
            r"ROCPROFSYS_CI_SKIP_PUSH_POP_CHECK,[\s\S]*"
            r"ROCPROFSYS_THREAD_POOL_SIZE,[\s\S]*"
            r"ROCPROFSYS_USE_PID,"
        ]
        fail_regex = [r"ROCPROFSYS_TRACE", "|ROCPROFSYS_ABORT_FAIL_REGEX"]

        try:
            runner = SysBinaryRunner(
                config=rocprof_config,
                target=self.target,
                output_dir=test_output_dir,
                args=[
                    "-R", "rocprofsys", "~timemory",
                    "-r", "_P", "~PERFETTO", "~PROCESS_SAMPLING", "~KOKKOSP", "~PAGE",
                    "--csv", "--brief", "--advanced",
                ],
                timeout=45,
                pass_regex=pass_regex,
                fail_regex=fail_regex,
            )
        except FileNotFoundError:
            pytest.skip(f"{self.target} not built")

        result = runner.run()
        assert result.success, f"Avail regex negation failed: {result.failure_reason}"

    def test_write_config(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
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

        try:
            runner = SysBinaryRunner(
                config=rocprof_config,
                target=self.target,
                output_dir=test_output_dir,
                args=[
                    "-G", str(config_base) + ".cfg",
                    "-F", "txt", "json", "xml",
                    "--force", "--all", "-c", "rocprofsys",
                ],
                timeout=45,
                pass_regex=pass_regex,
            )
        except FileNotFoundError:
            pytest.skip(f"{self.target} not built")

        result = runner.run()
        assert result.success, f"Avail write config failed: {result.failure_reason}"

        # Verify config files were created
        for ext in ["cfg", "json", "xml"]:
            config_file = test_output_dir / f"rocprof-sys-test.{ext}"
            assert config_file.exists(), f"Config file not created: {config_file}"

    def test_write_config_tweak(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
    ):
        config_base = test_output_dir / "rocprof-sys-tweak"

        env_overrides = {
            "ROCPROFSYS_TRACE": "OFF",
            "ROCPROFSYS_PROFILE": "ON",
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

        try:
            runner = SysBinaryRunner(
                config=rocprof_config,
                target=self.target,
                output_dir=test_output_dir,
                args=[
                    "-G", str(config_base) + ".cfg",
                    "-F", "txt", "json", "xml",
                    "--force",
                ],
                env=env_overrides,
                timeout=45,
                pass_regex=pass_regex,
            )
        except FileNotFoundError:
            pytest.skip(f"{self.target} not built")

        result = runner.run()
        assert result.success, f"Avail write config tweak failed: {result.failure_reason}"

        # Verify config files were created
        for ext in ["cfg", "json", "xml"]:
            config_file = test_output_dir / f"rocprof-sys-tweak.{ext}"
            assert config_file.exists(), f"Config file not created: {config_file}"

    def test_list_keys(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
    ):
        pass_regex = [r"Output Keys:[\s\S]*%argv%[\s\S]*%argv_hash%"]

        try:
            runner = SysBinaryRunner(
                config=rocprof_config,
                target=self.target,
                output_dir=test_output_dir,
                args=["--list-keys", "--expand-keys"],
                timeout=45,
                pass_regex=pass_regex,
            )
        except FileNotFoundError:
            pytest.skip(f"{self.target} not built")

        result = runner.run()
        assert result.success, f"Avail list keys failed: {result.failure_reason}"

    def test_list_keys_markdown(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
    ):
        pass_regex = [r"`%argv%`[\s\S]*`%argv_hash%`"]

        try:
            runner = SysBinaryRunner(
                config=rocprof_config,
                target=self.target,
                output_dir=test_output_dir,
                args=["--list-keys", "--expand-keys", "--markdown"],
                timeout=45,
                pass_regex=pass_regex,
            )
        except FileNotFoundError:
            pytest.skip(f"{self.target} not built")

        result = runner.run()
        assert result.success, f"Avail list keys markdown failed: {result.failure_reason}"

    def test_list_categories(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
    ):
        pass_regex = [r" component::[\s\S]* hw_counters::[\s\S]* settings::"]

        try:
            runner = SysBinaryRunner(
                config=rocprof_config,
                target=self.target,
                output_dir=test_output_dir,
                args=["--list-categories"],
                timeout=45,
                pass_regex=pass_regex,
            )
        except FileNotFoundError:
            pytest.skip(f"{self.target} not built")

        result = runner.run()
        assert result.success, f"Avail list categories failed: {result.failure_reason}"

    def test_core_categories(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
    ):
        pass_regex = [
            r"ROCPROFSYS_CONFIG_FILE[\s\S]*ROCPROFSYS_ENABLED[\s\S]*"
            r"ROCPROFSYS_SUPPRESS_CONFIG[\s\S]*ROCPROFSYS_SUPPRESS_PARSING[\s\S]*ROCPROFSYS_VERBOSE"
        ]

        try:
            runner = SysBinaryRunner(
                config=rocprof_config,
                target=self.target,
                output_dir=test_output_dir,
                args=["-c", "core"],
                timeout=45,
                pass_regex=pass_regex,
            )
        except FileNotFoundError:
            pytest.skip(f"{self.target} not built")

        result = runner.run()
        assert result.success, f"Avail core categories failed: {result.failure_reason}"


# ============================================================================
# rocprof-sys-run tests
# ============================================================================

class TestRunBinary:
    """Tests for rocprof-sys-run binary."""
    target = "rocprof-sys-run"

    def test_help(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
    ):
        """Test rocprof-sys-run --help output."""
        try:
            runner = SysBinaryRunner(
                config=rocprof_config,
                target=self.target,
                output_dir=test_output_dir,
                args=["--help"],
                timeout=45,
            )
        except FileNotFoundError:
            pytest.skip(f"{self.target} not built")

        result = runner.run()
        assert result.success, f"Run help failed: {result.failure_reason}"

    def test_args(
        self,
        rocprof_config: RocprofsysConfig,
        test_output_dir: Path,
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
        tmpdir.mkdir(parents=True, exist_ok=True)

        args = [
            "--monochrome",
            "--debug=false",
            "-v", "1",
            "-c", str(empty_cfg),
            "-o", str(test_output_dir), "run-args-output/",
            "-TPHD",
            "-S", "cputime", "realtime",
            "--trace-wait=1.0e-12",
            "--trace-duration=5.0",
            "--wait=1.0",
            "--duration=3.0",
            "--trace-file=perfetto-run-args-trace.proto",
            "--trace-buffer-size=100",
            "--trace-fill-policy=ring_buffer",
            "--profile-format", "console", "json", "text",
            "--process-freq", "1000",
            "--process-wait", "0.0",
            "--process-duration", "10",
            "--cpus", "0-4",
            "--gpus", "0",
            "-f", "1000",
            "--sampling-wait", "1.0",
            "--sampling-duration", "10",
            "-t", "0-3",
            "--sample-cputime", "1000", "1.0", "0-3",
            "--sample-realtime", "10", "0.5", "0-3",
            "-I", "all",
            "-E", "mutex-locks", "rw-locks", "spin-locks",
            "-C", "perf::INSTRUCTIONS",
            "--inlines",
            "--hsa-interrupt", "0",
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
            "--tmpdir", str(tmpdir),
            "--perfetto-backend", "inprocess",
            "--use-pid", "false",
            "--time-output", "off",
            "--thread-pool-size", "0",
            "--timemory-components", "wall_clock", "cpu_clock", "peak_rss", "page_rss",
            "--fork",
            "--", sleep_cmd, "5",
        ]

        try:
            runner = SysBinaryRunner(
                config=rocprof_config,
                target=self.target,
                output_dir=test_output_dir,
                args=args,
                timeout=45,
            )
        except FileNotFoundError:
            pytest.skip(f"{self.target} not built")

        result = runner.run()
        assert result.success, f"Run args failed: {result.failure_reason}"
