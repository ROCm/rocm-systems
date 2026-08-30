# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Python tests.
"""

from __future__ import annotations
import os
import subprocess
import sys
import pytest
from conftest import RocprofsysTest
from pathlib import Path

pytestmark = [pytest.mark.python]

# =============================================================================
# Python fixtures
# =============================================================================


@pytest.fixture
def python_rocpd_env() -> dict[str, str]:
    return {}


@pytest.fixture
def python_source_rocpd_rules(validation_rules_dir: Path) -> list[Path]:
    rules_dir = validation_rules_dir / "python"
    return [
        rules_dir / "python-source-rules.json",
    ]


@pytest.fixture
def python_builtin_rocpd_rules(validation_rules_dir: Path) -> list[Path]:
    rules_dir = validation_rules_dir / "python"
    return [
        rules_dir / "python-builtin-rules.json",
    ]


@pytest.fixture
def python_builtin_annotated_rocpd_rules(validation_rules_dir: Path) -> list[Path]:
    rules_dir = validation_rules_dir / "python"
    return [
        rules_dir / "python-builtin-annotated-rules.json",
    ]


@pytest.fixture(scope="session")
def get_cat_command() -> list[str]:
    """Get a command to concatenate files (like Unix cat).

    Uses 'cmake -E cat' if available, otherwise falls back to system 'cat'.
    """
    import shutil
    import subprocess

    # Try cmake -E cat first (available in CMake 3.18+)
    cmake_exe = shutil.which("cmake")
    if cmake_exe:
        try:
            result = subprocess.run(
                [cmake_exe, "-E", "cat", "--help"], capture_output=True, timeout=5
            )
            # cmake -E cat returns 0 even for --help
            if result.returncode == 0:
                return [cmake_exe, "-E", "cat"]
        except (subprocess.TimeoutExpired, subprocess.SubprocessError):
            pass

    # Fall back to system cat
    cat_exe = shutil.which("cat")
    if cat_exe:
        return [cat_exe]

    pytest.skip("No cat command available (neither 'cmake -E cat' nor 'cat')")


# =============================================================================
# Python tests
# =============================================================================


@pytest.mark.python_versions
class TestPython(RocprofsysTest):

    PYTHON_SOURCE_GENERAL = {
        "metric": "trip_count",  # Timemory
        "file": "trip_count.json",  # Timemory
        "categories": ["python", "rocm_marker_api"],  # Perfetto
        "labels": [
            "main_loop",
            "run",
            "fib",
            "fib",
            "fib",
            "fib",
            "fib",
            "inefficient",
            "_sum",
        ],
        # main_loop=4 as roctx.profilerPause() called mid 4th iteration
        # supresses all marker writes from that point on.
        # This includes the RoctxRange wrapper for 5th iteration
        "counts": [4, 3, 3, 6, 12, 18, 6, 3, 3],
        "depths": [0, 1, 2, 3, 4, 5, 6, 2, 3],
    }

    PYTHON_BUILTIN_GENERAL = {
        "metric": "trip_count",  # Timemory
        "file": "trip_count.json",  # Timemory
        "categories": ["python"],  # Perfetto
        "labels": [
            "[run][builtin.py:31]",
            "[fib][builtin.py:13]",
            "[fib][builtin.py:13]",
            "[fib][builtin.py:13]",
            "[fib][builtin.py:13]",
            "[fib][builtin.py:13]",
            "[fib][builtin.py:13]",
            "[fib][builtin.py:13]",
            "[fib][builtin.py:13]",
            "[fib][builtin.py:13]",
            "[fib][builtin.py:13]",
            "[inefficient][builtin.py:17]",
        ],
        "counts": [5, 5, 10, 20, 40, 80, 160, 260, 220, 80, 10, 5],
        "depths": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1],
    }

    # Per-function debug annotations that -a/--annotate-trace attaches to every
    # profiled python region (see libpyrocprofsys.cpp's config::annotations).
    PYTHON_ANNOTATE_DEBUG_KEYS = [
        "file",
        "line",
        "lasti",
        "argcount",
        "nlocals",
        "stacksize",
    ]

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize(
        "annotated, exclude",
        [
            pytest.param(False, False, id="base"),
            pytest.param(True, False, id="annotated"),
            pytest.param(False, True, id="inefficient"),
            pytest.param(True, True, id="inefficient-annotated"),
        ],
    )
    def test_external(self, python_version, annotated, exclude):
        if exclude:
            profile_args = ["-E", "^inefficient$"]
        else:
            profile_args = ["--label", "file"]
        result = self.run_test(
            "python",
            target="external.py",
            profile_args=profile_args,
            annotated=annotated,
            python_version=python_version,
            run_args=["-v", "10", "-n", "5"],
        )
        self.assert_regex(result)

        if not annotated and not exclude:
            file_pass_regex = [
                r"(\[compile\]).*"
                r"(\| \|0>>> \[run\]\[external.py\]).*"
                r"(\| \|0>>> \|_\[fib\]\[external.py\]).*"
                r"(\| \|0>>> \|_\[inefficient\]\[external.py\])"
            ]
            self.assert_file_regex(
                result.output_dir / "trip_count.txt",
                pass_regex=file_pass_regex,
                fail_regex=[r"(\|_inefficient).*(\|_sum)"],
            )
        elif not annotated and exclude:
            self.assert_file_regex(
                result.output_dir / "trip_count.txt",
                fail_regex=[r"(\|_inefficient).*(\|_sum)"],
            )

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize(
        "annotated",
        [
            pytest.param(False, marks=pytest.mark.rocpd("python_rocpd_env")),
            pytest.param(
                True, id="annotated", marks=pytest.mark.rocpd("python_rocpd_env")
            ),
        ],
    )
    def test_builtin(
        self,
        python_version,
        annotated,
        python_rocpd_env,
        python_builtin_rocpd_rules,
        python_builtin_annotated_rocpd_rules,
    ):
        result = self.run_test(
            "python",
            target="builtin.py",
            env=python_rocpd_env,
            profile_args=["-b", "--label", "file", "line"],
            annotated=annotated,
            python_version=python_version,
            run_args=["-v", "10", "-n", "5"],
        )
        self.assert_regex(result)
        if not annotated:
            self.assert_file_regex(
                result.output_dir / "trip_count.txt",
                pass_regex=[r"\[inefficient\]\[builtin.py:17\]"],
            )
            self.assert_timemory(
                result,
                file_name=self.PYTHON_BUILTIN_GENERAL["file"],
                metric=self.PYTHON_BUILTIN_GENERAL["metric"],
                labels=self.PYTHON_BUILTIN_GENERAL["labels"],
                counts=self.PYTHON_BUILTIN_GENERAL["counts"],
                depths=self.PYTHON_BUILTIN_GENERAL["depths"],
            )
            self.assert_perfetto(
                result,
                categories=self.PYTHON_BUILTIN_GENERAL["categories"],
                labels=self.PYTHON_BUILTIN_GENERAL["labels"],
                counts=self.PYTHON_BUILTIN_GENERAL["counts"],
                depths=self.PYTHON_BUILTIN_GENERAL["depths"],
            )
            self.assert_rocpd(
                result,
                rules_files=python_builtin_rocpd_rules,
            )
            # regression: without -a, no per-function debug annotations should
            # reach the trace (see test below for the annotated=True counterpart)
            self.assert_perfetto(
                result,
                key_names=self.PYTHON_ANNOTATE_DEBUG_KEYS,
                key_counts=[0] * len(self.PYTHON_ANNOTATE_DEBUG_KEYS),
            )
        else:
            # regression: -a/--annotate-trace annotations were dropped by
            # trace-cache replay (only debug.begin_ns/debug.corr_id ever reached
            # the perfetto trace, identical with or without -a). Every profiled
            # python region carries all six annotation fields, so each key's
            # count must equal the total number of profiled slices.
            total_slices = sum(self.PYTHON_BUILTIN_GENERAL["counts"])
            self.assert_perfetto(
                result,
                key_names=self.PYTHON_ANNOTATE_DEBUG_KEYS,
                key_counts=[total_slices] * len(self.PYTHON_ANNOTATE_DEBUG_KEYS),
            )
            self.assert_rocpd(
                result,
                rules_files=python_builtin_annotated_rocpd_rules,
            )

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize(
        "annotated",
        [
            False,
            pytest.param(True, id="annotated"),
        ],
    )
    def test_builtin_noprofile(self, python_version, annotated):
        result = self.run_test(
            "python",
            target="noprofile.py",
            profile_args=["-b", "--label", "file"],
            annotated=annotated,
            python_version=python_version,
            run_args=["-v", "15", "-n", "5"],
        )
        self.assert_regex(result)
        if not annotated:
            self.assert_file_regex(
                result.output_dir / "trip_count.txt",
                pass_regex=[r"run..noprofile.py."],
                fail_regex=[r"(fib|inefficient)..noprofile.py."],
            )

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize(
        "use_cli_flag, use_env_var",
        [
            pytest.param(False, False, id="no-flag-no-env"),
            pytest.param(False, True, id="no-flag-env-set"),
            pytest.param(True, False, id="flag-no-env"),
            pytest.param(True, True, id="flag-and-env"),
        ],
    )
    def test_config_option(
        self,
        python_version,
        create_config_file,
        use_cli_flag,
        use_env_var,
        rocprof_config,
    ):
        """Check that a config file is honored whether it comes from -c/--config or
        the ROCPROFSYS_CONFIG_FILE env var. We skip the base env (otherwise its env
        vars would override the file), then have -c turn profiling on and the env var
        turn tracing on, and confirm each shows up in the produced artifacts.

        Note: rocpd is the default output format (ROCPROFSYS_TRACE defaults to false),
        so this test uses ROCPROFSYS_CONFIG_FILE to *enable* tracing rather than to
        disable it (as was done when Perfetto was the default).
        """
        env: dict[str, str] = {
            # Keep artifacts at the top of output_dir with stable names (no PID
            # suffix, no timestamped subdir) so the file checks below can find them.
            "ROCPROFSYS_USE_PID": "OFF",
            "ROCPROFSYS_TIME_OUTPUT": "OFF",
            "ROCPROFSYS_TIMEMORY_COMPONENTS": "wall_clock,trip_count",
            "PYTHONPATH": str(rocprof_config.rocprofsys_site_packages or ""),
        }
        profile_args = ["--label", "file"]

        if use_env_var:
            env_config = create_config_file({"ROCPROFSYS_TRACE": "ON"}, "env_config.cfg")
            env["ROCPROFSYS_CONFIG_FILE"] = str(env_config)

        if use_cli_flag:
            cli_config = create_config_file(
                {"ROCPROFSYS_PROFILE": "ON"}, "cli_config.cfg"
            )
            profile_args = profile_args + ["-c", str(cli_config)]

        result = self.run_test(
            "python",
            target="external.py",
            env=env,
            profile_args=profile_args,
            python_version=python_version,
            run_args=["-v", "10", "-n", "5"],
            no_base_env=True,
        )
        self.assert_regex(result)

        trip_count_exists = (result.output_dir / "trip_count.json").exists()
        trace_exists = result.perfetto_file is not None

        if use_cli_flag:
            assert trip_count_exists, (
                "trip_count.json should be present: the -c config sets "
                "ROCPROFSYS_PROFILE=ON, so its absence means -c was ignored"
            )
        else:
            assert not trip_count_exists, (
                "trip_count.json should be absent: profiling is off by default "
                "and no -c config enabled it"
            )

        if use_env_var:
            assert trace_exists, (
                "perfetto trace should be present: the ROCPROFSYS_CONFIG_FILE "
                "config sets ROCPROFSYS_TRACE=ON, so its absence means the env "
                "var was ignored"
            )
        else:
            assert not trace_exists, (
                "perfetto trace should be absent: tracing is off by default "
                "(rocpd is the default output format) and no ROCPROFSYS_CONFIG_FILE "
                "config enabled it"
            )

    @pytest.mark.rocpd("python_rocpd_env")
    def test_source(
        self, python_version, python_rocpd_env, python_source_rocpd_rules, rocprof_config
    ):
        if not rocprof_config.capabilities.roctx_available_for(python_version):
            pytest.skip(
                f"roctx Python bindings not installed for Python {python_version}"
            )
        result = self.run_test(
            "python",
            target="source.py",
            env=python_rocpd_env,
            python_version=python_version,
            run_args=["-v", "5", "-n", "5", "-s", "3"],
            standalone=True,
        )
        self.assert_regex(result)
        self.assert_timemory(
            result,
            file_name=self.PYTHON_SOURCE_GENERAL["file"],
            metric=self.PYTHON_SOURCE_GENERAL["metric"],
            labels=self.PYTHON_SOURCE_GENERAL["labels"],
            counts=self.PYTHON_SOURCE_GENERAL["counts"],
            depths=self.PYTHON_SOURCE_GENERAL["depths"],
        )
        self.assert_perfetto(
            result,
            categories=self.PYTHON_SOURCE_GENERAL["categories"],
            labels=self.PYTHON_SOURCE_GENERAL["labels"],
            counts=self.PYTHON_SOURCE_GENERAL["counts"],
            depths=self.PYTHON_SOURCE_GENERAL["depths"],
        )
        self.assert_rocpd(
            result,
            rules_files=python_source_rocpd_rules,
        )


# =============================================================================
# Frontend regressions
# =============================================================================


class TestPythonFrontend:
    """CLI-level regressions in the rocprof-sys-python wrapper and package."""

    TIMEOUT_SEC = 120

    def _python_executable(self, rocprof_config) -> str:
        """Interpreter the bindings were built for.

        The wrapper otherwise falls back to whatever `python3` resolves to, which
        on some distributions is a system Python with no libpyrocprofsys module.
        """
        executables = rocprof_config.capabilities.supported_python_executables
        return str(executables[0]) if executables else sys.executable

    def _run_wrapper(
        self,
        rocprof_config,
        args: list[str],
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            [str(rocprof_config.rocprofsys_python), *args],
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PYTHON_EXECUTABLE": self._python_executable(rocprof_config),
                **(extra_env or {}),
            },
            timeout=self.TIMEOUT_SEC,
        )

    def _run_probe(
        self, rocprof_config, source: str, extra_env: dict[str, str] | None = None
    ) -> str:
        result = subprocess.run(
            [self._python_executable(rocprof_config), "-c", source],
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PYTHONPATH": str(rocprof_config.rocprofsys_site_packages),
                **(extra_env or {}),
            },
            timeout=self.TIMEOUT_SEC,
        )
        lines = result.stdout.splitlines()
        assert lines, f"probe produced no stdout; stderr:\n{result.stderr}"
        return lines[-1]

    @pytest.mark.parametrize("log_level", ["debug", "trace"])
    def test_help_survives_verbose_logging(self, rocprof_config, log_level: str) -> None:
        result = self._run_wrapper(
            rocprof_config, ["--help"], {"ROCPROFSYS_LOG_LEVEL": log_level}
        )
        assert result.returncode == 0, result.stderr

    def test_import_does_not_load_profiling_runtime(self, rocprof_config) -> None:
        loaded = self._run_probe(
            rocprof_config,
            "import rocprofsys\n"
            "with open('/proc/self/maps') as maps:\n"
            "    print(sum('librocprof-sys.so' in line for line in maps))\n",
            {"ROCPROFSYS_LOG_LEVEL": "debug"},
        )
        assert loaded == "0", "importing rocprofsys loaded the profiling runtime"

    def test_missing_script_is_reported(self, rocprof_config) -> None:
        result = self._run_wrapper(rocprof_config, ["--"])
        assert result.returncode != 0
        assert "Could not determine input script" in result.stderr

    def test_banner_names_the_project(self, rocprof_config) -> None:
        result = self._run_wrapper(rocprof_config, ["--help"])
        assert result.returncode == 0, result.stderr
        assert "rocprofiler-systems :: executing" in result.stdout

    def test_library_path_points_at_the_runtime(self, rocprof_config) -> None:
        library_path = self._run_probe(
            rocprof_config,
            "import os, rocprofsys\nprint(os.environ.get('ROCPROFSYS_PATH', ''))\n",
        )
        assert library_path, "rocprofsys did not export ROCPROFSYS_PATH"
        assert (Path(library_path) / "librocprof-sys-dl.so").exists()
