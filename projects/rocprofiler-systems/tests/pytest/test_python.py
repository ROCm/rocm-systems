# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Python tests.
"""

from __future__ import annotations
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
    # Timemory validation uses hierarchical output with multiple entries at different depths
    PYTHON_SOURCE_TIMEMORY = {
        "metric": "trip_count",
        "file": "trip_count.json",
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
        "counts": [5, 3, 3, 6, 12, 18, 6, 3, 3],
        "depths": [0, 1, 2, 3, 4, 5, 6, 2, 3],
    }

    # Perfetto (cached mode) aggregates entries by name
    PYTHON_SOURCE_PERFETTO = {
        "categories": ["python", "user"],
        "labels": ["main_loop", "run", "fib", "inefficient", "_sum"],
        "counts": [5, 3, 24, 3, 3],
        "depths": [0, 1, 2, 2, 3],
    }

    # Timemory validation for builtin profiling - hierarchical output with multiple entries at different depths
    PYTHON_BUILTIN_TIMEMORY = {
        "metric": "trip_count",
        "file": "trip_count.json",
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

    # Perfetto validation with trace caching aggregates all calls to the same function,
    # so we only expect one entry per unique label rather than hierarchical entries.
    PYTHON_BUILTIN_PERFETTO = {
        "categories": ["python"],
        "labels": [
            "[run][builtin.py:31]",
            "[fib][builtin.py:13]",
            "[inefficient][builtin.py:17]",
        ],
        "counts": [5, 445, 5],
        "depths": [0, 1, 1],
    }

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
            timeout=120,
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

    @pytest.mark.parametrize(
        "annotated",
        [
            pytest.param(False, marks=pytest.mark.rocpd("python_rocpd_env")),
            pytest.param(True, id="annotated"),
        ],
    )
    def test_builtin(
        self, python_version, annotated, python_rocpd_env, python_builtin_rocpd_rules
    ):
        result = self.run_test(
            "python",
            target="builtin.py",
            env=python_rocpd_env,
            profile_args=["-b", "--label", "file", "line"],
            annotated=annotated,
            python_version=python_version,
            run_args=["-v", "10", "-n", "5"],
            timeout=120,
        )
        self.assert_regex(result)
        if not annotated:
            self.assert_file_regex(
                result.output_dir / "trip_count.txt",
                pass_regex=[r"\[inefficient\]\[builtin.py:17\]"],
            )
            self.assert_timemory(
                result,
                file_name=self.PYTHON_BUILTIN_TIMEMORY["file"],
                metric=self.PYTHON_BUILTIN_TIMEMORY["metric"],
                labels=self.PYTHON_BUILTIN_TIMEMORY["labels"],
                counts=self.PYTHON_BUILTIN_TIMEMORY["counts"],
                depths=self.PYTHON_BUILTIN_TIMEMORY["depths"],
            )
            self.assert_perfetto(
                result,
                categories=self.PYTHON_BUILTIN_PERFETTO["categories"],
                labels=self.PYTHON_BUILTIN_PERFETTO["labels"],
                counts=self.PYTHON_BUILTIN_PERFETTO["counts"],
                depths=self.PYTHON_BUILTIN_PERFETTO["depths"],
            )
            self.assert_rocpd(
                result,
                rules_files=python_builtin_rocpd_rules,
            )

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
            timeout=120,
        )
        self.assert_regex(result)
        if not annotated:
            self.assert_file_regex(
                result.output_dir / "trip_count.txt",
                pass_regex=[r"run..noprofile.py."],
                fail_regex=[r"(fib|inefficient)..noprofile.py."],
            )

    @pytest.mark.rocpd("python_rocpd_env")
    def test_python_source(
        self, python_version, python_rocpd_env, python_source_rocpd_rules
    ):
        result = self.run_test(
            "python",
            target="source.py",
            env=python_rocpd_env,
            profile_args=["-v", "5", "-n", "5", "-s", "3"],
            python_version=python_version,
            run_args=["-v", "5", "-n", "5", "-s", "3"],
            standalone=True,
            timeout=120,
        )
        self.assert_regex(result)
        self.assert_timemory(
            result,
            file_name=self.PYTHON_SOURCE_TIMEMORY["file"],
            metric=self.PYTHON_SOURCE_TIMEMORY["metric"],
            labels=self.PYTHON_SOURCE_TIMEMORY["labels"],
            counts=self.PYTHON_SOURCE_TIMEMORY["counts"],
            depths=self.PYTHON_SOURCE_TIMEMORY["depths"],
        )
        self.assert_perfetto(
            result,
            categories=self.PYTHON_SOURCE_PERFETTO["categories"],
            labels=self.PYTHON_SOURCE_PERFETTO["labels"],
            counts=self.PYTHON_SOURCE_PERFETTO["counts"],
            depths=self.PYTHON_SOURCE_PERFETTO["depths"],
        )
        self.assert_rocpd(
            result,
            rules_files=python_source_rocpd_rules,
        )
