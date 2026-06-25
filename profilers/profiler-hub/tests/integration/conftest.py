# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared pytest fixtures for the integration suite: run an example binary that
writes one record to a given DB path, then let the test read that DB back."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import pytest

# This file lives at tests/integration/, so the repo root is two levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]
# Default spot the build drops the example binaries.
EXAMPLE_BIN_DIR = REPO_ROOT / "build" / "bin" / "examples"
DEFAULT_TIMEOUT = 180


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "timeout(seconds): per-test timeout for the generated CTest"
    )


def _timeout() -> int:
    return int(os.environ.get("PHUB_CI_TIMEOUT", DEFAULT_TIMEOUT))


def _module_timeout(items: List[pytest.Item]) -> int:
    """Largest 'timeout' marker among a module's tests, or DEFAULT_TIMEOUT."""
    values = []
    for item in items:
        marker = item.get_closest_marker("timeout")
        if marker and marker.args:
            values.append(int(marker.args[0]))
    return max(values) if values else DEFAULT_TIMEOUT


def pytest_addoption(parser):
    group = parser.getgroup("profiler-hub ctest")
    group.addoption(
        "--ctest-mode",
        action="store",
        default="off",
        choices=("off", "generate"),
        help="Generate CTest definitions from collected pytest tests.",
    )
    group.addoption(
        "--ctest-output-path",
        action="store",
        default=None,
        help="Path to write generated CTest definitions.",
    )


def pytest_collection_finish(session):
    if session.config.getoption("--ctest-mode", default="off") != "generate":
        return

    output_path = session.config.getoption("--ctest-output-path", default=None)
    if not output_path:
        raise pytest.UsageError("--ctest-output-path is required in generate mode")

    _write_ctest_file(session.items, Path(output_path))
    pytest.exit("CTest definitions generated", returncode=0)


def _cmake_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace(";", "\\;")


def _module_ctest_name(path: Path) -> str:
    return f"integration.{path.stem}"


def _write_ctest_file(items: List[pytest.Item], output_path: Path) -> None:
    python_exe = os.environ.get("PHUB_CTEST_PYTHON_EXECUTABLE", "python3")
    source_dir = os.environ.get("PHUB_CTEST_SOURCE_DIR", str(REPO_ROOT))
    example_bin_dir = os.environ.get("PHUB_CTEST_EXAMPLE_BIN_DIR", str(EXAMPLE_BIN_DIR))

    lines = [
        "# CTest definitions from profiler-hub pytest suite",
        "# regenerate by building the generate-profiler-hub-pytest-ctests target",
        "",
        f'set(_PHUB_PYTHON_EXECUTABLE "{_cmake_escape(python_exe)}")',
        f'set(_PHUB_SOURCE_DIR "{_cmake_escape(source_dir)}")',
        f'set(_PHUB_EXAMPLE_BIN_DIR "{_cmake_escape(example_bin_dir)}")',
        "",
        "# PHUB_CI_TIMEOUT (if set) overrides every test's timeout when ctest runs.",
        "if(DEFINED ENV{PHUB_CI_TIMEOUT})",
        '    set(_PHUB_CI_TIMEOUT "$ENV{PHUB_CI_TIMEOUT}")',
        "endif()",
        "",
    ]

    # Group the collected tests by module so each file's timeout marker applies.
    modules: Dict[Path, List[pytest.Item]] = {}
    for item in items:
        modules.setdefault(Path(str(item.path)), []).append(item)

    for path in sorted(modules):
        test_name = _cmake_escape(_module_ctest_name(path))
        rel_path = _cmake_escape(path.relative_to(REPO_ROOT).as_posix())
        module_timeout = _module_timeout(modules[path])
        lines.extend(
            [
                # Default to the module's own timeout; the env var still wins.
                "if(DEFINED _PHUB_CI_TIMEOUT)",
                "    set(_PHUB_TEST_TIMEOUT ${_PHUB_CI_TIMEOUT})",
                "else()",
                f"    set(_PHUB_TEST_TIMEOUT {module_timeout})",
                "endif()",
                f'add_test("{test_name}" "${{_PHUB_PYTHON_EXECUTABLE}}"',
                '    "-m" "pytest" "-q" "-p" "no:cacheprovider"',
                f'    "${{_PHUB_SOURCE_DIR}}/{rel_path}")',
                f'set_tests_properties("{test_name}" PROPERTIES',
                '    WORKING_DIRECTORY "${_PHUB_SOURCE_DIR}"',
                '    ENVIRONMENT "PHUB_EXAMPLE_BIN_DIR=${_PHUB_EXAMPLE_BIN_DIR}"',
                '    LABELS "integration;pytest"',
                "    TIMEOUT ${_PHUB_TEST_TIMEOUT}",
                ")",
                "",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _runnable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _find_launcher(name: str) -> Optional[Path]:
    binary_name = f"profiler-hub_{name}"

    # 1. Honor the env override first.
    env_dir = os.environ.get("PHUB_EXAMPLE_BIN_DIR")
    if env_dir:
        cand = Path(env_dir) / binary_name
        if _runnable(cand):
            return cand

    # 2. Build output location.
    cand = EXAMPLE_BIN_DIR / binary_name
    if _runnable(cand):
        return cand

    # 3. Search for it anywhere in the tree.
    for found in sorted(REPO_ROOT.rglob(binary_name)):
        if _runnable(found):
            return found

    return None


def _parse_kv(stdout: str) -> Dict[str, str]:
    """Parse the launcher's 'key=value' lines into a dict."""
    out: Dict[str, str] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        assert "=" in line, f"malformed launcher output line: {line!r}"
        key, value = line.split("=", 1)
        assert key not in out, (
            f"duplicate key {key!r} in launcher output: previous value "
            f"{out[key]!r}, new value {value!r}"
        )
        out[key] = value
    return out


@pytest.fixture(scope="session")
def run_launcher_db():
    """Helper that runs an example and returns
    the database file it wrote, so the test can read it."""

    def _run(name: str, db_path: Path) -> Path:
        binary = _find_launcher(name)
        if binary is None:
            pytest.fail(
                f"example 'profiler-hub_{name}' not found under "
                f"{EXAMPLE_BIN_DIR} (or $PHUB_EXAMPLE_BIN_DIR)"
            )

        # Start from a clean slate so we never read a stale database.
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if db_path.exists():
            db_path.unlink()

        proc = subprocess.run(
            [str(binary), str(db_path)],
            capture_output=True,
            text=True,
            timeout=_timeout(),
        )
        assert proc.returncode == 0, (
            f"{binary} exited {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

        # The launcher echoes back the path it wrote; make sure it matches.
        output = _parse_kv(proc.stdout)
        assert "db_path" in output, f"missing db_path in launcher output: {proc.stdout}"
        actual_db_path = Path(output["db_path"])
        assert actual_db_path == db_path
        assert actual_db_path.is_file(), f"launcher did not create database: {db_path}"
        return actual_db_path

    return _run
