# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared fixtures for the pytest-driven integration suite.

The model here is: an executable example binary (examples/*) writes records to
a Python-provided rocpd database path, then pytest reads that database directly.

Locating an example binary 'profiler-hub_<name>' (first match wins):
  1. $PHUB_EXAMPLE_BIN_DIR/profiler-hub_<name>, if the env var is set.
  2. '<repo>/build/bin/examples/profiler-hub_<name>', produced by the main build
     (cmake -S . -B build && cmake --build build, with PROFILER_HUB_BUILD_EXAMPLES).
  3. a fallback repo-wide search for the same binary.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

# tests/integration/conftest.py -> repo root is two levels up.
REPO_ROOT = Path(__file__).resolve().parents[2]
# Where the examples CMake places the built executables.
EXAMPLE_BIN_DIR = REPO_ROOT / "build" / "bin" / "examples"


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


def _write_ctest_file(items: list[pytest.Item], output_path: Path) -> None:
    python_exe = os.environ.get("PHUB_CTEST_PYTHON_EXECUTABLE", "python3")
    source_dir = os.environ.get("PHUB_CTEST_SOURCE_DIR", str(REPO_ROOT))
    example_bin_dir = os.environ.get("PHUB_CTEST_EXAMPLE_BIN_DIR", str(EXAMPLE_BIN_DIR))

    lines = [
        "# Auto-generated CTest definitions from profiler-hub pytest suite",
        "# DO NOT EDIT - regenerate by building the generate-profiler-hub-pytest-ctests target",
        "",
        f'set(_PHUB_PYTHON_EXECUTABLE "{_cmake_escape(python_exe)}")',
        f'set(_PHUB_SOURCE_DIR "{_cmake_escape(source_dir)}")',
        f'set(_PHUB_EXAMPLE_BIN_DIR "{_cmake_escape(example_bin_dir)}")',
        "",
    ]

    test_modules = sorted({Path(str(item.path)) for item in items})
    for path in test_modules:
        test_name = _cmake_escape(_module_ctest_name(path))
        rel_path = _cmake_escape(path.relative_to(REPO_ROOT).as_posix())
        lines.extend(
            [
                f'add_test("{test_name}" "${{_PHUB_PYTHON_EXECUTABLE}}"',
                '    "-m" "pytest" "-q" "-p" "no:cacheprovider"',
                f'    "${{_PHUB_SOURCE_DIR}}/{rel_path}")',
                f'set_tests_properties("{test_name}" PROPERTIES',
                f'    WORKING_DIRECTORY "${{_PHUB_SOURCE_DIR}}"',
                f'    ENVIRONMENT "PHUB_EXAMPLE_BIN_DIR=${{_PHUB_EXAMPLE_BIN_DIR}}"',
                '    LABELS "integration;pytest"',
                "    TIMEOUT 120",
                ")",
                "",
            ]
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _find_launcher(name: str) -> Path | None:
    binary_name = f"profiler-hub_{name}"

    env_dir = os.environ.get("PHUB_EXAMPLE_BIN_DIR")
    if env_dir:
        cand = Path(env_dir) / binary_name
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand

    cand = EXAMPLE_BIN_DIR / binary_name
    if cand.is_file() and os.access(cand, os.X_OK):
        return cand

    for found in sorted(REPO_ROOT.rglob(binary_name)):
        if found.is_file() and os.access(found, os.X_OK):
            return found
    return None


def _parse_kv(stdout: str) -> dict[str, str]:
    """Parse 'key=value' lines into a flat dict. Splits on the first '=' only,
    so values may themselves contain '=' or spaces."""
    out: dict[str, str] = {}
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        assert "=" in line, f"malformed launcher output line: {line!r}"
        key, value = line.split("=", 1)
        out[key] = value
    return out


@pytest.fixture(scope="session")
def run_launcher_db():
    """Return a callable run(name, db_path) -> Path for SQLite validation."""

    def _run(name: str, db_path: Path) -> Path:
        binary = _find_launcher(name)
        if binary is None:
            pytest.fail(
                f"example 'profiler-hub_{name}' not found under "
                f"{EXAMPLE_BIN_DIR} (or $PHUB_EXAMPLE_BIN_DIR)"
            )

        db_path.parent.mkdir(parents=True, exist_ok=True)
        if db_path.exists():
            db_path.unlink()

        proc = subprocess.run(
            [str(binary), str(db_path)], capture_output=True, text=True, timeout=120
        )
        assert proc.returncode == 0, (
            f"{binary} exited {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

        output = _parse_kv(proc.stdout)
        assert "db_path" in output, f"missing db_path in launcher output: {proc.stdout}"
        actual_db_path = Path(output["db_path"])
        assert actual_db_path == db_path
        assert actual_db_path.is_file(), f"launcher did not create database: {db_path}"
        return actual_db_path

    return _run
