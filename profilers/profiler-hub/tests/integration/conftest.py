# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Shared fixtures for the pytest-driven integration suite.

The model here is: a  executable example binary (examples/*) writes
records to a local rocpd database, reads them back, and prints the recovered
fields as "key=value" lines on stdout.

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
def run_launcher():
    """Return a callable run(name) -> dict[dotted-key, str] for an example."""

    def _run(name: str) -> dict[str, str]:
        binary = _find_launcher(name)
        if binary is None:
            pytest.fail(
                f"example 'profiler-hub_{name}' not found under "
                f"{EXAMPLE_BIN_DIR} (or $PHUB_EXAMPLE_BIN_DIR)"
            )

        proc = subprocess.run(
            [str(binary)], capture_output=True, text=True, timeout=120
        )
        assert proc.returncode == 0, (
            f"{binary} exited {proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
        return _parse_kv(proc.stdout)

    return _run
