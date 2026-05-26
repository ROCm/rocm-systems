#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import csv
import json
import os
from pathlib import Path

import pytest

from attachment_utils import (
    EXECUTE_FAILED_MARKER,
    MPI_SKIP_MARKER,
    PC_SAMPLING_SKIP_MARKER,
)
from rocprofiler_sdk.pytest_utils import collapse_dict_list
from rocprofiler_sdk.pytest_utils.dotdict import dotdict


def pytest_addoption(parser):
    parser.addoption("--test-app", action="store", help="attachment-test binary path")
    parser.addoption("--rocprofv3", action="store", help="rocprofv3 binary path")
    parser.addoption("--output-dir", action="store", help="CTest output directory")
    parser.addoption(
        "--rocprof-log-level",
        action="store",
        default="info",
        help="rocprofv3 log level (not pytest --log-level)",
    )
    parser.addoption(
        "--output-name", action="store", default="out", help="rocprofv3 output base name"
    )
    parser.addoption("--rocpd-input", action="store", help="rocpd database path")
    parser.addoption("--kernel-input", action="store", help="kernel trace CSV path")
    parser.addoption("--skip-if", action="store", help="Skip validation if this file exists")
    parser.addoption("--json-input", action="store", help="rocprofv3 JSON results path")
    parser.addoption("--mpiexec", action="store", default="", help="MPI launcher (mpiexec)")
    parser.addoption(
        "--mpi-numproc-flag",
        action="store",
        default="-n",
        help="MPI process count flag (e.g. -n or -np)",
    )


def _skip_if_needed(request):
    skip_file = request.config.getoption("--skip-if")
    if skip_file and os.path.exists(skip_file):
        pytest.skip("Process attachment tests skipped (insufficient ptrace permissions)")

    json_input = request.config.getoption("--json-input")
    if json_input:
        artifact_parent = Path(json_input).parent
        if (artifact_parent / PC_SAMPLING_SKIP_MARKER).is_file():
            pytest.skip("PC sampling unavailable; skipping validate")
        if (artifact_parent / MPI_SKIP_MARKER).is_file():
            pytest.skip("MPI unavailable; skipping validate")
        if (artifact_parent / EXECUTE_FAILED_MARKER).is_file():
            pytest.skip("Execute phase failed; skipping validate")

    for option in ("rocpd_input", "kernel_input"):
        artifact = request.config.getoption(option)
        if not artifact:
            continue
        failed = Path(artifact).parent / EXECUTE_FAILED_MARKER
        if failed.is_file():
            pytest.skip("Execute phase failed; skipping validate")


@pytest.fixture
def rocpd_input_path(request):
    _skip_if_needed(request)
    path = request.config.getoption("--rocpd-input")
    assert path, "rocpd input path required"
    return path


@pytest.fixture
def kernel_trace_csv_path(request):
    _skip_if_needed(request)
    return request.config.getoption("--kernel-input")


@pytest.fixture
def kernel_input_data(kernel_trace_csv_path):
    if not kernel_trace_csv_path:
        return []
    with open(kernel_trace_csv_path, "r", encoding="utf-8") as inp:
        return list(csv.DictReader(inp))


@pytest.fixture
def json_input_path(request):
    _skip_if_needed(request)
    path = request.config.getoption("--json-input")
    assert path, "json input path required"
    return path


@pytest.fixture
def json_data(json_input_path):
    """Parse full JSON output (PC sampling attach is ~7 MiB)."""
    if not os.path.isfile(json_input_path):
        pytest.fail(f"missing JSON from execute phase: {json_input_path}")
    if os.path.getsize(json_input_path) > 512 * 1024 * 1024:
        pytest.skip("JSON too large for in-memory validation")
    with open(json_input_path, "r", encoding="utf-8") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))
