#!/usr/bin/env python3

import csv
import glob
import pytest


def pytest_addoption(parser):
    """Add custom command line options for the fork test."""
    parser.addoption(
        "--output-dir",
        action="store",
        help="Output directory containing profiler database files",
    )
    parser.addoption(
        "--kernel-trace-pattern",
        action="store",
        help="Glob pattern for kernel trace CSV files",
    )


@pytest.fixture
def output_dir(request):
    """Fixture to provide the output directory to tests."""
    output_dir = request.config.getoption("--output-dir")
    if not output_dir:
        pytest.skip(
            "output_dir fixture requires --output-dir to be provided "
            "when running this test outside the CTest harness."
        )
    return output_dir


@pytest.fixture
def kernel_trace_files(request):
    """Fixture to provide list of kernel trace CSV files."""
    pattern = request.config.getoption("--kernel-trace-pattern")
    if not pattern:
        pytest.skip(
            "kernel_trace_files fixture requires --kernel-trace-pattern to be provided "
            "when running this test outside the CTest harness."
        )
    files = glob.glob(pattern)
    return files


@pytest.fixture
def kernel_trace_data(kernel_trace_files):
    """Fixture to provide kernel trace data from all CSV files."""
    all_data = {}
    for filename in kernel_trace_files:
        data = []
        with open(filename, "r") as inp:
            reader = csv.DictReader(inp)
            for row in reader:
                data.append(row)
        all_data[filename] = data
    return all_data
