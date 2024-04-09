#!/usr/bin/env python3

import csv
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--hsa-trace-input",
        action="store",
        help="Path to HSA API tracing CSV file.",
    )
    parser.addoption(
        "--kernel-trace-input",
        action="store",
        help="Path to Kernel API tracing CSV file.",
    )


@pytest.fixture
def hsa_trace_input_data(request):
    filename = request.config.getoption("--hsa-trace-input")
    data = []
    with open(filename, "r") as inp:
        reader = csv.DictReader(inp)
        for row in reader:
            data.append(row)

    return data


@pytest.fixture
def kernel_trace_input_data(request):
    filename = request.config.getoption("--kernel-trace-input")
    data = []
    with open(filename, "r") as inp:
        reader = csv.DictReader(inp)
        for row in reader:
            data.append(row)

    return data
