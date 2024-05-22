#!/usr/bin/env python3

import csv
import pytest
import json

from rocprofiler_sdk.pytest_utils.dotdict import dotdict
from rocprofiler_sdk.pytest_utils import collapse_dict_list


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
    parser.addoption(
        "--json-input",
        action="store",
        help="Path to JSON file.",
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


@pytest.fixture
def json_data(request):
    filename = request.config.getoption("--json-input")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))
