#!/usr/bin/env python3

import csv
import json
import pytest

from rocprofiler_sdk.pytest_utils.dotdict import dotdict
from rocprofiler_sdk.pytest_utils import collapse_dict_list


def pytest_addoption(parser):
    parser.addoption(
        "--json-input",
        action="store",
        default="scratch-memory-tracing/out_results.json",
        help="Input JSON",
    )
    parser.addoption(
        "--csv-input",
        action="store",
        default="scratch-memory-tracing/out_scratch_memory_trace.csv",
        help="Input CSV",
    )


@pytest.fixture
def json_input_data(request):
    filename = request.config.getoption("--json-input")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))


@pytest.fixture
def csv_input_data(request):
    filename = request.config.getoption("--csv-input")
    data = []
    with open(filename, "r") as inp:
        reader = csv.DictReader(inp)
        for row in reader:
            data.append(row)

    return data
