#!/usr/bin/env python3

import csv
import json
import os
import pytest

from rocprofiler_sdk.pytest_utils.dotdict import dotdict
from rocprofiler_sdk.pytest_utils import collapse_dict_list
from rocprofiler_sdk.pytest_utils.perfetto_reader import PerfettoReader


def pytest_addoption(parser):
    parser.addoption(
        "--json-input",
        action="store",
        default="hip-stream-display/out_results.json",
        help="Input JSON",
    )
    parser.addoption(
        "--pftrace-input",
        action="store",
        default="hip-stream-display/out_results.pftrace",
        help="Input pftrace file",
    )


@pytest.fixture
def json_data(request):
    filename = request.config.getoption("--json-input")
    if not os.path.isfile(filename):
        return pytest.skip("stream tracing unavailable")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))


@pytest.fixture
def pftrace_data(request):
    filename = request.config.getoption("--pftrace-input")
    if not os.path.isfile(filename):
        return pytest.skip("stream tracing unavailable")
    return PerfettoReader(filename).read()[0]
