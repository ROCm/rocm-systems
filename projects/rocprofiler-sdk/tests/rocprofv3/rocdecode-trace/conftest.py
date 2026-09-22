#!/usr/bin/env python3

import csv
import json
import os
import pytest

from rocprofiler_sdk.pytest_utils.dotdict import dotdict
from rocprofiler_sdk.pytest_utils import collapse_dict_list
from rocprofiler_sdk.pytest_utils.perfetto_reader import PerfettoReader
from rocprofiler_sdk.pytest_utils.otf2_reader import OTF2Reader


def pytest_addoption(parser):
    parser.addoption(
        "--json-input",
        action="store",
        default="rocdecode-trace/rocdecode-demo-trace/out_results.json",
        help="Input JSON",
    )
    parser.addoption(
        "--otf2-input",
        action="store",
        default="rocdecode-trace/rocdecode-demo-trace/out_results.otf2",
        help="Input OTF2",
    )
    parser.addoption(
        "--pftrace-input",
        action="store",
        default="rocdecode-trace/rocdecode-demo-trace/out_results.pftrace",
        help="Input pftrace file",
    )
    parser.addoption(
        "--csv-input",
        action="store",
        default="rocdecode-trace/rocdecode-demo-trace/out_rocdecode_api_trace.csv",
        help="Input CSV",
    )


@pytest.fixture
def json_data(request):
    filename = request.config.getoption("--json-input")
    if not os.path.isfile(filename):
        return pytest.skip("rocdecode tracing unavailable")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))


@pytest.fixture
def csv_data(request):
    filename = request.config.getoption("--csv-input")
    data = []
    if not os.path.isfile(filename):
        # The CSV file is not generated, because the dependency test
        # responsible to generate this file was skipped or failed.
        # Thus emit the message to skip this test as well.
        return pytest.skip("rocdecode tracing unavailable")
    else:
        with open(filename, "r") as inp:
            reader = csv.DictReader(inp)
            for row in reader:
                data.append(row)
    return data


@pytest.fixture
def otf2_data(request):
    filename = request.config.getoption("--otf2-input")
    if not os.path.isfile(filename):
        return pytest.skip("rocdecode tracing unavailable")
    return OTF2Reader(filename).read()[0]


@pytest.fixture
def pftrace_data(request):
    filename = request.config.getoption("--pftrace-input")
    if not os.path.isfile(filename):
        return pytest.skip("rocdecode tracing unavailable")
    return PerfettoReader(filename).read()[0]
