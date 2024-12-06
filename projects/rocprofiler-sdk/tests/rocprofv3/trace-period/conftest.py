#!/usr/bin/env python3

import pytest
import json
import os

from rocprofiler_sdk.pytest_utils.dotdict import dotdict
from rocprofiler_sdk.pytest_utils import collapse_dict_list
from rocprofiler_sdk.pytest_utils.perfetto_reader import PerfettoReader
from rocprofiler_sdk.pytest_utils.otf2_reader import OTF2Reader


def pytest_addoption(parser):
    parser.addoption(
        "--json-input",
        action="store",
        help="Path to JSON file.",
    )
    parser.addoption(
        "--collection-period-input",
        action="store",
        help="Path to OUTPUT Timestamps file.",
    )
    parser.addoption(
        "--pftrace-input",
        action="store",
        help="Path to Perfetto trace file.",
    )
    parser.addoption(
        "--otf2-input",
        action="store",
        help="Path to OTF2 trace file.",
    )


@pytest.fixture
def json_data(request):
    filename = request.config.getoption("--json-input")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))


@pytest.fixture
def collection_period_data(request):
    filename = request.config.getoption("--collection-period-input")
    with open(filename, "r") as inp:
        data = inp.read()

        # Split the content by '--'
        sections = [section.strip() for section in data.split("--") if section.strip()]

        result = []
        for section in sections:
            section_data = {}
            for line in section.splitlines():
                label, start, stop = [itr.strip() for itr in line.split(":")]
                section_data[label] = {"start": int(start), "stop": int(stop)}

            if section_data:
                result += [dotdict(section_data)]

        return result


@pytest.fixture
def pftrace_data(request):
    filename = request.config.getoption("--pftrace-input")
    return PerfettoReader(filename).read()[0]


@pytest.fixture
def otf2_data(request):
    filename = request.config.getoption("--otf2-input")
    if not os.path.exists(filename):
        raise FileExistsError(f"{filename} does not exist")
    return OTF2Reader(filename).read()[0]
