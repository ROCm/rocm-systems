#!/usr/bin/env python3

import csv
import pytest
import json

from rocprofiler_sdk.pytest_utils.dotdict import dotdict
from rocprofiler_sdk.pytest_utils import collapse_dict_list
from rocprofiler_sdk.pytest_utils.perfetto_reader import PerfettoReader

import re
import os


def pytest_addoption(parser):

    parser.addoption(
        "--input",
        action="store",
        help="Path to JSON file.",
    )
    parser.addoption(
        "--code-object-input",
        action="store",
        help="Path to code object file.",
    )
    parser.addoption(
        "--output-path",
        action="store",
        help="Output Path.",
    )


@pytest.fixture
def json_data(request):
    filename = request.config.getoption("--input")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))


@pytest.fixture
def output_path(request):
    return request.config.getoption("--output-path")


@pytest.fixture
def code_object_file_path(request):
    file_path = request.config.getoption("--code-object-input")
    # hsa_file_load = re.compile(".*copy.hsaco$")
    code_object_files = {}
    code_object_memory = []
    hsa_memory_load_pattern = "gfx[a-z0-9]+_copy_memory.hsaco"
    for root, dirs, files in os.walk(file_path, topdown=True):
        for file in files:
            filename = os.path.join(root, file)
            if re.search(hsa_memory_load_pattern, filename):
                code_object_memory.append(filename)
    code_object_files["hsa_memory_load"] = code_object_memory
    return code_object_files
