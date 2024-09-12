#!/usr/bin/env python3

import json
import pytest

from rocprofiler_sdk.pytest_utils.dotdict import dotdict
from rocprofiler_sdk.pytest_utils import collapse_dict_list


def pytest_addoption(parser):
    parser.addoption(
        "--input",
        action="store",
        default="scratch-memory-tracing/out_results.json",
        help="Input JSON",
    )


@pytest.fixture
def input_data(request):
    filename = request.config.getoption("--input")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))
