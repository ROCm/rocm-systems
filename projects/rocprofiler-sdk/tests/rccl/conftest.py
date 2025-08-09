#!/usr/bin/env python3

import json
import glob
import pytest

from rocprofiler_sdk.pytest_utils.dotdict import dotdict
from rocprofiler_sdk.pytest_utils import collapse_dict_list


def pytest_addoption(parser):
    parser.addoption(
        "--input",
        action="store",
        default="rccl-tracing-test.json",
        help="Input JSON",
    )
    parser.addoption(
        "--input-dir",
        action="store",
        help="Input JSON directory",
    )


@pytest.fixture
def input_data(request):
    filename = request.config.getoption("--input")
    data = None
    with open(filename, "r") as inp:
        data = json.load(inp)

    return data
