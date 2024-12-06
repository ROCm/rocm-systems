#!/usr/bin/env python3

import json
import pytest

from rocprofiler_sdk.pytest_utils.dotdict import dotdict


def pytest_addoption(parser):
    parser.addoption(
        "--input",
        action="store",
        default="openmp-tools-test.json",
        help="Input JSON",
    )


@pytest.fixture
def input_data(request):
    filename = request.config.getoption("--input")
    with open(filename, "r") as inp:
        return dotdict(json.load(inp))
