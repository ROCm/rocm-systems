#!/usr/bin/env python3

import json
import os
import pytest

from rocprofiler_sdk.pytest_utils.dotdict import dotdict


def pytest_addoption(parser):
    parser.addoption(
        "--input",
        action="store",
        default="rocdecode-tracing-test.json",
        help="Input JSON",
    )


@pytest.fixture
def input_data(request):
    filename = request.config.getoption("--input")
    if not os.path.isfile(filename):
        return pytest.skip("rocdecode tracing unavailable")
    with open(filename, "r") as inp:
        return dotdict(json.load(inp))
