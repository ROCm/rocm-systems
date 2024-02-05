#!/usr/bin/env python3

import json
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--input",
        action="store",
        default="counter-collection-test.json",
        help="Input JSON",
    )


@pytest.fixture
def input_data(request):
    filename = request.config.getoption("--input")
    with open(filename, "r") as inp:
        return json.load(inp)
