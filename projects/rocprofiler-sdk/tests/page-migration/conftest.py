#!/usr/bin/env python3

import json
import pytest
from rocprofiler_sdk.pytest_utils.dotdict import dotdict


def pytest_addoption(parser):
    parser.addoption(
        "--input",
        action="store",
        default="page-migration-test.json",
        help="Input JSON",
    )


@pytest.fixture
def input_data(request):
    filename = request.config.getoption("--input")
    data = None
    with open(filename, "r") as inp:
        data = json.load(inp)

    if data["rocprofiler-sdk-json-tool"]["metadata"]["validate_page_migration"] is False:
        return pytest.skip(
            "Skipping test because KFD does not support SVM event reporting"
        )
    return dotdict(data)
