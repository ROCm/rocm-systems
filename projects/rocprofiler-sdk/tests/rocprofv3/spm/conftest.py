#!/usr/bin/env python3

import pytest
import pandas as pd
import json

from rocprofiler_sdk.pytest_utils.dotdict import dotdict
from rocprofiler_sdk.pytest_utils import collapse_dict_list


def pytest_addoption(parser):
    parser.addoption("--pmc", action="store", help="Path to PMC csv file.")
    parser.addoption("--spm", action="store", help="Path to SPM csv file.")
    parser.addoption("--spm-json", action="store", help="Path to SPM JSON file.")


@pytest.fixture
def pmc_csv(request):
    filename = request.config.getoption("--pmc")
    with open(filename, "r") as inp:
        return pd.read_csv(inp)


@pytest.fixture
def spm_csv(request):
    filename = request.config.getoption("--spm")
    with open(filename, "r") as inp:
        return pd.read_csv(inp)


@pytest.fixture
def json_data(request):
    filename = request.config.getoption("--spm-json")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))
