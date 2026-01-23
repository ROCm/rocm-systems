#!/usr/bin/env python3

import json
import pandas as pd
import pytest

from rocprofiler_sdk.pytest_utils.dotdict import dotdict
from rocprofiler_sdk.pytest_utils import collapse_dict_list
from rocprofiler_sdk.pytest_utils.rocpd_reader import RocpdReader


def pytest_addoption(parser):
    parser.addoption("--pmc-json", action="store", help="Path to PMC JSON file.")
    parser.addoption("--spm-json", action="store", help="Path to SPM JSON file.")
    parser.addoption("--rocpd-input", action="store", help="Path to rocpd DB file.")
    parser.addoption(
        "--counter-csv", action="store", help="Path to rocpd counter CSV file."
    )


@pytest.fixture
def pmc_json_data(request):
    filename = request.config.getoption("--pmc-json")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))


@pytest.fixture
def spm_json_data(request):
    filename = request.config.getoption("--spm-json")
    with open(filename, "r") as inp:
        return dotdict(collapse_dict_list(json.load(inp)))


@pytest.fixture
def rocpd_data(request):
    filename = request.config.getoption("--rocpd-input")
    return RocpdReader(filename).read()[0]


@pytest.fixture
def counter_csv(request):
    filename = request.config.getoption("--counter-csv")
    with open(filename, "r") as inp:
        return pd.read_csv(inp)
