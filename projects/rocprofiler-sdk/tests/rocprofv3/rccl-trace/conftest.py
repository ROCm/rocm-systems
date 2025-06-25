#!/usr/bin/env python3

import csv
import json
import glob
import pytest
from pathlib import Path

from rocprofiler_sdk.pytest_utils.dotdict import dotdict
from rocprofiler_sdk.pytest_utils import collapse_dict_list


def pytest_addoption(parser):
    parser.addoption(
        "--json-input-dir",
        action="store",
        default="rccl-tracing",
        help="Input JSON",
    )
    parser.addoption(
        "--csv-input-dir",
        action="store",
        default="rccl-tracing",
        help="Input CSV",
    )


@pytest.fixture
def json_sp_input_data(request):
    dirname = request.config.getoption("--json-input-dir")
    files = glob.glob(f"{dirname}/rccl-tracing-single-process*.json")

    data_items = []
    for file in files:
        with open(file, "r") as inp:
            data_items.append(dotdict(collapse_dict_list(json.load(inp))))
    return data_items


@pytest.fixture
def csv_sp_input_data(request):
    dirname = request.config.getoption("--csv-input-dir")
    files = glob.glob(f"{dirname}/*trace.csv")

    data_items = []
    for file in files:
        data = []
        with open(file, "r") as inp:
            reader = csv.DictReader(inp)
            for row in reader:
                data.append(row)
        data_items.append(data)

    return data_items
