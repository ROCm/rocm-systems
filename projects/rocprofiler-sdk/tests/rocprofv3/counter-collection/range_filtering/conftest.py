#!/usr/bin/env python3

import json
import pytest
import pandas as pd
import re

from rocprofiler_sdk.pytest_utils.dotdict import dotdict
from rocprofiler_sdk.pytest_utils import collapse_dict_list


def pytest_addoption(parser):

    parser.addoption(
        "--input-json-pass1",
        action="store",
        help="Path to JSON file.",
    )

    parser.addoption(
        "--input-json-pass2",
        action="store",
        help="Path to JSON file.",
    )

    parser.addoption(
        "--input-json-pass3",
        action="store",
        help="Path to JSON file.",
    )

    parser.addoption(
        "--json-config",
        action="store",
        help="Path to input JSON file.",
    )


def extract_iteration_list(jobs, pass_):

    kernel_iteration_range = jobs[pass_]["kernel_iteration_range"]
    _range = re.split(r"\[|,|\],|\[|,|\]", kernel_iteration_range)
    _range = list(filter(lambda itr: itr != "", _range))
    range_list = []
    for itr in _range:
        if "-" in itr:
            interval = re.split("-", itr)
            range_list.append(list(range((int)(interval[0]), (int)(interval[1]))))
        else:

            range_list.append(itr)
    return range_list


def process_config(out_file, input_config, pass_):

    ret_dict = {}

    with open(out_file, "r") as inp:
        ret_dict["json_data"] = dotdict(collapse_dict_list(json.load(inp)))

    with open(input_config, "r") as inp:
        jobs = dotdict(collapse_dict_list(json.load(inp)))["jobs"]
        ret_dict["iteration_range"] = extract_iteration_list(jobs, pass_)

    return ret_dict


@pytest.fixture
def input_json_pass1(request):
    out_file = request.config.getoption("--input-json-pass1")
    input_config = request.config.getoption("--json-config")
    return process_config(out_file, input_config, 0)


@pytest.fixture
def input_json_pass2(request):
    out_file = request.config.getoption("--input-json-pass2")
    input_config = request.config.getoption("--json-config")
    return process_config(out_file, input_config, 1)


@pytest.fixture
def input_json_pass3(request):
    out_file = request.config.getoption("--input-json-pass3")
    input_config = request.config.getoption("--json-config")
    return process_config(out_file, input_config, 2)
