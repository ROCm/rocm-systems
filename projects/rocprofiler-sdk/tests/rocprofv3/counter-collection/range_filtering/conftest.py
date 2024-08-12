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


def tokenize(kernel_iteration_range):
    range_str = kernel_iteration_range.replace("[", "").replace("]", "")
    split_list = range_str.split(",")
    _range = []
    for split_string in split_list:
        if "-" in split_string:
            interval = split_string.split("-")
            [
                _range.append(i)
                for i in list(range((int)(interval[0]), (int)(interval[1]) + 1))
            ]
        else:
            _range.append(int(split_string))
    return _range


def extract_iteration_list(jobs, pass_):

    kernel_iteration_range = jobs[pass_]["kernel_iteration_range"].strip()
    return tokenize(kernel_iteration_range)


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
