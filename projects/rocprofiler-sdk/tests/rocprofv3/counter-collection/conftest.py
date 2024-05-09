#!/usr/bin/env python3

import json
import pytest
import csv
import pandas as pd


def pytest_addoption(parser):
    parser.addoption("--input", action="store", help="Path to csv file.")
    parser.addoption(
        "--agent-input",
        action="store",
        help="Path to agent info CSV file.",
    )
    parser.addoption(
        "--counter-input",
        action="store",
        help="Path to counter collection CSV file.",
    )


@pytest.fixture
def input_data(request):
    filename = request.config.getoption("--input")
    if filename:
        with open(filename, "r") as inp:
            return pd.read_csv(filename)
    else:
        return None


@pytest.fixture
def agent_info_input_data(request):
    filename = request.config.getoption("--agent-input")
    data = []
    with open(filename, "r") as inp:
        reader = csv.DictReader(inp)
        for row in reader:
            data.append(row)

    return data


@pytest.fixture
def counter_input_data(request):
    filename = request.config.getoption("--counter-input")
    data = []
    with open(filename, "r") as inp:
        reader = csv.DictReader(inp)
        for row in reader:
            data.append(row)

    return data
