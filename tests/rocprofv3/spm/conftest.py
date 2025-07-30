#!/usr/bin/env python3

import pytest
import pandas as pd


def pytest_addoption(parser):
    parser.addoption("--pmc", action="store", help="Path to PMC csv file.")
    parser.addoption("--spm", action="store", help="Path to SPM csv file.")


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
